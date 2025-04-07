use std::{fs::OpenOptions, sync::Arc};

use rand::Rng;
use tempfile::{tempdir, TempDir};

use crate::models::{
    buffered_io::{BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    inverted_index::{InvertedIndexNode, InvertedIndexNodeData, InvertedIndexRoot},
    page::VersionedPagepool,
    serializer::inverted::InvertedIndexSerialize,
    types::FileOffset,
    versioning::Hash,
};

fn get_cache(
    dim_bufman: Arc<BufferManager>,
    data_bufmans: Arc<BufferManagerFactory<u8>>,
) -> InvertedIndexCache {
    InvertedIndexCache::new(dim_bufman, data_bufmans, 8)
}

fn setup_test() -> (
    Arc<BufferManager>,
    Arc<BufferManagerFactory<u8>>,
    InvertedIndexCache,
    u64,
    TempDir,
) {
    let dir = tempdir().unwrap();
    let dim_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.as_ref().join("index-tree.idim"))
        .unwrap();
    let dim_bufman = Arc::new(
        BufferManager::new(dim_file, InvertedIndexNode::get_serialized_size(6) as usize).unwrap(),
    );
    let data_bufmans = Arc::new(BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx: &u8| root.join(format!("{}.idat", idx)),
        InvertedIndexNode::get_serialized_size(6) as usize,
    ));
    let cache = get_cache(dim_bufman.clone(), data_bufmans.clone());
    let cursor = dim_bufman.open_cursor().unwrap();
    (dim_bufman, data_bufmans, cache, cursor, dir)
}

fn get_random_versioned_pagepool<const LEN: usize>(
    rng: &mut impl Rng,
    version: Hash,
) -> VersionedPagepool<LEN> {
    let pool = VersionedPagepool::new(version);
    let count = rng.gen_range(20..50);
    add_random_items_to_versioned_pagepool(rng, &pool, count, version);
    pool
}

fn add_random_items_to_versioned_pagepool<const LEN: usize>(
    rng: &mut impl Rng,
    pool: &VersionedPagepool<LEN>,
    count: usize,
    version: Hash,
) {
    for _ in 0..count {
        pool.push(version, rng.gen_range(0..u32::MAX));
    }
}

#[test]
fn test_inverted_index_data_serialization() {
    let mut rng = rand::thread_rng();
    let table = InvertedIndexNodeData::new(6);

    // only even keys
    for i in (0..32).map(|x| x * 2) {
        table
            .map
            .insert(i, get_random_versioned_pagepool(&mut rng, 0.into()));
    }

    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();
    dim_bufman.update_u8_with_cursor(cursor, 6).unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();
    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexNodeData::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    let mut table_list = table.map.to_list();
    let mut deserialized_list = deserialized.map.to_list();

    table_list.sort_by_key(|(k, _)| *k);
    deserialized_list.sort_by_key(|(k, _)| *k);

    assert_eq!(table_list, deserialized_list);
    assert_eq!(table.max_key, deserialized.max_key);
}

#[test]
fn test_inverted_index_data_incremental_serialization_with_updated_values() {
    let mut rng = rand::thread_rng();
    let table = InvertedIndexNodeData::new(6);

    // only even keys
    for i in (0..32).map(|x| x * 2) {
        table
            .map
            .insert(i, get_random_versioned_pagepool(&mut rng, 0.into()));
    }

    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();
    dim_bufman.update_u8_with_cursor(cursor, 6).unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    for i in (0..32).map(|x| x * 2) {
        table.map.with_value(&i, |pool| {
            let mut rng = rand::thread_rng();
            add_random_items_to_versioned_pagepool(&mut rng, pool, 100, 0.into());
        });
    }
    dim_bufman.seek_with_cursor(cursor, offset as u64).unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();
    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexNodeData::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    let mut table_list = table.map.to_list();
    let mut deserialized_list = deserialized.map.to_list();

    table_list.sort_by_key(|(k, _)| *k);
    deserialized_list.sort_by_key(|(k, _)| *k);

    assert_eq!(table_list, deserialized_list);
    assert_eq!(table.max_key, deserialized.max_key);
}

#[test]
fn test_inverted_index_data_incremental_serialization_with_new_entries() {
    let mut rng = rand::thread_rng();
    let table = InvertedIndexNodeData::new(6);

    // only even keys
    for i in (0..32).map(|x| x * 2) {
        table
            .map
            .insert(i, get_random_versioned_pagepool(&mut rng, 0.into()));
    }

    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();
    dim_bufman.update_u8_with_cursor(cursor, 6).unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    for i in (0..32).map(|x| (x * 2) + 1) {
        table
            .map
            .insert(i, get_random_versioned_pagepool(&mut rng, 0.into()));
    }
    dim_bufman.seek_with_cursor(cursor, offset as u64).unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();
    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexNodeData::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    let mut table_list = table.map.to_list();
    let mut deserialized_list = deserialized.map.to_list();

    table_list.sort_by_key(|(k, _)| *k);
    deserialized_list.sort_by_key(|(k, _)| *k);

    assert_eq!(table_list, deserialized_list);
    assert_eq!(table.max_key, deserialized.max_key);
}

#[test]
fn test_inverted_index_node_serialization() {
    let mut rng = rand::thread_rng();
    let inverted_index_node = InvertedIndexNode::new(0, false, 6, FileOffset(0));
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                1.0,
            )
            .unwrap();
    }

    let offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexNode::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(inverted_index_node, deserialized);
}

#[test]
fn test_inverted_index_node_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let inverted_index_node = InvertedIndexNode::new(0, false, 6, FileOffset(0));
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                1.0,
            )
            .unwrap();
    }

    let _offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                1.0,
            )
            .unwrap();
    }

    let offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexNode::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(inverted_index_node, deserialized);
}

#[test]
fn test_inverted_index_node_incremental_serialization_with_multiple_versions() {
    let mut rng = rand::thread_rng();
    let inverted_index_node = InvertedIndexNode::new(0, false, 6, FileOffset(0));
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                1.0,
            )
            .unwrap();
    }

    let _offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                1.0,
            )
            .unwrap();
    }

    let _offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                1.into(),
                1.0,
            )
            .unwrap();
    }

    let _offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                1.into(),
                1.0,
            )
            .unwrap();
    }

    let offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexNode::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(inverted_index_node, deserialized);
}

#[test]
fn test_inverted_index_root_serialization() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let inverted_index = InvertedIndexRoot::new(temp_dir.as_ref().into(), 6, 8).unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();
    inverted_index.cache.dim_bufman.flush().unwrap();
    inverted_index.cache.data_bufmans.flush_all().unwrap();

    let deserialized = InvertedIndexRoot::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}

#[test]
fn test_inverted_root_index_incremental_serialization() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let inverted_index = InvertedIndexRoot::new(temp_dir.as_ref().into(), 6, 8).unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();
    inverted_index.cache.dim_bufman.flush().unwrap();
    inverted_index.cache.data_bufmans.flush_all().unwrap();

    let deserialized = InvertedIndexRoot::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}

#[test]
fn test_inverted_index_root_incremental_serialization_with_multiple_versions() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let inverted_index = InvertedIndexRoot::new(temp_dir.as_ref().into(), 6, 8).unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                1.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                1.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                2.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100000 {
        inverted_index
            .insert(
                rng.gen_range(0..1000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                2.into(),
                1.0,
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();
    inverted_index.cache.dim_bufman.flush().unwrap();
    inverted_index.cache.data_bufmans.flush_all().unwrap();

    let deserialized = InvertedIndexRoot::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}
