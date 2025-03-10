use std::{fs::OpenOptions, sync::Arc};

use rand::Rng;
use tempfile::{tempdir, TempDir};

use crate::{
    models::{
        buffered_io::{BufferManager, BufferManagerFactory},
        cache_loader::InvertedIndexCache,
        fixedset::VersionedInvertedFixedSetIndex,
        serializer::inverted::InvertedIndexSerialize,
        types::FileOffset,
        versioning::Hash,
    },
    storage::{
        inverted_index_sparse_ann_basic::{
            InvertedIndexSparseAnnBasicTSHashmap, InvertedIndexSparseAnnNodeBasicTSHashmap,
            InvertedIndexSparseAnnNodeBasicTSHashmapData,
        },
        page::{Pagepool, VersionedPagepool},
    },
};

fn get_cache(
    dim_bufman: Arc<BufferManager>,
    data_bufmans: Arc<BufferManagerFactory<u8>>,
) -> Arc<InvertedIndexCache> {
    Arc::new(InvertedIndexCache::new(dim_bufman, data_bufmans, 8))
}

fn setup_test(
    idx: u8,
) -> (
    Arc<BufferManager>,
    Arc<BufferManagerFactory<u8>>,
    Arc<InvertedIndexCache>,
    Arc<BufferManager>,
    u64,
    u64,
    TempDir,
) {
    let dir = tempdir().unwrap();
    let dim_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(dir.as_ref().join("index-tree.idim"))
        .unwrap();
    let dim_bufman = Arc::new(
        BufferManager::new(
            dim_file,
            InvertedIndexSparseAnnNodeBasicTSHashmap::get_serialized_size(6) as usize,
        )
        .unwrap(),
    );
    let data_bufmans = Arc::new(BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx: &u8| root.join(format!("{}.idat", idx)),
        InvertedIndexSparseAnnNodeBasicTSHashmap::get_serialized_size(6) as usize,
    ));
    let cache = get_cache(dim_bufman.clone(), data_bufmans.clone());
    let dim_cursor = dim_bufman.open_cursor().unwrap();
    let data_bufman = data_bufmans.get(idx).unwrap();
    let data_cursor = data_bufman.open_cursor().unwrap();
    (
        dim_bufman,
        data_bufmans,
        cache,
        data_bufman,
        dim_cursor,
        data_cursor,
        dir,
    )
}

fn get_random_pagepool<const LEN: usize>(rng: &mut impl Rng) -> Pagepool<LEN> {
    let mut pool = Pagepool::default();
    let count = rng.gen_range(20..50);
    add_random_items_to_pagepool(rng, &mut pool, count);
    pool
}

fn get_random_versioned_pagepool<const LEN: usize>(
    rng: &mut impl Rng,
    version: Hash,
) -> VersionedPagepool<LEN> {
    let mut pool = VersionedPagepool::new(version);
    let count = rng.gen_range(20..50);
    add_random_items_to_versioned_pagepool(rng, &mut pool, count, version);
    pool
}

fn get_random_versioned_fixedset_index(
    rng: &mut impl Rng,
    version: Hash,
) -> VersionedInvertedFixedSetIndex {
    let sets = VersionedInvertedFixedSetIndex::new(6, version);
    let count = rng.gen_range(20..50);
    add_random_items_to_versioned_fixedset_index(rng, &sets, count, version);
    sets
}

fn add_random_items_to_pagepool<const LEN: usize>(
    rng: &mut impl Rng,
    pool: &mut Pagepool<LEN>,
    count: usize,
) {
    for _ in 0..count {
        pool.push(rng.gen_range(0..u32::MAX));
    }
}

fn add_random_items_to_versioned_pagepool<const LEN: usize>(
    rng: &mut impl Rng,
    pool: &mut VersionedPagepool<LEN>,
    count: usize,
    version: Hash,
) {
    for _ in 0..count {
        pool.push(version, rng.gen_range(0..u32::MAX));
    }
}

fn add_random_items_to_versioned_fixedset_index(
    rng: &mut impl Rng,
    sets: &VersionedInvertedFixedSetIndex,
    count: usize,
    version: Hash,
) {
    for _ in 0..count {
        sets.insert(version, rng.gen_range(0..64), rng.gen_range(0..u32::MAX));
    }
}

#[test]
fn test_pagepool_serialization() {
    let mut rng = rand::thread_rng();
    let page_pool = get_random_pagepool(&mut rng);

    let (dim_bufman, data_bufmans, cache, data_bufman, _dim_cursor, data_cursor, _temp_dir) =
        setup_test(0);
    let offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    data_bufman.close_cursor(data_cursor).unwrap();

    let deserialized =
        Pagepool::<10>::deserialize(&dim_bufman, &data_bufmans, FileOffset(offset), 0, 8, &cache)
            .unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_pagepool_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let mut page_pool = get_random_pagepool(&mut rng);

    let (dim_bufman, data_bufmans, cache, data_bufman, _dim_cursor, data_cursor, _temp_dir) =
        setup_test(0);
    let _offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_pagepool(&mut rng, &mut page_pool, 100);

    let offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();
    data_bufman.close_cursor(data_cursor).unwrap();

    let deserialized =
        Pagepool::<10>::deserialize(&dim_bufman, &data_bufmans, FileOffset(offset), 0, 8, &cache)
            .unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_versioned_pagepool_serialization() {
    let mut rng = rand::thread_rng();
    let page_pool: VersionedPagepool<10> = get_random_versioned_pagepool(&mut rng, 0.into());

    let (dim_bufman, data_bufmans, cache, data_bufman, _dim_cursor, data_cursor, _temp_dir) =
        setup_test(0);
    let offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    data_bufman.close_cursor(data_cursor).unwrap();

    let deserialized = VersionedPagepool::<10>::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_versioned_pagepool_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let mut page_pool = get_random_versioned_pagepool(&mut rng, 0.into());

    let (dim_bufman, data_bufmans, cache, data_bufman, _dim_cursor, data_cursor, _temp_dir) =
        setup_test(0);
    let _offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_versioned_pagepool(&mut rng, &mut page_pool, 100, 0.into());

    let offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();
    data_bufman.close_cursor(data_cursor).unwrap();

    let deserialized = VersionedPagepool::<10>::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_versioned_pagepool_incremental_serialization2() {
    let mut rng = rand::thread_rng();
    let mut page_pool = get_random_versioned_pagepool(&mut rng, 0.into());

    let (dim_bufman, data_bufmans, cache, data_bufman, _dim_cursor, data_cursor, _temp_dir) =
        setup_test(0);
    let _offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_versioned_pagepool(&mut rng, &mut page_pool, 100, 0.into());

    let _offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_versioned_pagepool(&mut rng, &mut page_pool, 100, 1.into());

    let _offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_versioned_pagepool(&mut rng, &mut page_pool, 100, 1.into());

    let offset = page_pool
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();
    data_bufman.close_cursor(data_cursor).unwrap();

    let deserialized = VersionedPagepool::<10>::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_inverted_index_data_serialization() {
    let mut rng = rand::thread_rng();
    let table = InvertedIndexSparseAnnNodeBasicTSHashmapData::new(6);

    // only even keys
    for i in (0..32).map(|x| x * 2) {
        table
            .map
            .insert(i, get_random_versioned_pagepool(&mut rng, 0.into()));
    }

    let (dim_bufman, data_bufmans, cache, _data_bufman, dim_cursor, _data_cursor, _temp_dir) =
        setup_test(0);
    dim_bufman.update_u8_with_cursor(dim_cursor, 6).unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
        .unwrap();
    dim_bufman.close_cursor(dim_cursor).unwrap();

    let deserialized = InvertedIndexSparseAnnNodeBasicTSHashmapData::deserialize(
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
    let table = InvertedIndexSparseAnnNodeBasicTSHashmapData::new(6);

    // only even keys
    for i in (0..32).map(|x| x * 2) {
        table
            .map
            .insert(i, get_random_versioned_pagepool(&mut rng, 0.into()));
    }

    let (dim_bufman, data_bufmans, cache, _data_bufman, dim_cursor, _data_cursor, _temp_dir) =
        setup_test(0.into());
    dim_bufman.update_u8_with_cursor(dim_cursor, 6).unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
        .unwrap();

    for i in (0..32).map(|x| x * 2) {
        table.map.mutate(i, |pool| {
            let mut pool = pool.unwrap();
            add_random_items_to_versioned_pagepool(&mut rng, &mut pool, 100, 0.into());
            Some(pool)
        });
    }
    dim_bufman
        .seek_with_cursor(dim_cursor, offset as u64)
        .unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
        .unwrap();
    dim_bufman.close_cursor(dim_cursor).unwrap();

    let deserialized = InvertedIndexSparseAnnNodeBasicTSHashmapData::deserialize(
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
    let table = InvertedIndexSparseAnnNodeBasicTSHashmapData::new(6);

    // only even keys
    for i in (0..32).map(|x| x * 2) {
        table
            .map
            .insert(i, get_random_versioned_pagepool(&mut rng, 0.into()));
    }

    let (dim_bufman, data_bufmans, cache, _data_bufman, dim_cursor, _data_cursor, _temp_dir) =
        setup_test(0.into());
    dim_bufman.update_u8_with_cursor(dim_cursor, 6).unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
        .unwrap();

    for i in (0..32).map(|x| (x * 2) + 1) {
        table
            .map
            .insert(i, get_random_versioned_pagepool(&mut rng, 0.into()));
    }
    dim_bufman
        .seek_with_cursor(dim_cursor, offset as u64)
        .unwrap();
    let offset = table
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
        .unwrap();
    dim_bufman.close_cursor(dim_cursor).unwrap();

    let deserialized = InvertedIndexSparseAnnNodeBasicTSHashmapData::deserialize(
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
fn test_fixedset_serialization() {
    let mut rng = rand::thread_rng();
    let (dim_bufman, data_bufmans, cache, data_bufman, dim_cursor, data_cursor, _temp_dir) =
        setup_test(0.into());
    let sets = get_random_versioned_fixedset_index(&mut rng, 0.into());

    let offset = sets
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    data_bufman.close_cursor(dim_cursor).unwrap();

    let deserialized = VersionedInvertedFixedSetIndex::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(sets, deserialized);
}

#[test]
fn test_fixedset_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let (dim_bufman, data_bufmans, cache, data_bufman, _dim_cursor, data_cursor, _temp_dir) =
        setup_test(0.into());
    let sets = get_random_versioned_fixedset_index(&mut rng, 0.into());

    let offset = sets
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_versioned_fixedset_index(&mut rng, &sets, 20, 0.into());

    data_bufman
        .seek_with_cursor(data_cursor, offset as u64)
        .unwrap();

    let offset = sets
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    data_bufman.close_cursor(data_cursor).unwrap();

    let deserialized = VersionedInvertedFixedSetIndex::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(sets, deserialized);
}

#[test]
fn test_fixedset_incremental_serialization_with_multiple_versions() {
    let mut rng = rand::thread_rng();
    let (dim_bufman, data_bufmans, cache, data_bufman, _dim_cursor, data_cursor, _temp_dir) =
        setup_test(0.into());
    let sets = get_random_versioned_fixedset_index(&mut rng, 0.into());

    let offset = sets
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_versioned_fixedset_index(&mut rng, &sets, 20, 0.into());

    data_bufman
        .seek_with_cursor(data_cursor, offset as u64)
        .unwrap();

    let offset = sets
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_versioned_fixedset_index(&mut rng, &sets, 20, 1.into());

    data_bufman
        .seek_with_cursor(data_cursor, offset as u64)
        .unwrap();

    let offset = sets
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    add_random_items_to_versioned_fixedset_index(&mut rng, &sets, 20, 1.into());

    data_bufman
        .seek_with_cursor(data_cursor, offset as u64)
        .unwrap();

    let offset = sets
        .serialize(&dim_bufman, &data_bufmans, 0, 8, data_cursor)
        .unwrap();

    data_bufman.close_cursor(data_cursor).unwrap();

    let deserialized = VersionedInvertedFixedSetIndex::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(sets, deserialized);
}

#[test]
fn test_inverted_index_node_serialization() {
    let mut rng = rand::thread_rng();
    let inverted_index_node =
        InvertedIndexSparseAnnNodeBasicTSHashmap::new(0, false, 6, 0.into(), FileOffset(0));
    let (dim_bufman, data_bufmans, cache, _data_bufman, dim_cursor, _data_cursor, _temp_dir) =
        setup_test(0.into());

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
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
        .unwrap();

    dim_bufman.close_cursor(dim_cursor).unwrap();

    let deserialized = InvertedIndexSparseAnnNodeBasicTSHashmap::deserialize(
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
    let inverted_index_node =
        InvertedIndexSparseAnnNodeBasicTSHashmap::new(0, false, 6, 0.into(), FileOffset(0));
    let (dim_bufman, data_bufmans, cache, _data_bufman, dim_cursor, _data_cursor, _temp_dir) =
        setup_test(0.into());

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
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
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
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
        .unwrap();

    dim_bufman.close_cursor(dim_cursor).unwrap();

    let deserialized = InvertedIndexSparseAnnNodeBasicTSHashmap::deserialize(
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
    let inverted_index_node =
        InvertedIndexSparseAnnNodeBasicTSHashmap::new(0, false, 6, 0.into(), FileOffset(0));
    let (dim_bufman, data_bufmans, cache, _data_bufman, dim_cursor, _data_cursor, _temp_dir) =
        setup_test(0.into());

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
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
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
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
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
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
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
        .serialize(&dim_bufman, &data_bufmans, 0, 8, dim_cursor)
        .unwrap();

    dim_bufman.close_cursor(dim_cursor).unwrap();

    let deserialized = InvertedIndexSparseAnnNodeBasicTSHashmap::deserialize(
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
fn test_inverted_index_serialization() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let inverted_index =
        InvertedIndexSparseAnnBasicTSHashmap::new(temp_dir.as_ref().into(), 6, 0.into(), 8)
            .unwrap();

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

    let deserialized =
        InvertedIndexSparseAnnBasicTSHashmap::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}

#[test]
fn test_inverted_index_incremental_serialization() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let inverted_index =
        InvertedIndexSparseAnnBasicTSHashmap::new(temp_dir.as_ref().into(), 6, 0.into(), 8)
            .unwrap();

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

    let deserialized =
        InvertedIndexSparseAnnBasicTSHashmap::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}

#[test]
fn test_inverted_index_incremental_serialization_with_multiple_versions() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let inverted_index =
        InvertedIndexSparseAnnBasicTSHashmap::new(temp_dir.as_ref().into(), 6, 0.into(), 8)
            .unwrap();

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

    let deserialized =
        InvertedIndexSparseAnnBasicTSHashmap::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}
