use std::{
    fs::OpenOptions,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, RwLock,
    },
};

use rand::Rng;
use tempfile::{tempdir, TempDir};

use crate::models::{
    buffered_io::{BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexIDFCache,
    inverted_index::InvertedIndexNode,
    inverted_index_idf::{
        DocumentIDList, InvertedIndexIDFNode, InvertedIndexIDFNodeData, InvertedIndexIDFRoot,
        TermFrequencyMap, TermInfo,
    },
    page::VersionedPagepool,
    serializer::inverted_idf::INVERTED_INDEX_DATA_CHUNK_SIZE,
    types::FileOffset,
    versioning::Hash,
};

use super::InvertedIndexIDFSerialize;

fn get_cache(
    dim_bufman: Arc<BufferManager>,
    data_bufmans: Arc<BufferManagerFactory<u8>>,
) -> InvertedIndexIDFCache {
    InvertedIndexIDFCache::new(dim_bufman, data_bufmans, AtomicU32::new(0), 8, 6)
}

fn setup_test() -> (
    Arc<BufferManager>,
    Arc<BufferManagerFactory<u8>>,
    InvertedIndexIDFCache,
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
        BufferManager::new(
            dim_file,
            InvertedIndexIDFNode::get_serialized_size() as usize,
        )
        .unwrap(),
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

fn get_random_term_info(rng: &mut impl Rng, version: Hash) -> TermInfo {
    let term = TermInfo::new(rng.gen_range(0..4234));
    let count = rng.gen_range(20..50);
    add_random_items_to_term_info(rng, &term, count, version);
    term
}

fn add_random_items_to_term_info(rng: &mut impl Rng, term: &TermInfo, count: usize, version: Hash) {
    for _ in 0..count {
        let idx = rng.gen_range(0..64);
        term.frequency_map.modify_or_insert(
            idx,
            |pool| add_random_items_to_versioned_pagepool(rng, pool, count, version),
            || {
                let mut rng = rand::thread_rng();
                get_random_versioned_pagepool(&mut rng, version)
            },
        );
    }
    term.documents_count
        .fetch_add(count as u32, Ordering::Relaxed);
}

fn get_random_inverted_index_data(rng: &mut impl Rng, version: Hash) -> InvertedIndexIDFNodeData {
    let data = InvertedIndexIDFNodeData::new();
    let count = rng.gen_range(1000..2000);
    add_random_items_to_inverted_index_data(rng, &data, count, version);
    data
}

fn add_random_items_to_inverted_index_data(
    rng: &mut impl Rng,
    data: &InvertedIndexIDFNodeData,
    count: usize,
    version: Hash,
) {
    for _ in 0..count {
        let quotient = rng.gen_range(0..2000);
        let quantized_value = rng.gen_range(0..64);
        let document_id = rng.gen_range(0..u32::MAX);
        data.map.modify_or_insert(
            quotient,
            |v| {
                // Update or insert quantized value in inner map
                v.frequency_map.modify_or_insert(
                    quantized_value,
                    |document_id_list| {
                        document_id_list.push(version, document_id);
                    },
                    || {
                        let pool = DocumentIDList::new(version);
                        pool.push(version, document_id);
                        pool
                    },
                );

                v.documents_count.fetch_add(1, Ordering::Relaxed);
            },
            || {
                // Create new inner map if quotient not found
                let frequency_map = TermFrequencyMap::new(8);
                let sequence_idx = data.map_len.fetch_add(1, Ordering::Relaxed);
                let pool = DocumentIDList::new(version);
                pool.push(version, document_id);
                frequency_map.insert(quantized_value, pool);
                Arc::new(TermInfo {
                    serialized_at: RwLock::new(None),
                    frequency_map,
                    sequence_idx,
                    documents_count: AtomicU32::new(1),
                })
            },
        );
    }
}

#[test]
fn test_term_info_serialization() {
    let (dim_bufman, data_bufmans, cache, cursor, _temp) = setup_test();
    let mut rng = rand::thread_rng();

    let term_info = get_random_term_info(&mut rng, 0.into());

    let offset_counter = AtomicU32::new(0);

    let offset = term_info
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    let mut deserialized = TermInfo::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        6,
        0,
        8,
        &cache,
    )
    .unwrap();

    deserialized.sequence_idx = term_info.sequence_idx;

    assert_eq!(term_info, deserialized);
    assert_eq!(offset_counter.load(Ordering::Relaxed), 260);
}

#[test]
fn test_term_info_incremental_serialization_with_updated_values() {
    let (dim_bufman, data_bufmans, cache, cursor, _temp) = setup_test();
    let mut rng = rand::thread_rng();

    let term_info = get_random_term_info(&mut rng, 0.into());

    let offset_counter = AtomicU32::new(0);

    let _offset = term_info
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    let count = rng.gen_range(100..200);
    add_random_items_to_term_info(&mut rng, &term_info, count, 0.into());

    let offset = term_info
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    let mut deserialized = TermInfo::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        6,
        0,
        8,
        &cache,
    )
    .unwrap();

    deserialized.sequence_idx = term_info.sequence_idx;

    assert_eq!(term_info, deserialized);
    assert_eq!(offset_counter.load(Ordering::Relaxed), 260);
}

#[test]
fn test_inverted_index_data_serialization() {
    let (dim_bufman, data_bufmans, cache, cursor, _temp) = setup_test();
    let mut rng = rand::thread_rng();

    let data = get_random_inverted_index_data(&mut rng, 0.into());

    let offset_counter = AtomicU32::new(INVERTED_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 6);

    let offset = data
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexIDFNodeData::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        6,
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(data, deserialized);
}

#[test]
fn test_inverted_index_data_incremental_serialization() {
    let (dim_bufman, data_bufmans, cache, cursor, _temp) = setup_test();
    let mut rng = rand::thread_rng();

    let data = get_random_inverted_index_data(&mut rng, 0.into());

    let offset_counter = AtomicU32::new(INVERTED_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 6);

    let offset = data
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    let count = rng.gen_range(1000..2000);

    add_random_items_to_inverted_index_data(&mut rng, &data, count, 0.into());

    dim_bufman.seek_with_cursor(cursor, offset as u64).unwrap();

    let offset = data
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexIDFNodeData::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        6,
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(data, deserialized);
}

#[test]
fn test_inverted_index_node_serialization() {
    let mut rng = rand::thread_rng();
    let inverted_index_node = InvertedIndexIDFNode::new(0, false, 6, FileOffset(0));
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();
    let most_common_term_documents_count = AtomicU32::new(0);
    let least_common_term_documents_count = AtomicU64::new(1 << 32);

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                &most_common_term_documents_count,
                &least_common_term_documents_count,
            )
            .unwrap();
    }

    let offset_counter = AtomicU32::new(InvertedIndexIDFNode::get_serialized_size());

    let offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexIDFNode::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        6,
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
    let inverted_index_node = InvertedIndexIDFNode::new(0, false, 6, FileOffset(0));
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();
    let most_common_term_documents_count = AtomicU32::new(0);
    let least_common_term_documents_count = AtomicU64::new(1 << 32);

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                &most_common_term_documents_count,
                &least_common_term_documents_count,
            )
            .unwrap();
    }

    let offset_counter = AtomicU32::new(InvertedIndexIDFNode::get_serialized_size());

    let _offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                &most_common_term_documents_count,
                &least_common_term_documents_count,
            )
            .unwrap();
    }

    let offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexIDFNode::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        6,
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
    let inverted_index_node = InvertedIndexIDFNode::new(0, false, 6, FileOffset(0));
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();
    let most_common_term_documents_count = AtomicU32::new(0);
    let least_common_term_documents_count = AtomicU64::new(1 << 32);

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                &most_common_term_documents_count,
                &least_common_term_documents_count,
            )
            .unwrap();
    }

    let offset_counter = AtomicU32::new(InvertedIndexIDFNode::get_serialized_size());

    let _offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
                &most_common_term_documents_count,
                &least_common_term_documents_count,
            )
            .unwrap();
    }

    let _offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                1.into(),
                &most_common_term_documents_count,
                &least_common_term_documents_count,
            )
            .unwrap();
    }

    let _offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        inverted_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                1.into(),
                &most_common_term_documents_count,
                &least_common_term_documents_count,
            )
            .unwrap();
    }

    let offset = inverted_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 6, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = InvertedIndexIDFNode::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        6,
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
    let inverted_index = InvertedIndexIDFRoot::new(temp_dir.as_ref().into(), 6, 8).unwrap();

    for _ in 0..1000 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();
    inverted_index.cache.dim_bufman.flush().unwrap();
    inverted_index.cache.data_bufmans.flush_all().unwrap();

    let deserialized = InvertedIndexIDFRoot::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}

#[test]
fn test_inverted_root_index_incremental_serialization() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let inverted_index = InvertedIndexIDFRoot::new(temp_dir.as_ref().into(), 6, 8).unwrap();

    for _ in 0..1000 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..1000 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();
    inverted_index.cache.dim_bufman.flush().unwrap();
    inverted_index.cache.data_bufmans.flush_all().unwrap();

    let deserialized = InvertedIndexIDFRoot::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}

#[test]
fn test_inverted_index_root_incremental_serialization_with_multiple_versions() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let inverted_index = InvertedIndexIDFRoot::new(temp_dir.as_ref().into(), 6, 8).unwrap();

    for _ in 0..100 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                1.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                1.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                2.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    for _ in 0..100 {
        inverted_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                2.into(),
            )
            .unwrap();
    }

    inverted_index.serialize().unwrap();

    inverted_index.cache.dim_bufman.flush().unwrap();
    inverted_index.cache.data_bufmans.flush_all().unwrap();

    let deserialized = InvertedIndexIDFRoot::deserialize(temp_dir.as_ref().into(), 6, 8).unwrap();

    assert_eq!(inverted_index, deserialized);
}
