use std::{
    fs::OpenOptions,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, RwLock,
    },
};

use rand::Rng;
use tempfile::{tempdir, TempDir};

use crate::models::{
    buffered_io::{BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    serializer::tf_idf::TF_IDF_INDEX_DATA_CHUNK_SIZE,
    tf_idf_index::{TFIDFIndexNode, TFIDFIndexNodeData, TFIDFIndexRoot, TermInfo},
    types::FileOffset,
    versioned_vec::VersionedVec,
    versioning::VersionNumber,
};

use super::TFIDFIndexSerialize;

fn get_cache(
    dim_bufman: Arc<BufferManager>,
    data_bufmans: Arc<BufferManagerFactory<u8>>,
) -> TFIDFIndexCache {
    TFIDFIndexCache::new(dim_bufman, data_bufmans, AtomicU32::new(0), 8)
}

fn setup_test() -> (
    Arc<BufferManager>,
    Arc<BufferManagerFactory<u8>>,
    TFIDFIndexCache,
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
        BufferManager::new(dim_file, TFIDFIndexNode::get_serialized_size() as usize).unwrap(),
    );
    let data_bufmans = Arc::new(BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx: &u8| root.join(format!("{}.idat", idx)),
        TFIDFIndexNode::get_serialized_size() as usize,
    ));
    let cache = get_cache(dim_bufman.clone(), data_bufmans.clone());
    let cursor = dim_bufman.open_cursor().unwrap();
    (dim_bufman, data_bufmans, cache, cursor, dir)
}

fn get_random_term_info(rng: &mut impl Rng, version: VersionNumber) -> TermInfo {
    let term = TermInfo::new(rng.gen_range(0..4234), version);
    let count = rng.gen_range(20..50);
    add_random_items_to_term_info(rng, &term, count, version);
    term
}

fn add_random_items_to_term_info(
    rng: &mut impl Rng,
    term: &TermInfo,
    count: usize,
    version: VersionNumber,
) {
    for _ in 0..count {
        term.documents.write().unwrap().push(
            version,
            (rng.gen_range(0..u32::MAX), rng.gen_range(0.0..1.0)),
        );
    }
}

fn get_random_tf_idf_index_data(rng: &mut impl Rng, version: VersionNumber) -> TFIDFIndexNodeData {
    let data = TFIDFIndexNodeData::new();
    let count = rng.gen_range(1000..2000);
    add_random_items_to_tf_idf_index_data(rng, &data, count, version);
    data
}

fn add_random_items_to_tf_idf_index_data(
    rng: &mut impl Rng,
    data: &TFIDFIndexNodeData,
    count: usize,
    version: VersionNumber,
) {
    for _ in 0..count {
        let quotient = rng.gen_range(0..2000);
        let value = rng.gen_range(0.0..1.0);
        let document_id = rng.gen_range(0..u32::MAX);
        data.map.modify_or_insert(
            quotient,
            |term| {
                term.documents
                    .write()
                    .unwrap()
                    .push(version, (document_id, value));
            },
            || {
                // Create new inner map if quotient not found
                let mut documents = VersionedVec::new(version);
                let sequence_idx = data.map_len.fetch_add(1, Ordering::Relaxed);
                documents.push(version, (document_id, value));
                Arc::new(TermInfo {
                    documents: RwLock::new(documents),
                    sequence_idx,
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
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    let mut deserialized =
        TermInfo::deserialize(&dim_bufman, &data_bufmans, FileOffset(offset), 0, 8, &cache)
            .unwrap();

    deserialized.sequence_idx = term_info.sequence_idx;

    assert_eq!(term_info, deserialized);
    assert_eq!(offset_counter.load(Ordering::Relaxed), 0);
}

#[test]
fn test_term_info_incremental_serialization_with_updated_values() {
    let (dim_bufman, data_bufmans, cache, cursor, _temp) = setup_test();
    let mut rng = rand::thread_rng();

    let term_info = get_random_term_info(&mut rng, 0.into());

    let offset_counter = AtomicU32::new(0);

    let _offset = term_info
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    let count = rng.gen_range(100..200);
    add_random_items_to_term_info(&mut rng, &term_info, count, 1.into());

    let offset = term_info
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    let mut deserialized =
        TermInfo::deserialize(&dim_bufman, &data_bufmans, FileOffset(offset), 0, 8, &cache)
            .unwrap();

    deserialized.sequence_idx = term_info.sequence_idx;

    assert_eq!(term_info, deserialized);
    assert_eq!(offset_counter.load(Ordering::Relaxed), 0);
}

#[test]
fn test_tf_idf_index_data_serialization() {
    let (dim_bufman, data_bufmans, cache, cursor, _temp) = setup_test();
    let mut rng = rand::thread_rng();

    let data = get_random_tf_idf_index_data(&mut rng, 0.into());

    let offset_counter = AtomicU32::new(TF_IDF_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 6);

    let offset = data
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = TFIDFIndexNodeData::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(data, deserialized);
}

#[test]
fn test_tf_idf_index_data_incremental_serialization() {
    let (dim_bufman, data_bufmans, cache, cursor, _temp) = setup_test();
    let mut rng = rand::thread_rng();

    let data = get_random_tf_idf_index_data(&mut rng, 0.into());

    let offset_counter = AtomicU32::new(TF_IDF_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 6);

    let offset = data
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    let count = rng.gen_range(1000..2000);

    add_random_items_to_tf_idf_index_data(&mut rng, &data, count, 1.into());

    dim_bufman.seek_with_cursor(cursor, offset as u64).unwrap();

    let offset = data
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = TFIDFIndexNodeData::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        0,
        8,
        &cache,
    )
    .unwrap();

    assert_eq!(data, deserialized);
}

#[test]
fn test_tf_idf_index_node_serialization() {
    let mut rng = rand::thread_rng();
    let tf_idf_index_node = TFIDFIndexNode::new(0, FileOffset(0));
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();

    for _ in 0..300 {
        tf_idf_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
            )
            .unwrap();
    }

    let offset_counter = AtomicU32::new(TFIDFIndexNode::get_serialized_size());

    let offset = tf_idf_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized =
        TFIDFIndexNode::deserialize(&dim_bufman, &data_bufmans, FileOffset(offset), 0, 8, &cache)
            .unwrap();

    assert_eq!(tf_idf_index_node, deserialized);
}

#[test]
fn test_tf_idf_index_node_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let tf_idf_index_node = TFIDFIndexNode::new(0, FileOffset(0));
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test();

    for _ in 0..300 {
        tf_idf_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                0.into(),
            )
            .unwrap();
    }

    let offset_counter = AtomicU32::new(TFIDFIndexNode::get_serialized_size());

    let _offset = tf_idf_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        tf_idf_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                1.into(),
            )
            .unwrap();
    }

    let _offset = tf_idf_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        tf_idf_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                2.into(),
            )
            .unwrap();
    }

    let _offset = tf_idf_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    for _ in 0..300 {
        tf_idf_index_node
            .insert(
                rng.gen_range(0..10000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                &cache,
                3.into(),
            )
            .unwrap();
    }

    let offset = tf_idf_index_node
        .serialize(&dim_bufman, &data_bufmans, &offset_counter, 0, 8, cursor)
        .unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized =
        TFIDFIndexNode::deserialize(&dim_bufman, &data_bufmans, FileOffset(offset), 0, 8, &cache)
            .unwrap();

    assert_eq!(tf_idf_index_node, deserialized);
}

#[test]
fn test_tf_idf_index_root_serialization() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let tf_idf_index = TFIDFIndexRoot::new(temp_dir.as_ref().into(), 8).unwrap();

    for _ in 0..1000 {
        tf_idf_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
            )
            .unwrap();
    }

    tf_idf_index.serialize().unwrap();
    tf_idf_index.cache.dim_bufman.flush().unwrap();
    tf_idf_index.cache.data_bufmans.flush_all().unwrap();

    let deserialized = TFIDFIndexRoot::deserialize(temp_dir.as_ref().into(), 8).unwrap();

    assert_eq!(tf_idf_index, deserialized);
}

#[test]
fn test_tf_idf_index_root_incremental_serialization() {
    let temp_dir = tempdir().unwrap();
    let mut rng = rand::thread_rng();
    let tf_idf_index = TFIDFIndexRoot::new(temp_dir.as_ref().into(), 8).unwrap();

    for _ in 0..100 {
        tf_idf_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                0.into(),
            )
            .unwrap();
    }

    tf_idf_index.serialize().unwrap();

    for _ in 0..100 {
        tf_idf_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                1.into(),
            )
            .unwrap();
    }

    tf_idf_index.serialize().unwrap();

    for _ in 0..100 {
        tf_idf_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                2.into(),
            )
            .unwrap();
    }

    tf_idf_index.serialize().unwrap();

    for _ in 0..100 {
        tf_idf_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                3.into(),
            )
            .unwrap();
    }

    tf_idf_index.serialize().unwrap();

    for _ in 0..100 {
        tf_idf_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                4.into(),
            )
            .unwrap();
    }

    tf_idf_index.serialize().unwrap();

    for _ in 0..100 {
        tf_idf_index
            .insert(
                rng.gen_range(0..10000000),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0..u32::MAX),
                5.into(),
            )
            .unwrap();
    }

    tf_idf_index.serialize().unwrap();

    tf_idf_index.cache.dim_bufman.flush().unwrap();
    tf_idf_index.cache.data_bufmans.flush_all().unwrap();

    let deserialized = TFIDFIndexRoot::deserialize(temp_dir.as_ref().into(), 8).unwrap();

    assert_eq!(tf_idf_index, deserialized);
}
