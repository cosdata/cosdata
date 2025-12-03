use std::{
    fs::OpenOptions,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use rand::{thread_rng, Rng};
use tempfile::{tempdir, TempDir};

use crate::models::{
    buffered_io::{BufferManager, BufferManagerFactory},
    cache_loader::USVIndexCache,
    serializer::usv::USV_INDEX_DATA_CHUNK_SIZE,
    types::FileOffset,
    usv_index::{USVIndexNode, USVIndexNodeData, USVIndexRoot},
    versioning::VersionNumber,
};

use super::USVIndexSerialize;

fn get_cache(
    dim_bufman: Arc<BufferManager>,
    data_bufmans: Arc<BufferManagerFactory<VersionNumber>>,
    offset_counter: AtomicU32,
    quantization_bits: u8,
) -> USVIndexCache {
    USVIndexCache::new(dim_bufman, data_bufmans, offset_counter, quantization_bits)
}

fn setup_test(
    quantization_bits: u8,
) -> (
    Arc<BufferManager>,
    Arc<BufferManagerFactory<VersionNumber>>,
    USVIndexCache,
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
        BufferManager::new(dim_file, USVIndexNode::get_serialized_size() as usize).unwrap(),
    );
    let data_bufmans = Arc::new(BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, version: &VersionNumber| root.join(format!("{}.idat", **version)),
        USVIndexNode::get_serialized_size() as usize,
    ));
    let cache = get_cache(
        dim_bufman.clone(),
        data_bufmans.clone(),
        AtomicU32::new(0),
        quantization_bits,
    );
    let cursor = dim_bufman.open_cursor().unwrap();
    (dim_bufman, data_bufmans, cache, cursor, dir)
}

#[test]
fn test_usv_index_node_data_serialization() {
    let quantization_bits = 6;
    let mut rng = thread_rng();
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test(quantization_bits);

    let data = USVIndexNodeData::default();

    cache
        .offset_counter
        .fetch_add(USV_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 6, Ordering::Relaxed);

    for vector_id in 0..100 {
        let len = rng.gen_range(10..20);
        for _ in 0..len {
            let quotient = rng.gen_range(100..2000);
            let quantized_value = rng.gen_range(10..50);
            data.insert(
                quotient,
                quantized_value,
                vector_id,
                VersionNumber::from(0),
                || {
                    cache
                        .offset_counter
                        .fetch_add(8 * (1u32 << quantization_bits), Ordering::Relaxed)
                },
            );
        }
    }

    let offset = data
        .serialize(
            &dim_bufman,
            &data_bufmans,
            quantization_bits,
            &cache.offset_counter,
            cursor,
        )
        .unwrap();

    let deserialized = USVIndexNodeData::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(u32::MAX),
        &cache,
    )
    .unwrap();

    assert_eq!(data, deserialized);
}

#[test]
fn test_usv_index_node_serialization() {
    let quantization_bits = 6;
    let mut rng = thread_rng();
    let (dim_bufman, data_bufmans, cache, cursor, _temp_dir) = setup_test(quantization_bits);

    let data = USVIndexNode::new(0, quantization_bits, FileOffset(0));

    cache
        .offset_counter
        .fetch_add(USVIndexNode::get_serialized_size(), Ordering::Relaxed);

    for vector_id in 0..100 {
        let len = rng.gen_range(10..20);
        for _ in 0..len {
            let quotient = rng.gen_range(100..2000);
            let value = rng.gen_range(0.0..1.0);
            data.insert(
                quotient,
                value,
                vector_id,
                &cache,
                VersionNumber::from(0),
                1.0,
            )
            .unwrap();
        }
    }

    let offset = data
        .serialize(
            &dim_bufman,
            &data_bufmans,
            quantization_bits,
            &cache.offset_counter,
            cursor,
        )
        .unwrap();

    let deserialized = USVIndexNode::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(u32::MAX),
        &cache,
    )
    .unwrap();

    assert_eq!(data, deserialized);
}

#[test]
fn end_to_end_test() {
    let mut rng = thread_rng();
    let dir = tempdir().unwrap();
    let root = USVIndexRoot::new(dir.as_ref().into(), 6).unwrap();

    for vector_id in 0..100 {
        let len = rng.gen_range(10..20);

        for _ in 0..len {
            let dim_index = rng.gen();
            let value = rng.gen_range(0.0..1.0);
            root.insert(dim_index, value, vector_id, VersionNumber::from(0), 1.0)
                .unwrap();
        }
    }

    root.serialize().unwrap();

    for vector_id in 100..200 {
        let len = rng.gen_range(10..20);

        for _ in 0..len {
            let dim_index = rng.gen();
            let value = rng.gen_range(0.0..1.0);
            root.insert(dim_index, value, vector_id, VersionNumber::from(1), 1.0)
                .unwrap();
        }
    }

    root.serialize().unwrap();

    root.cache.flush_all().unwrap();

    let deserialized = USVIndexRoot::deserialize(dir.as_ref().into(), 6).unwrap();

    assert_eq!(root, deserialized);
}
