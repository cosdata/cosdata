use std::fs::OpenOptions;
use std::sync::Arc;

use crate::indexes::hnsw::offset_counter::IndexFileId;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::collection::RawVectorEmbedding;
use crate::models::inverted_index::InvertedIndexNode;
use crate::models::serializer::*;
use crate::models::tf_idf_index::VersionedVec;
use crate::models::tree_map::QuotientsMap;
use crate::models::tree_map::TreeMap;
use crate::models::tree_map::TreeMapVec;
use crate::models::types::*;
use crate::models::versioning::VersionNumber;
use crate::storage::Storage;
use half::f16;
use rand::distributions::Alphanumeric;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;
use tempfile::tempdir;
use tempfile::TempDir;

#[allow(clippy::type_complexity)]
fn setup_test() -> (Arc<BufferManager>, u64, TempDir) {
    let dir = tempdir().unwrap();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.as_ref().join("test"))
        .unwrap();
    let bufman = Arc::new(
        BufferManager::new(file, InvertedIndexNode::get_serialized_size(6) as usize).unwrap(),
    );
    let cursor = bufman.open_cursor().unwrap();
    (bufman, cursor, dir)
}

#[test]
fn test_storage_serialization() {
    let storages = [
        Storage::UnsignedByte {
            mag: 10.0,
            quant_vec: vec![0, 1, 4],
        },
        Storage::SubByte {
            mag: 34.0,
            quant_vec: vec![vec![55, 35], vec![56, 23]],
            resolution: 2,
        },
        Storage::HalfPrecisionFP {
            mag: 4234.34,
            quant_vec: vec![f16::from_f32(534.324), f16::from_f32(6453.3)],
        },
    ];
    let tempdir = TempDir::new().unwrap();
    let bufmans = BufferManagerFactory::new(
        tempdir.as_ref().into(),
        |root, ver: &IndexFileId| root.join(format!("{}.index", **ver)),
        8192,
    );

    for (file_id, storage) in storages.into_iter().enumerate() {
        let version_id = IndexFileId::from(file_id as u32);
        let bufman = bufmans.get(version_id).unwrap();
        let cursor = bufman.open_cursor().unwrap();
        let offset = SimpleSerialize::serialize(&storage, &bufman, cursor).unwrap();
        let deserialized: Storage =
            SimpleSerialize::deserialize(&bufman, FileOffset(offset)).unwrap();

        assert_eq!(deserialized, storage);
    }
}

fn get_random_versioned_vec<T>(rng: &mut impl Rng, version: VersionNumber) -> VersionedVec<T>
where
    Standard: Distribution<T>,
{
    let mut pool = VersionedVec::new(version);
    let count = rng.gen_range(20..50);
    add_random_items_to_versioned_vec(rng, &mut pool, count, version);
    pool
}

fn add_random_items_to_versioned_vec<T>(
    rng: &mut impl Rng,
    pool: &mut VersionedVec<T>,
    count: usize,
    version: VersionNumber,
) where
    Standard: Distribution<T>,
{
    for _ in 0..count {
        pool.push(version, rng.gen());
    }
}

#[test]
fn test_versioned_vec_serialization() {
    let mut rng = rand::thread_rng();
    let vec: VersionedVec<u32> = get_random_versioned_vec(&mut rng, 0.into());

    let (bufman, cursor, _temp_dir) = setup_test();
    let offset = vec.serialize(&bufman, cursor).unwrap();

    bufman.close_cursor(cursor).unwrap();

    let deserialized = VersionedVec::<u32>::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(vec, deserialized);
}

#[test]
fn test_versioned_vec_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let mut vec: VersionedVec<u32> = get_random_versioned_vec(&mut rng, 0.into());

    let (bufman, cursor, _temp_dir) = setup_test();
    let _offset = vec.serialize(&bufman, cursor).unwrap();

    add_random_items_to_versioned_vec(&mut rng, &mut vec, 100, 1.into());

    let offset = vec.serialize(&bufman, cursor).unwrap();
    bufman.close_cursor(cursor).unwrap();

    let deserialized = VersionedVec::<u32>::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(vec, deserialized);
}

#[test]
fn test_quotients_serialization() {
    let (bufman, cursor, _temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let quotients_map = QuotientsMap::default();
    for _ in 0..10000 {
        quotients_map.insert(
            0.into(),
            rng.gen_range(0..u64::MAX),
            rng.gen_range(0..u16::MAX),
        );
    }
    let offset = quotients_map.serialize(&bufman, cursor).unwrap();
    let deserialized = QuotientsMap::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(quotients_map, deserialized);
}

#[test]
fn test_quotients_incremental_serialization() {
    let (bufman, cursor, _temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let quotients_map = QuotientsMap::default();
    for i in 0..1000 {
        quotients_map.insert(0.into(), i, rng.gen_range(0..u16::MAX));
    }
    let _offset = quotients_map.serialize(&bufman, cursor).unwrap();
    for i in 1000..2001 {
        quotients_map.insert(0.into(), i, rng.gen_range(0..u16::MAX));
    }
    let _offset = quotients_map.serialize(&bufman, cursor).unwrap();
    for i in 2001..3000 {
        quotients_map.insert(0.into(), i, rng.gen_range(0..u16::MAX));
    }
    let offset = quotients_map.serialize(&bufman, cursor).unwrap();
    let deserialized = QuotientsMap::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(quotients_map, deserialized);
}

#[test]
fn test_tree_map_serialization() {
    let dir = tempdir().unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );
    let mut rng = rand::thread_rng();
    let map = TreeMap::new(bufmans);

    for i in 0..1000 {
        map.insert(0.into(), &i, rng.gen_range(0..u16::MAX));
    }

    // edge case
    map.insert(0.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize(8).unwrap();

    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );

    let deserialized = TreeMap::<u64, u16>::deserialize(bufmans, 8).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_map_incremental_serialization() {
    let dir = tempdir().unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );
    let mut rng = rand::thread_rng();
    let map = TreeMap::new(bufmans);

    for i in 0..1000 {
        map.insert(0.into(), &i, rng.gen_range(0..u16::MAX));
    }

    // edge case
    map.insert(0.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize(8).unwrap();

    for i in 1000..2001 {
        map.insert(0.into(), &i, rng.gen_range(0..u16::MAX));
    }

    map.serialize(8).unwrap();

    for i in 2001..3000 {
        map.insert(0.into(), &i, rng.gen_range(0..u16::MAX));
    }

    map.serialize(8).unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );

    let deserialized = TreeMap::<u64, u16>::deserialize(bufmans, 8).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_map_incremental_serialization_with_multiple_versions() {
    let dir = tempdir().unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );
    let mut rng = rand::thread_rng();
    let map = TreeMap::new(bufmans);

    for i in 0..1000 {
        map.insert(0.into(), &i, rng.gen_range(0..u16::MAX));
    }

    // edge case
    map.insert(1.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize(8).unwrap();

    for i in 1000..2001 {
        map.insert(2.into(), &i, rng.gen_range(0..u16::MAX));
    }

    map.serialize(8).unwrap();

    for i in 2001..3000 {
        map.insert(3.into(), &i, rng.gen_range(0..u16::MAX));
    }

    map.serialize(8).unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );

    let deserialized = TreeMap::<u64, u16>::deserialize(bufmans, 8).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_map_vec_incremental_serialization_with_multiple_versions() {
    let dir = tempdir().unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );
    let mut rng = rand::thread_rng();
    let map = TreeMapVec::new(bufmans);

    for i in 0..1000 {
        let count: u8 = rng.gen_range(1..5);
        for _ in 0..count {
            map.push(0.into(), &i, rng.gen_range(0..u16::MAX));
        }
    }

    // edge case
    map.push(1.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize(8).unwrap();

    for i in 1000..2001 {
        let count: u8 = rng.gen_range(1..5);
        for _ in 0..count {
            map.push(2.into(), &i, rng.gen_range(0..u16::MAX));
        }
    }

    map.serialize(8).unwrap();

    for i in 2001..3000 {
        let count: u8 = rng.gen_range(1..5);
        for _ in 0..count {
            map.push(3.into(), &i, rng.gen_range(0..u16::MAX));
        }
    }

    map.serialize(8).unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );

    let deserialized = TreeMapVec::<u64, u16>::deserialize(bufmans, 8).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_double_serialization_raw_vec() {
    let dir = tempdir().unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );
    let mut rng = rand::thread_rng();
    let map: TreeMap<InternalId, RawVectorEmbedding> = TreeMap::new(bufmans);

    map.serialize(8).unwrap();

    let deserialized = TreeMap::deserialize(map.bufmans, 8).unwrap();

    for i in 0..1000 {
        deserialized.insert(
            VersionNumber::from(1),
            &InternalId::from(i),
            random_raw_vector_embedding(&mut rng),
        );
    }

    deserialized.serialize(8).unwrap();

    TreeMap::<InternalId, RawVectorEmbedding>::deserialize(deserialized.bufmans, 8).unwrap();
}

#[test]
fn test_tree_double_serialization_u64() {
    let dir = tempdir().unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, idx| root.join(format!("{}.tree-map", idx)),
        8192,
    );
    let mut rng = rand::thread_rng();
    let map: TreeMap<InternalId, u64> = TreeMap::new(bufmans);
    println!("first serialization:");

    map.serialize(8).unwrap();
    println!("\nfirst deserialization:");

    let deserialized = TreeMap::<InternalId, u64>::deserialize(map.bufmans, 8).unwrap();

    for i in 0..1000 {
        deserialized.insert(VersionNumber::from(1), &InternalId::from(i), rng.gen());
    }

    deserialized.serialize(8).unwrap();

    TreeMap::<InternalId, u64>::deserialize(deserialized.bufmans, 8).unwrap();
}

fn random_raw_vector_embedding(rng: &mut impl Rng) -> RawVectorEmbedding {
    let id_len = rng.gen_range(10..20);
    RawVectorEmbedding {
        id: VectorId::from(
            rng.sample_iter(&Alphanumeric)
                .take(id_len)
                .map(char::from)
                .collect::<String>(),
        ),
        document_id: None,
        dense_values: Some(
            (0..rng.gen_range(100u32..200u32))
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect(),
        ),
        metadata: None,
        sparse_values: None,
        text: None,
    }
}
