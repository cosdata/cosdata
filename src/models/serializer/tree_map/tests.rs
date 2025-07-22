use std::{fs::OpenOptions, path::Path};

use rand::{
    distributions::{Alphanumeric, Standard},
    prelude::Distribution,
    Rng,
};
use tempfile::{tempdir, TempDir};

use crate::models::{
    buffered_io::{BufferManager, BufferManagerFactory},
    collection::RawVectorEmbedding,
    serializer::tree_map::TreeMapSerialize,
    tree_map::{QuotientsMap, QuotientsMapVec, TreeMap, TreeMapVec},
    types::{FileOffset, InternalId, VectorId},
    versioned_vec::{VersionedVec, VersionedVecItem},
    versioning::VersionNumber,
};

fn setup_test() -> (
    BufferManager,
    BufferManagerFactory<VersionNumber>,
    u64,
    TempDir,
) {
    let dir = tempdir().unwrap();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.as_ref().join("tree_map.dim"))
        .unwrap();
    let dim_bufman = BufferManager::new(file, 8192).unwrap();
    let data_bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, version: &VersionNumber| root.join(format!("tree_map.{}.data", **version)),
        8192,
    );
    let cursor = dim_bufman.open_cursor().unwrap();
    (dim_bufman, data_bufmans, cursor, dir)
}

fn setup_test_in_dir(dir: &Path) -> (BufferManager, BufferManagerFactory<VersionNumber>) {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("tree_map.dim"))
        .unwrap();
    let dim_bufman = BufferManager::new(file, 8192).unwrap();
    let data_bufmans = BufferManagerFactory::new(
        dir.into(),
        |root, version: &VersionNumber| root.join(format!("tree_map.{}.data", **version)),
        8192,
    );
    (dim_bufman, data_bufmans)
}

fn get_random_versioned_vec<T>(rng: &mut impl Rng, version: VersionNumber) -> VersionedVec<T>
where
    T: VersionedVecItem,
    Standard: Distribution<T> + Distribution<<T as VersionedVecItem>::Id>,
    <T as VersionedVecItem>::Id: Eq,
{
    let mut vec = VersionedVec::new(version);
    let count = rng.gen_range(20..50);
    add_random_items_to_versioned_vec(rng, &mut vec, count, version);
    vec
}

fn add_random_items_to_versioned_vec<T>(
    rng: &mut impl Rng,
    vec: &mut VersionedVec<T>,
    count: usize,
    version: VersionNumber,
) where
    T: VersionedVecItem,
    Standard: Distribution<T> + Distribution<<T as VersionedVecItem>::Id>,
    <T as VersionedVecItem>::Id: Eq,
{
    for _ in 0..count {
        vec.push(version, rng.gen());
    }

    for _ in 0..count {
        vec.delete(version, rng.gen());
    }
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

#[test]
fn test_quotients_serialization() {
    let (dim_bufman, data_bufmans, cursor, _temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let quotients_map = QuotientsMap::default();
    for _ in 0..10000 {
        quotients_map.insert(
            0.into(),
            rng.gen_range(0..u64::MAX),
            rng.gen_range(0..u16::MAX),
        );
    }
    let offset = quotients_map
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    let deserialized = QuotientsMap::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(u32::MAX), // not used
    )
    .unwrap();

    assert_eq!(quotients_map, deserialized);
}

#[test]
fn test_quotients_vec_serialization() {
    let (dim_bufman, data_bufmans, cursor, _temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let quotients_vec = QuotientsMapVec::default();
    for _ in 0..10000 {
        quotients_vec.push(
            0.into(),
            rng.gen_range(0..u64::MAX),
            rng.gen_range(0..u16::MAX),
        );
    }
    let offset = quotients_vec
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    let deserialized = QuotientsMapVec::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(u32::MAX), // not used
    )
    .unwrap();

    assert_eq!(quotients_vec, deserialized);
}

#[test]
fn test_empty_quotients_vec_serialization() {
    let (dim_bufman, data_bufmans, cursor, _temp_dir) = setup_test();
    let quotients_vec = QuotientsMapVec::<InternalId>::default();
    let offset = quotients_vec
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    let deserialized = QuotientsMapVec::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(u32::MAX), // not used
    )
    .unwrap();

    assert_eq!(quotients_vec, deserialized);
}

#[test]
fn test_quotients_incremental_serialization() {
    let (dim_bufman, data_bufmans, cursor, _temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let quotients_map = QuotientsMap::default();
    for i in 0..1000 {
        quotients_map.insert(0.into(), i, rng.gen_range(0..u16::MAX));
    }
    let _offset = quotients_map
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    for i in 1000..2001 {
        quotients_map.insert(1.into(), i, rng.gen_range(0..u16::MAX));
    }
    let _offset = quotients_map
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    for i in 2001..3000 {
        quotients_map.insert(2.into(), i, rng.gen_range(0..u16::MAX));
    }
    let offset = quotients_map
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    let deserialized = QuotientsMap::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(u32::MAX), // not used
    )
    .unwrap();

    assert_eq!(quotients_map, deserialized);
}

#[test]
fn test_quotients_vec_incremental_serialization() {
    let (dim_bufman, data_bufmans, cursor, _temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let quotients_vec = QuotientsMapVec::default();
    for i in 0..1000 {
        quotients_vec.push(0.into(), i, rng.gen_range(0..u16::MAX));
    }
    let _offset = quotients_vec
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    for i in 1000..2001 {
        quotients_vec.push(1.into(), i, rng.gen_range(0..u16::MAX));
    }
    let _offset = quotients_vec
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    for i in 2001..3000 {
        quotients_vec.push(2.into(), i, rng.gen_range(0..u16::MAX));
    }
    let offset = quotients_vec
        .serialize(&dim_bufman, &data_bufmans, cursor)
        .unwrap();
    let deserialized = QuotientsMapVec::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(u32::MAX), // not used
    )
    .unwrap();

    assert_eq!(quotients_vec, deserialized);
}

#[test]
fn test_versioned_vec_serialization() {
    let mut rng = rand::thread_rng();
    let vec: VersionedVec<u32> = get_random_versioned_vec(&mut rng, 0.into());

    let (dim_bufman, data_bufmans, cursor, _temp_dir) = setup_test();
    let offset = vec.serialize(&dim_bufman, &data_bufmans, cursor).unwrap();

    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = VersionedVec::<u32>::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(0),
    )
    .unwrap();

    assert_eq!(vec, deserialized);
}

#[test]
fn test_versioned_vec_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let mut vec: VersionedVec<u32> = get_random_versioned_vec(&mut rng, 0.into());

    let (dim_bufman, data_bufmans, cursor, _temp_dir) = setup_test();
    let _offset = vec.serialize(&dim_bufman, &data_bufmans, cursor).unwrap();

    add_random_items_to_versioned_vec(&mut rng, &mut vec, 100, 1.into());

    let offset = vec.serialize(&dim_bufman, &data_bufmans, cursor).unwrap();
    dim_bufman.close_cursor(cursor).unwrap();

    let deserialized = VersionedVec::<u32>::deserialize(
        &dim_bufman,
        &data_bufmans,
        FileOffset(offset),
        VersionNumber::from(0),
    )
    .unwrap();

    assert_eq!(vec, deserialized);
}

#[test]
fn test_tree_map_serialization() {
    let (dim_bufman, data_bufmans, _cursor, temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let map = TreeMap::new(dim_bufman, data_bufmans);

    for i in 0..1000 {
        map.insert(0.into(), &i, rng.gen_range(0..u16::MAX));
    }

    // edge case
    map.insert(0.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize().unwrap();

    let (dim_bufman, data_bufmans) = setup_test_in_dir(temp_dir.as_ref());

    let deserialized = TreeMap::<u64, u16>::deserialize(dim_bufman, data_bufmans).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_map_vec_serialization() {
    let (dim_bufman, data_bufmans, _cursor, temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let map = TreeMapVec::new(dim_bufman, data_bufmans);

    for i in 0..10000 {
        map.push(0.into(), &(i / 10), rng.gen_range(0..u16::MAX));
    }

    // edge case
    map.push(0.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize().unwrap();

    let (dim_bufman, data_bufmans) = setup_test_in_dir(temp_dir.as_ref());

    let deserialized = TreeMapVec::<u64, u16>::deserialize(dim_bufman, data_bufmans).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_map_incremental_serialization() {
    let (dim_bufman, data_bufmans, _cursor, temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let map = TreeMap::new(dim_bufman, data_bufmans);

    for i in 0..1000 {
        map.insert(0.into(), &i, rng.gen_range(0..u16::MAX));
    }

    // edge case
    map.insert(0.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize().unwrap();

    for i in 1000..2001 {
        map.insert(1.into(), &i, rng.gen_range(0..u16::MAX));
    }

    map.serialize().unwrap();

    for i in 2001..3000 {
        map.insert(2.into(), &i, rng.gen_range(0..u16::MAX));
    }

    map.serialize().unwrap();

    let (dim_bufman, data_bufmans) = setup_test_in_dir(temp_dir.as_ref());

    let deserialized = TreeMap::<u64, u16>::deserialize(dim_bufman, data_bufmans).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_map_vec_incremental_serialization() {
    let (dim_bufman, data_bufmans, _cursor, temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let map = TreeMapVec::new(dim_bufman, data_bufmans);

    for i in 0..10000 {
        map.push(0.into(), &(i / 10), rng.gen_range(0..u16::MAX));
    }

    // edge case
    map.push(0.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize().unwrap();

    for i in 5000..15000 {
        map.push(1.into(), &(i / 10), rng.gen_range(0..u16::MAX));
    }

    map.serialize().unwrap();

    for i in 10000..20000 {
        map.push(2.into(), &(i / 10), rng.gen_range(0..u16::MAX));
    }

    map.serialize().unwrap();

    let (dim_bufman, data_bufmans) = setup_test_in_dir(temp_dir.as_ref());

    let deserialized = TreeMapVec::<u64, u16>::deserialize(dim_bufman, data_bufmans).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_map_incremental_serialization_with_multiple_versions() {
    let (dim_bufman, data_bufmans, _cursor, temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let map = TreeMap::new(dim_bufman, data_bufmans);

    for i in 0..1000 {
        map.insert(0.into(), &i, rng.gen_range(0..u16::MAX));
    }

    // edge case
    map.insert(1.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize().unwrap();

    for i in 1000..2001 {
        map.insert(2.into(), &i, rng.gen_range(0..u16::MAX));
    }

    map.serialize().unwrap();

    for i in 2001..3000 {
        map.insert(3.into(), &i, rng.gen_range(0..u16::MAX));
    }

    map.serialize().unwrap();

    let (dim_bufman, data_bufmans) = setup_test_in_dir(temp_dir.as_ref());

    let deserialized = TreeMap::<u64, u16>::deserialize(dim_bufman, data_bufmans).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_map_vec_incremental_serialization_with_multiple_versions() {
    let (dim_bufman, data_bufmans, _cursor, temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let map = TreeMapVec::new(dim_bufman, data_bufmans);

    for i in 0..1000 {
        let count: u8 = rng.gen_range(1..5);
        for _ in 0..count {
            map.push(0.into(), &i, rng.gen_range(0..u16::MAX));
        }
    }

    // edge case
    map.push(1.into(), &u64::MAX, rng.gen_range(0..u16::MAX));

    map.serialize().unwrap();

    for i in 1000..2001 {
        let count: u8 = rng.gen_range(1..5);
        for _ in 0..count {
            map.push(2.into(), &i, rng.gen_range(0..u16::MAX));
        }
    }

    map.serialize().unwrap();

    for i in 2001..3000 {
        let count: u8 = rng.gen_range(1..5);
        for _ in 0..count {
            map.push(3.into(), &i, rng.gen_range(0..u16::MAX));
        }
    }

    map.serialize().unwrap();

    let (dim_bufman, data_bufmans) = setup_test_in_dir(temp_dir.as_ref());

    let deserialized = TreeMapVec::<u64, u16>::deserialize(dim_bufman, data_bufmans).unwrap();

    assert_eq!(map, deserialized);
}

#[test]
fn test_tree_double_serialization_raw_vec() {
    let (dim_bufman, data_bufmans, _cursor, _temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let map: TreeMap<InternalId, RawVectorEmbedding> = TreeMap::new(dim_bufman, data_bufmans);

    map.serialize().unwrap();

    let deserialized = TreeMap::deserialize(map.dim_bufman, map.data_bufmans).unwrap();

    for i in 0..1000 {
        deserialized.insert(
            VersionNumber::from(1),
            &InternalId::from(i),
            random_raw_vector_embedding(&mut rng),
        );
    }

    deserialized.serialize().unwrap();

    TreeMap::<InternalId, RawVectorEmbedding>::deserialize(
        deserialized.dim_bufman,
        deserialized.data_bufmans,
    )
    .unwrap();
}

#[test]
fn test_tree_double_serialization_u64() {
    let (dim_bufman, data_bufmans, _cursor, _temp_dir) = setup_test();
    let mut rng = rand::thread_rng();
    let map: TreeMap<InternalId, u64> = TreeMap::new(dim_bufman, data_bufmans);

    map.serialize().unwrap();

    let deserialized =
        TreeMap::<InternalId, u64>::deserialize(map.dim_bufman, map.data_bufmans).unwrap();

    for i in 0..1000 {
        deserialized.insert(VersionNumber::from(1), &InternalId::from(i), rng.gen());
    }

    deserialized.serialize().unwrap();

    TreeMap::<InternalId, u64>::deserialize(deserialized.dim_bufman, deserialized.data_bufmans)
        .unwrap();
}
