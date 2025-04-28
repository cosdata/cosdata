use std::fs::OpenOptions;
use std::sync::Arc;

use crate::models::buffered_io::BufferManagerFactory;
use crate::models::inverted_index::InvertedIndexNode;
use crate::models::page::Pagepool;
use crate::models::page::VersionedPagepool;
use crate::models::serializer::*;
use crate::models::tree_map::QuotientsMap;
use crate::models::tree_map::TreeMap;
use crate::models::tree_map::TreeMapVec;
use crate::models::types::*;
use crate::models::versioning::Hash;
use crate::storage::Storage;
use half::f16;
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
        |root, ver: &Hash| root.join(format!("{}.index", **ver)),
        8192,
    );

    for (version, storage) in storages.into_iter().enumerate() {
        let version_id = Hash::from(version as u32);
        let bufman = bufmans.get(version_id).unwrap();
        let cursor = bufman.open_cursor().unwrap();
        let offset = SimpleSerialize::serialize(&storage, &bufman, cursor).unwrap();
        let deserialized: Storage =
            SimpleSerialize::deserialize(&bufman, FileOffset(offset)).unwrap();

        assert_eq!(deserialized, storage);
    }
}

fn get_random_pagepool<const LEN: usize>(rng: &mut impl Rng) -> Pagepool<LEN> {
    let pool = Pagepool::default();
    let count = rng.gen_range(20..50);
    add_random_items_to_pagepool(rng, &pool, count);
    pool
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

fn add_random_items_to_pagepool<const LEN: usize>(
    rng: &mut impl Rng,
    pool: &Pagepool<LEN>,
    count: usize,
) {
    for _ in 0..count {
        pool.push(rng.gen_range(0..u32::MAX));
    }
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
fn test_pagepool_serialization() {
    let mut rng = rand::thread_rng();
    let page_pool = get_random_pagepool(&mut rng);

    let (bufman, cursor, _temp_dir) = setup_test();
    let offset = page_pool.serialize(&bufman, cursor).unwrap();

    bufman.close_cursor(cursor).unwrap();

    let deserialized = Pagepool::<10>::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_pagepool_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let page_pool = get_random_pagepool(&mut rng);

    let (bufman, cursor, _temp_dir) = setup_test();
    let _offset = page_pool.serialize(&bufman, cursor).unwrap();

    add_random_items_to_pagepool(&mut rng, &page_pool, 100);

    let offset = page_pool.serialize(&bufman, cursor).unwrap();
    bufman.close_cursor(cursor).unwrap();

    let deserialized = Pagepool::<10>::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_versioned_pagepool_serialization() {
    let mut rng = rand::thread_rng();
    let page_pool: VersionedPagepool<10> = get_random_versioned_pagepool(&mut rng, 0.into());

    let (bufman, cursor, _temp_dir) = setup_test();
    let offset = page_pool.serialize(&bufman, cursor).unwrap();

    bufman.close_cursor(cursor).unwrap();

    let deserialized = VersionedPagepool::<10>::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_versioned_pagepool_incremental_serialization() {
    let mut rng = rand::thread_rng();
    let page_pool = get_random_versioned_pagepool(&mut rng, 0.into());

    let (bufman, cursor, _temp_dir) = setup_test();
    let _offset = page_pool.serialize(&bufman, cursor).unwrap();

    add_random_items_to_versioned_pagepool(&mut rng, &page_pool, 100, 0.into());

    let offset = page_pool.serialize(&bufman, cursor).unwrap();
    bufman.close_cursor(cursor).unwrap();

    let deserialized = VersionedPagepool::<10>::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(page_pool, deserialized);
}

#[test]
fn test_versioned_pagepool_incremental_serialization2() {
    let mut rng = rand::thread_rng();
    let page_pool = get_random_versioned_pagepool(&mut rng, 0.into());

    let (bufman, cursor, _temp_dir) = setup_test();
    let _offset = page_pool.serialize(&bufman, cursor).unwrap();

    add_random_items_to_versioned_pagepool(&mut rng, &page_pool, 100, 0.into());

    let _offset = page_pool.serialize(&bufman, cursor).unwrap();

    add_random_items_to_versioned_pagepool(&mut rng, &page_pool, 100, 1.into());

    let _offset = page_pool.serialize(&bufman, cursor).unwrap();

    add_random_items_to_versioned_pagepool(&mut rng, &page_pool, 100, 1.into());

    let offset = page_pool.serialize(&bufman, cursor).unwrap();
    bufman.close_cursor(cursor).unwrap();

    let deserialized = VersionedPagepool::<10>::deserialize(&bufman, FileOffset(offset)).unwrap();

    assert_eq!(page_pool, deserialized);
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
