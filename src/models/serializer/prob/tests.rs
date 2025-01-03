use super::ProbSerialize;
use crate::{
    distance::cosine::CosineSimilarity,
    models::{
        buffered_io::{BufferManager, BufferManagerFactory},
        cache_loader::ProbCache,
        file_persist::write_prop_to_file,
        lazy_load::{FileIndex, SyncPersist},
        prob_lazy_load::{lazy_item::ProbLazyItem, lazy_item_array::ProbLazyItemArray},
        prob_node::{ProbNode, SharedNode},
        types::{FileOffset, HNSWLevel, MetricResult, NodeProp, VectorId},
        versioning::{Hash, Version, VersionControl},
    },
    storage::Storage,
};
use lmdb::{DatabaseFlags, Environment};
use std::{
    collections::HashSet,
    fs::{File, OpenOptions},
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, RwLock,
    },
};
use tempfile::{tempdir, TempDir};

pub struct EqualityTester {
    checked: HashSet<(*mut ProbLazyItem<ProbNode>, *mut ProbLazyItem<ProbNode>)>,
    cache: Arc<ProbCache>,
}

pub trait EqualityTest {
    fn assert_eq(&self, _other: &Self, _tester: &mut EqualityTester) {}
}

impl EqualityTest for ProbNode {
    fn assert_eq(&self, other: &Self, tester: &mut EqualityTester) {
        assert_eq!(self.hnsw_level, other.hnsw_level);
        assert_eq!(self.prop, other.prop);
        self.versions.assert_eq(&other.versions, tester);

        let parent = self.get_parent();
        if !parent.is_null() {
            let other_parent = other.get_parent();
            assert!(!other_parent.is_null());
            parent.assert_eq(&other_parent, tester);
        } else {
            assert!(other.get_parent().is_null());
        }

        let child = self.get_child();
        if !child.is_null() {
            let other_child = other.get_child();
            assert!(!other_child.is_null());
            child.assert_eq(&other_child, tester);
        } else {
            assert!(other.get_child().is_null());
        }

        self.get_neighbors_raw()
            .assert_eq(other.get_neighbors_raw(), tester);
    }
}

impl EqualityTest for Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> {
    fn assert_eq(&self, other: &Self, tester: &mut EqualityTester) {
        assert_eq!(self.len(), other.len());
        for i in 0..self.len() {
            let self_el = unsafe { self[i].load(Ordering::Relaxed).as_ref().cloned() };
            let other_el = unsafe { other[i].load(Ordering::Relaxed).as_ref().cloned() };

            let Some((self_id, self_node, self_dist)) = self_el else {
                assert!(other_el.is_none());
                continue;
            };
            let (other_id, other_node, other_dist) = other_el.unwrap();
            assert_eq!(self_dist, other_dist);
            assert_eq!(self_id, other_id);
            self_node.assert_eq(&other_node, tester);
        }
    }
}

impl<const N: usize> EqualityTest for ProbLazyItemArray<ProbNode, N> {
    fn assert_eq(&self, other: &Self, tester: &mut EqualityTester) {
        assert_eq!(self.len(), other.len());
        for i in 0..self.len() {
            let self_el = self.get(i).unwrap();
            let other_el = other.get(i).unwrap();
            self_el.assert_eq(&other_el, tester);
        }
    }
}

impl EqualityTest for SharedNode {
    fn assert_eq(&self, other: &Self, tester: &mut EqualityTester) {
        if tester.checked.insert((*self, *other)) {
            let self_ = unsafe { &**self };
            let other = unsafe { &**other };
            assert_eq!(self_.get_current_version(), other.get_current_version());
            assert_eq!(
                self_.get_current_version_number(),
                other.get_current_version_number()
            );
            let self_data = self_.try_get_data(&tester.cache).unwrap();
            let other_data = other.try_get_data(&tester.cache).unwrap();
            self_data.assert_eq(&other_data, tester);
        }
    }
}

impl EqualityTester {
    pub fn new(cache: Arc<ProbCache>) -> Self {
        Self {
            checked: HashSet::new(),
            cache,
        }
    }
}

fn get_cache(
    bufmans: Arc<BufferManagerFactory<Hash>>,
    prop_file: Arc<RwLock<File>>,
) -> Arc<ProbCache> {
    Arc::new(ProbCache::new(1000, bufmans, prop_file))
}

fn create_prob_node(id: u64, prop_file: &RwLock<File>) -> ProbNode {
    let id = VectorId(id);
    let value = Arc::new(Storage::UnsignedByte {
        mag: 10,
        quant_vec: vec![1, 2, 3],
    });
    let mut prop_file_guard = prop_file.write().unwrap();
    let location = write_prop_to_file(&id, value.clone(), &mut *prop_file_guard).unwrap();
    drop(prop_file_guard);
    let prop = Arc::new(NodeProp {
        id,
        value,
        location,
    });
    ProbNode::new(
        HNSWLevel(2),
        prop.clone(),
        ptr::null_mut(),
        ptr::null_mut(),
        8,
    )
}

fn setup_test(
    root_version: Hash,
) -> (
    Arc<BufferManagerFactory<Hash>>,
    Arc<ProbCache>,
    Arc<BufferManager>,
    u64,
    Arc<RwLock<File>>,
    TempDir,
) {
    let dir = tempdir().unwrap();
    let bufmans = Arc::new(BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, ver: &Hash| root.join(format!("{}.index", **ver)),
        1.0,
    ));
    let prop_file = Arc::new(RwLock::new(
        OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(dir.as_ref().join("prop.data"))
            .unwrap(),
    ));
    let cache = get_cache(bufmans.clone(), prop_file.clone());
    let bufman = bufmans.get(root_version).unwrap();
    let cursor = bufman.open_cursor().unwrap();
    (bufmans, cache, bufman, cursor, prop_file, dir)
}

#[test]
fn test_lazy_item_serialization() {
    let root_version_number = 0;
    let root_version_id = Hash::from(0);
    let (bufmans, cache, bufman, cursor, prop_file, _temp_dir) = setup_test(root_version_id);
    let node = create_prob_node(0, &prop_file);

    let lazy_item = ProbLazyItem::new(node, root_version_id, root_version_number);

    let offset = lazy_item
        .serialize(&bufmans, root_version_id, cursor)
        .unwrap();
    bufman.close_cursor(cursor).unwrap();

    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: root_version_number,
        version_id: root_version_id,
    };

    let deserialized: SharedNode = cache.load_item(file_index).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    lazy_item.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_node_acyclic_serialization() {
    let root_version_id = Hash::from(0);
    let (bufmans, cache, bufman, cursor, prop_file, _temp_dir) = setup_test(root_version_id);

    let node = create_prob_node(0, &prop_file);

    let offset = node.serialize(&bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: ProbNode = cache.load_item(file_index).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    node.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_lazy_item_array_serialization() {
    let root_version_id = Hash::from(0);
    let root_version_number = 0;
    let (bufmans, cache, bufman, cursor, prop_file, _temp_dir) = setup_test(root_version_id);
    let array = ProbLazyItemArray::<_, 8>::new();

    for i in 0..5 {
        let node = create_prob_node(i, &prop_file);

        let lazy_item = ProbLazyItem::new(node, root_version_id, root_version_number);
        array.push(lazy_item);
    }

    let offset = array.serialize(&bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: ProbLazyItemArray<ProbNode, 8> = cache.load_item(file_index).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    array.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_node_serialization_with_neighbors() {
    let root_version_id = Hash::from(0);
    let root_version_number = 0;
    let (bufmans, cache, bufman, cursor, prop_file, _temp_dir) = setup_test(root_version_id);

    let node = create_prob_node(0, &prop_file);

    for i in 0..10 {
        let neighbor_node = create_prob_node(i, &prop_file);

        let lazy_item = ProbLazyItem::new(neighbor_node, root_version_id, root_version_number);
        let dist = MetricResult::CosineSimilarity(CosineSimilarity((i as f32) / 5.0));
        node.add_neighbor(i as u32, lazy_item, dist);
    }

    let offset = node.serialize(&bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: ProbNode = cache.load_item(file_index).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    node.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_lazy_item_cyclic_serialization() {
    let root_version_id = Hash::from(0);
    let root_version_number = 0;
    let (bufmans, cache, bufman, cursor, prop_file, _temp_dir) = setup_test(root_version_id);

    let node0 = create_prob_node(0, &prop_file);
    let node1 = create_prob_node(1, &prop_file);

    let lazy0 = ProbLazyItem::new(node0, root_version_id, root_version_number);
    let lazy1 = ProbLazyItem::new(node1, root_version_id, root_version_number);

    unsafe { &*lazy0 }
        .get_lazy_data()
        .unwrap()
        .set_parent(lazy1.clone());

    unsafe { &*lazy1 }
        .get_lazy_data()
        .unwrap()
        .set_child(lazy0.clone());

    let offset = lazy0.serialize(&bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: SharedNode = cache.load_item(file_index).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    lazy0.assert_eq(&deserialized, &mut tester);
}

fn validate_lazy_item_versions(
    cache: &Arc<ProbCache>,
    lazy_item: &ProbLazyItem<ProbNode>,
    version_number: u16,
) {
    let data = lazy_item.try_get_data(cache).unwrap();
    let versions = &data.versions;

    for i in 0..versions.len() {
        let version = unsafe { &*versions.get(i).unwrap() };
        let version = if version.get_lazy_data().is_none() {
            let file_index = version.get_file_index().unwrap();
            unsafe { &*cache.clone().load_item::<*mut _>(file_index).unwrap() }
        } else {
            version
        };

        let current_version_number = version.get_current_version_number();

        assert_eq!(current_version_number - version_number, 4_u16.pow(i as u32));
        validate_lazy_item_versions(cache, version, current_version_number);
    }
}

#[test]
fn test_prob_lazy_item_with_versions_serialization_and_validation() {
    let temp_dir = tempdir().unwrap();
    let env = Arc::new(
        Environment::new()
            .set_max_dbs(2)
            .set_map_size(10485760) // 10MB
            .open(temp_dir.as_ref())
            .unwrap(),
    );
    let prop_file = Arc::new(RwLock::new(
        OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(temp_dir.as_ref().join("prop.data"))
            .unwrap(),
    ));
    let db = Arc::new(env.create_db(None, DatabaseFlags::empty()).unwrap());
    let vcs = VersionControl::new(env, db).unwrap().0;

    let v0_hash = vcs.generate_hash("main", Version::from(0)).unwrap();
    let root = ProbLazyItem::new(create_prob_node(0, &prop_file), v0_hash, 0);

    let bufmans = Arc::new(BufferManagerFactory::new(
        temp_dir.as_ref().into(),
        |root, ver: &Hash| root.join(format!("{}.index", **ver)),
        1.0,
    ));
    let cache = get_cache(bufmans.clone(), prop_file.clone());
    let bufman = bufmans.get(v0_hash).unwrap();
    let cursor = bufman.open_cursor().unwrap();

    for i in 1..=100 {
        let (hash, _) = vcs.add_next_version("main").unwrap();
        let next_version = ProbLazyItem::new(create_prob_node(0, &prop_file), hash, i);
        ProbLazyItem::add_version(root, next_version, &cache)
            .unwrap()
            .map_err(|_| "unable to insert neighbor")
            .unwrap();
    }

    validate_lazy_item_versions(&cache, unsafe { &*root }, 0);

    let offset = root.serialize(&bufmans, v0_hash, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: v0_hash,
    };
    bufman.close_cursor(cursor).unwrap();
    bufmans.flush_all().unwrap();

    let deserialized: SharedNode = cache.clone().load_item(file_index).unwrap();

    validate_lazy_item_versions(&cache, unsafe { &*deserialized }, 0);

    let mut tester = EqualityTester::new(cache.clone());

    root.assert_eq(&deserialized, &mut tester);
}
