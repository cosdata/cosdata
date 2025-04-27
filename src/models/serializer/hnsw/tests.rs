use super::HNSWIndexSerialize;
use crate::{
    distance::cosine::CosineSimilarity,
    models::{
        buffered_io::{BufferManager, BufferManagerFactory},
        cache_loader::HNSWIndexCache,
        file_persist::write_prop_value_to_file,
        prob_lazy_load::{
            lazy_item::{FileIndex, ProbLazyItem},
            lazy_item_array::ProbLazyItemArray,
        },
        prob_node::{ProbNode, SharedNode},
        types::{DistanceMetric, FileOffset, HNSWLevel, InternalId, MetricResult, NodePropValue},
        versioning::Hash,
    },
    storage::Storage,
};
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
    cache: Arc<HNSWIndexCache>,
}

pub trait EqualityTest {
    fn assert_eq(&self, _other: &Self, _tester: &mut EqualityTester) {}
}

impl EqualityTest for ProbNode {
    fn assert_eq(&self, other: &Self, tester: &mut EqualityTester) {
        assert_eq!(self.hnsw_level, other.hnsw_level);
        assert_eq!(self.prop_value, other.prop_value);

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

impl EqualityTest for Box<[AtomicPtr<(InternalId, SharedNode, MetricResult)>]> {
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
            assert_eq!(self_.file_index, other.file_index);
            let self_data = self_.try_get_data(&tester.cache).unwrap();
            let other_data = other.try_get_data(&tester.cache).unwrap();
            self_data.assert_eq(other_data, tester);
        }
    }
}

impl EqualityTester {
    pub fn new(cache: Arc<HNSWIndexCache>) -> Self {
        Self {
            checked: HashSet::new(),
            cache,
        }
    }
}

fn get_cache(
    bufmans: Arc<BufferManagerFactory<Hash>>,
    prop_file: RwLock<File>,
) -> Arc<HNSWIndexCache> {
    Arc::new(HNSWIndexCache::new(
        bufmans.clone(),
        bufmans,
        prop_file,
        Arc::new(RwLock::new(DistanceMetric::Cosine)),
    ))
}

fn create_prob_node(id: u32, prop_file: &RwLock<File>) -> ProbNode {
    let id = InternalId::from(id);
    let value = Storage::UnsignedByte {
        mag: 10.0,
        quant_vec: vec![1, 2, 3],
    };
    let mut prop_file_guard = prop_file.write().unwrap();
    let location = write_prop_value_to_file(&id, &value, &mut prop_file_guard).unwrap();
    drop(prop_file_guard);
    let prop = Arc::new(NodePropValue {
        id,
        vec: Arc::new(value),
        location,
    });
    ProbNode::new(
        HNSWLevel(2),
        prop.clone(),
        // @TODO(vineet): Add tests for optional metadata dimensions
        None,
        ptr::null_mut(),
        ptr::null_mut(),
        8,
        DistanceMetric::Cosine,
    )
}

fn setup_test(
    root_version: Hash,
) -> (
    Arc<BufferManagerFactory<Hash>>,
    Arc<HNSWIndexCache>,
    Arc<BufferManager>,
    u64,
    TempDir,
) {
    let dir = tempdir().unwrap();
    let bufmans = Arc::new(BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, ver: &Hash| root.join(format!("{}.index", **ver)),
        ProbNode::get_serialized_size(8),
    ));
    let prop_file = RwLock::new(
        OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(dir.as_ref().join("prop.data"))
            .unwrap(),
    );
    let cache = get_cache(bufmans.clone(), prop_file);
    let bufman = bufmans.get(root_version).unwrap();
    let cursor = bufman.open_cursor().unwrap();
    (bufmans, cache, bufman, cursor, dir)
}

#[test]
fn test_lazy_item_serialization() {
    let root_version_number = 0;
    let root_version_id = Hash::from(0);
    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(root_version_id);
    let node = create_prob_node(0, &cache.prop_file);

    let lazy_item = ProbLazyItem::new(
        node,
        root_version_id,
        root_version_number,
        false,
        FileOffset(0),
    );

    let offset = lazy_item
        .serialize(&bufmans, root_version_id, cursor)
        .unwrap();
    bufman.close_cursor(cursor).unwrap();

    let file_index = FileIndex {
        offset: FileOffset(offset),
        version_number: root_version_number,
        version_id: root_version_id,
    };

    let deserialized: SharedNode = cache.load_item(file_index, false).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    lazy_item.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_node_acyclic_serialization() {
    let root_version_id = Hash::from(0);
    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(root_version_id);

    let node = create_prob_node(0, &cache.prop_file);

    let offset = node.serialize(&bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: ProbNode = cache.load_item(file_index, false).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    node.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_lazy_item_array_serialization() {
    let root_version_id = Hash::from(0);
    let root_version_number = 0;
    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(root_version_id);
    let array = ProbLazyItemArray::<_, 8>::new();
    let node_size = ProbNode::get_serialized_size(8) as u32;

    for i in 0..5 {
        let node = create_prob_node(i, &cache.prop_file);

        let lazy_item = ProbLazyItem::new(
            node,
            root_version_id,
            root_version_number,
            false,
            FileOffset(node_size * (i + 1)),
        );
        array.push(lazy_item);
    }
    let offset = node_size - 80;
    bufman.seek_with_cursor(cursor, offset as u64).unwrap();
    array.serialize(&bufmans, root_version_id, cursor).unwrap();
    for i in 0..5 {
        array
            .get(i)
            .unwrap()
            .serialize(&bufmans, root_version_id, cursor)
            .unwrap();
    }
    let file_index = FileIndex {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: ProbLazyItemArray<ProbNode, 8> = cache.load_item(file_index, false).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    array.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_node_serialization_with_neighbors() {
    let root_version_id = Hash::from(0);
    let root_version_number = 0;
    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(root_version_id);

    let mut nodes = Vec::new();

    let node = create_prob_node(0, &cache.prop_file);
    let lazy_node = ProbLazyItem::new(
        node,
        root_version_id,
        root_version_number,
        false,
        FileOffset(0),
    );
    let node_size = ProbNode::get_serialized_size(8) as u32;

    nodes.push(lazy_node);

    for i in 1..11 {
        let neighbor_node = create_prob_node(i, &cache.prop_file);

        let lazy_item = ProbLazyItem::new(
            neighbor_node,
            root_version_id,
            root_version_number,
            false,
            FileOffset(node_size * i),
        );
        let dist = MetricResult::CosineSimilarity(CosineSimilarity((i as f32) / 10.0));
        unsafe { &*lazy_node }
            .get_lazy_data()
            .unwrap()
            .add_neighbor(
                InternalId::from(i),
                lazy_item,
                dist,
                &cache,
                DistanceMetric::Cosine,
            );
        nodes.push(lazy_item);
    }

    for node in nodes {
        node.serialize(&bufmans, root_version_id, cursor).unwrap();
    }

    let file_index = FileIndex {
        offset: FileOffset(0),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: SharedNode = cache.load_item(file_index, false).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    lazy_node.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_lazy_item_cyclic_serialization() {
    let root_version_id = Hash::from(0);
    let root_version_number = 0;
    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(root_version_id);

    let node0 = create_prob_node(0, &cache.prop_file);
    let node1 = create_prob_node(1, &cache.prop_file);
    let node_size = ProbNode::get_serialized_size(8) as u32;

    let lazy0 = ProbLazyItem::new(
        node0,
        root_version_id,
        root_version_number,
        false,
        FileOffset(0),
    );
    let lazy1 = ProbLazyItem::new(
        node1,
        root_version_id,
        root_version_number,
        false,
        FileOffset(node_size),
    );

    unsafe { &*lazy0 }
        .get_lazy_data()
        .unwrap()
        .set_parent(lazy1);

    unsafe { &*lazy1 }.get_lazy_data().unwrap().set_child(lazy0);

    lazy0.serialize(&bufmans, root_version_id, cursor).unwrap();
    lazy1.serialize(&bufmans, root_version_id, cursor).unwrap();

    let file_index = FileIndex {
        offset: FileOffset(0),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: SharedNode = cache.load_item(file_index, false).unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    lazy0.assert_eq(&deserialized, &mut tester);
}
