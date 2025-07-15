use super::HNSWIndexSerialize;
use crate::{
    distance::cosine::CosineSimilarity,
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufferManager, BufferManagerFactory, FilelessBufferManager},
        cache_loader::HNSWIndexCache,
        file_persist::write_prop_value_to_file,
        lazy_item::{FileIndex, LazyItem},
        prob_node::{LatestNode, Neighbors, ProbNode, SharedLatestNode, SharedNode},
        types::{DistanceMetric, FileOffset, HNSWLevel, InternalId, MetricResult, NodePropValue},
        versioning::VersionNumber,
    },
    storage::Storage,
};
use rustc_hash::FxHashSet;
use std::{
    collections::HashSet,
    fs::{File, OpenOptions},
    ptr,
    sync::{atomic::Ordering, Arc, RwLock},
};
use tempfile::{tempdir, TempDir};

pub struct EqualityTester {
    checked: HashSet<(SharedNode, SharedNode)>,
    cache: Arc<HNSWIndexCache>,
}

pub trait EqualityTest {
    fn assert_eq(&self, _other: &Self, _tester: &mut EqualityTester) {}
}

impl EqualityTest for ProbNode {
    fn assert_eq(&self, other: &Self, tester: &mut EqualityTester) {
        assert_eq!(self.hnsw_level, other.hnsw_level);
        assert_eq!(
            self.version.load(Ordering::Relaxed),
            other.version.load(Ordering::Relaxed)
        );
        assert_eq!(self.prop_value, other.prop_value);
        assert_eq!(self.prop_metadata, other.prop_metadata);

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

impl EqualityTest for Neighbors {
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

impl EqualityTest for SharedLatestNode {
    fn assert_eq(&self, other: &Self, tester: &mut EqualityTester) {
        if self.is_null() {
            assert!(other.is_null());
            return;
        }
        let this = unsafe { &**self };
        let other = unsafe { &**other };
        this.latest.assert_eq(&other.latest, tester);
        assert_eq!(this.file_offset, other.file_offset);
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

fn create_prob_node(id: u32, version: VersionNumber, prop_file: &RwLock<File>) -> ProbNode {
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
        version,
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
    root_version_file_id: IndexFileId,
) -> (Arc<HNSWIndexCache>, Arc<BufferManager>, u64, TempDir) {
    let dir = tempdir().unwrap();
    let bufmans = BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, ver: &IndexFileId| root.join(format!("{}.index", **ver)),
        ProbNode::get_serialized_size(8),
    );
    let latest_version_links_bufman = FilelessBufferManager::new(8192).unwrap();
    let prop_file = RwLock::new(
        OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(dir.as_ref().join("prop.data"))
            .unwrap(),
    );
    let bufman = bufmans.get(root_version_file_id).unwrap();
    let cursor = bufman.open_cursor().unwrap();
    let cache = Arc::new(HNSWIndexCache::new(
        bufmans,
        latest_version_links_bufman,
        dir.as_ref().to_path_buf(),
        true,
        prop_file,
        Arc::new(RwLock::new(DistanceMetric::Cosine)),
    ));
    (cache, bufman, cursor, dir)
}

#[test]
fn test_lazy_item_serialization() {
    let root_version = VersionNumber::from(0);
    let root_version_file_id = IndexFileId::from(0);
    let (cache, bufman, cursor, _temp_dir) = setup_test(root_version_file_id);
    let node = create_prob_node(0, root_version, &cache.prop_file);

    let lazy_item = LazyItem::new(node, root_version_file_id, FileOffset(0));

    let offset = lazy_item
        .serialize(
            &bufman,
            &cache.latest_version_links_bufman,
            cursor,
            u64::MAX, // invalid cursor (not used)
        )
        .unwrap();
    bufman.close_cursor(cursor).unwrap();

    let file_index = FileIndex {
        offset: FileOffset(offset),
        file_id: root_version_file_id,
    };

    let deserialized = SharedNode::deserialize(
        &bufman,
        &cache.latest_version_links_bufman,
        file_index,
        &cache,
        1000,
        &mut FxHashSet::default(),
    )
    .unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    lazy_item.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_node_acyclic_serialization() {
    let root_version = VersionNumber::from(0);
    let root_version_file_id = IndexFileId::from(0);
    let (cache, bufman, cursor, _temp_dir) = setup_test(root_version_file_id);

    let node = create_prob_node(0, root_version, &cache.prop_file);

    let offset = node
        .serialize(
            &bufman,
            &cache.latest_version_links_bufman,
            cursor,
            u64::MAX, // invalid cursor (not used)
        )
        .unwrap();
    bufman.close_cursor(cursor).unwrap();

    let file_index = FileIndex {
        offset: FileOffset(offset),
        file_id: root_version_file_id,
    };

    let deserialized = ProbNode::deserialize(
        &bufman,
        &cache.latest_version_links_bufman,
        file_index,
        &cache,
        1000,
        &mut FxHashSet::default(),
    )
    .unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    node.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_node_serialization_with_neighbors() {
    let root_version = VersionNumber::from(0);
    let root_version_file_id = IndexFileId::from(0);
    let (cache, bufman, cursor, _temp_dir) = setup_test(root_version_file_id);

    let mut nodes = Vec::new();

    let node = create_prob_node(0, root_version, &cache.prop_file);
    let lazy_node = LazyItem::new(node, root_version_file_id, FileOffset(0));
    let lazy_node_ptr = LatestNode::new(lazy_node, FileOffset(0));
    let node_size = ProbNode::get_serialized_size(8) as u32;

    nodes.push(lazy_node_ptr);

    for i in 1..11 {
        let neighbor_node = create_prob_node(i, root_version, &cache.prop_file);

        let lazy_item = LazyItem::new(
            neighbor_node,
            root_version_file_id,
            FileOffset(node_size * i),
        );
        let lazy_item_ptr = LatestNode::new(lazy_item, FileOffset(i * 8));
        let dist = MetricResult::CosineSimilarity(CosineSimilarity((i as f32) / 10.0));
        unsafe { &*lazy_node }
            .get_lazy_data()
            .unwrap()
            .add_neighbor(
                InternalId::from(i),
                lazy_item_ptr,
                dist,
                &cache,
                DistanceMetric::Cosine,
            );
        nodes.push(lazy_item_ptr);
    }

    let latest_version_links_cursor = cache.latest_version_links_bufman.open_cursor().unwrap();
    for node in nodes {
        node.serialize(
            &bufman,
            &cache.latest_version_links_bufman,
            cursor,
            latest_version_links_cursor,
        )
        .unwrap();
    }
    cache
        .latest_version_links_bufman
        .close_cursor(latest_version_links_cursor)
        .unwrap();
    bufman.close_cursor(cursor).unwrap();

    let file_index = FileIndex {
        offset: FileOffset(0),
        file_id: root_version_file_id,
    };

    let deserialized = SharedLatestNode::deserialize(
        &bufman,
        &cache.latest_version_links_bufman,
        file_index,
        &cache,
        1000,
        &mut FxHashSet::default(),
    )
    .unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    lazy_node_ptr.assert_eq(&deserialized, &mut tester);
}

#[test]
fn test_prob_lazy_item_cyclic_serialization() {
    let root_version = VersionNumber::from(0);
    let root_version_file_id = IndexFileId::from(0);
    let (cache, bufman, cursor, _temp_dir) = setup_test(root_version_file_id);

    let node0 = create_prob_node(0, root_version, &cache.prop_file);
    let node1 = create_prob_node(1, root_version, &cache.prop_file);
    let node_size = ProbNode::get_serialized_size(8) as u32;

    let lazy0 = LazyItem::new(node0, root_version_file_id, FileOffset(0));
    let lazy1 = LazyItem::new(node1, root_version_file_id, FileOffset(node_size));

    let lazy0_ptr = LatestNode::new(lazy0, FileOffset(0));
    let lazy1_ptr = LatestNode::new(lazy1, FileOffset(8));

    unsafe { &*lazy0 }
        .get_lazy_data()
        .unwrap()
        .set_parent(lazy1_ptr);

    unsafe { &*lazy1 }
        .get_lazy_data()
        .unwrap()
        .set_child(lazy0_ptr);

    let latest_version_links_cursor = cache.latest_version_links_bufman.open_cursor().unwrap();
    lazy0_ptr
        .serialize(
            &bufman,
            &cache.latest_version_links_bufman,
            cursor,
            latest_version_links_cursor,
        )
        .unwrap();
    lazy1_ptr
        .serialize(
            &bufman,
            &cache.latest_version_links_bufman,
            cursor,
            latest_version_links_cursor,
        )
        .unwrap();

    bufman.close_cursor(cursor).unwrap();
    cache
        .latest_version_links_bufman
        .close_cursor(latest_version_links_cursor)
        .unwrap();

    let file_index = FileIndex {
        offset: FileOffset(0),
        file_id: root_version_file_id,
    };

    let deserialized = SharedLatestNode::deserialize(
        &bufman,
        &cache.latest_version_links_bufman,
        file_index,
        &cache,
        1000,
        &mut FxHashSet::default(),
    )
    .unwrap();

    let mut tester = EqualityTester::new(cache.clone());

    lazy0_ptr.assert_eq(&deserialized, &mut tester);
}
