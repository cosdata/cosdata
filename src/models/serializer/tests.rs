use crate::distance::cosine::CosineSimilarity;
use crate::models::buffered_io::BufferManager;
use crate::models::lazy_load::*;
use crate::models::serializer::*;
use crate::models::types::*;
use crate::models::versioning::BranchId;
use crate::models::versioning::VersionHash;
use crate::models::versioning::{Version, VersionControl};
use crate::storage::Storage;
use arcshift::ArcShift;
use half::f16;
use lmdb::Environment;
use std::sync::Arc;
use tempfile::{tempdir, TempDir};

fn get_cache(bufmans: Arc<BufferManagerFactory>) -> Arc<NodeRegistry> {
    Arc::new(NodeRegistry::new(1000, bufmans))
}

fn setup_test(
    root_version: &Hash,
) -> (
    Arc<BufferManagerFactory>,
    Arc<NodeRegistry>,
    Arc<BufferManager>,
    u64,
    TempDir,
) {
    let dir = tempdir().unwrap();
    let bufmans = Arc::new(BufferManagerFactory::new(
        dir.as_ref().into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));
    let cache = get_cache(bufmans.clone());
    let bufman = bufmans.get(root_version).unwrap();
    let cursor = bufman.open_cursor().unwrap();
    (bufmans, cache, bufman, cursor, dir)
}

#[test]
fn test_lazy_item_serialization() {
    let node = MergedNode::new(HNSWLevel(2));
    let root_version_number = 0;
    let root_version_id = Hash::from(0);
    let lazy_item = LazyItemRef::new(root_version_id, root_version_number, node);

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = lazy_item
        .serialize(bufmans.clone(), root_version_id, cursor)
        .unwrap();
    bufman.close_cursor(cursor).unwrap();

    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: root_version_number,
        version_id: root_version_id,
    };

    let deserialized: LazyItemRef<MergedNode> = cache.load_item(file_index).unwrap();
    let mut deserialized_arc = deserialized.item.clone();
    let mut original_arc = lazy_item.item.clone();

    match (original_arc.get(), deserialized_arc.get()) {
        (
            LazyItem::Valid {
                data: Some(original),
                version_id: original_version_id,
                version_number: original_version_number,
                ..
            },
            LazyItem::Valid {
                data: Some(deserialized),
                version_id: deserialized_version_id,
                version_number: deserialized_version_number,
                ..
            },
        ) => {
            let mut original_arc = original.clone();
            let mut deserialized_arc = deserialized.clone();
            let original = original_arc.get();
            let deserialized = deserialized_arc.get();

            assert_eq!(original.hnsw_level, deserialized.hnsw_level);
            assert_eq!(original_version_id, deserialized_version_id);
            assert_eq!(original_version_number, deserialized_version_number);
        }
        _ => panic!("Deserialization mismatch"),
    }
}

#[test]
fn test_eager_lazy_item_serialization() {
    let root_version_number = 0;
    let root_version_id = Hash::from(0);
    let item = EagerLazyItem(
        10.5,
        LazyItem::new(
            root_version_id,
            root_version_number,
            MergedNode::new(HNSWLevel(2)),
        ),
    );

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = item.serialize(bufmans, root_version_id, cursor).unwrap();
    bufman.close_cursor(cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: root_version_number,
        version_id: root_version_id,
    };

    let deserialized: EagerLazyItem<MergedNode, f32> = cache.load_item(file_index).unwrap();

    assert_eq!(item.0, deserialized.0);
    assert_eq!(
        item.1.get_current_version(),
        deserialized.1.get_current_version()
    );
    assert_eq!(
        item.1.get_current_version_number(),
        deserialized.1.get_current_version_number()
    );

    if let (Some(mut node_arc), Some(mut deserialized_arc)) =
        (item.1.get_lazy_data(), deserialized.1.get_lazy_data())
    {
        let node = node_arc.get();
        let deserialized = deserialized_arc.get();

        assert_eq!(node.hnsw_level, deserialized.hnsw_level);
    } else {
        panic!("Deserialization mismatch");
    }
}

#[test]
fn test_lazy_item_set_serialization() {
    let root_version_id = Hash::from(0);
    let lazy_items = LazyItemSet::new();

    lazy_items.insert(LazyItem::from_data(
        1.into(),
        1,
        MergedNode::new(HNSWLevel(2)),
    ));
    lazy_items.insert(LazyItem::from_data(
        2.into(),
        2,
        MergedNode::new(HNSWLevel(2)),
    ));

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = lazy_items
        .serialize(bufmans, root_version_id, cursor)
        .unwrap();
    bufman.close_cursor(cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };

    let deserialized: LazyItemSet<MergedNode> = cache.load_item(file_index).unwrap();

    assert_eq!(lazy_items.len(), deserialized.len());
    for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
        match (original, deserialized) {
            (
                LazyItem::Valid {
                    data: Some(mut original_arc),
                    version_id: original_version_id,
                    version_number: original_version_number,
                    ..
                },
                LazyItem::Valid {
                    data: Some(mut deserialized_arc),
                    version_id: deserialized_version_id,
                    version_number: deserialized_version_number,
                    ..
                },
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();

                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
                assert_eq!(original_version_id, deserialized_version_id);
                assert_eq!(original_version_number, deserialized_version_number);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

#[test]
fn test_eager_lazy_item_set_serialization() {
    let root_version_id = Hash::from(0);
    let lazy_items = EagerLazyItemSet::new();
    lazy_items.insert(EagerLazyItem(
        1.0,
        LazyItem::from_data(1.into(), 1, MergedNode::new(HNSWLevel(2))),
    ));
    lazy_items.insert(EagerLazyItem(
        2.5,
        LazyItem::from_data(2.into(), 2, MergedNode::new(HNSWLevel(2))),
    ));

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = lazy_items
        .serialize(bufmans, root_version_id, cursor)
        .unwrap();
    bufman.close_cursor(cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };

    let deserialized: EagerLazyItemSet<MergedNode, f32> = cache.load_item(file_index).unwrap();

    assert_eq!(lazy_items.len(), deserialized.len());
    for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
        match (original, deserialized) {
            (
                EagerLazyItem(
                    original_data,
                    LazyItem::Valid {
                        data: Some(mut original_arc),
                        version_id: original_version_id,
                        version_number: original_version_number,
                        ..
                    },
                ),
                EagerLazyItem(
                    deserialized_data,
                    LazyItem::Valid {
                        data: Some(mut deserialized_arc),
                        version_id: deserialized_version_id,
                        version_number: deserialized_version_number,
                        ..
                    },
                ),
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();

                assert_eq!(original_data, deserialized_data);
                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
                assert_eq!(original_version_id, deserialized_version_id);
                assert_eq!(original_version_number, deserialized_version_number);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

#[test]
fn test_merged_node_acyclic_serialization() {
    let root_version_id = Hash::from(0);
    let node = MergedNode::new(HNSWLevel(2));

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = node.serialize(bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: MergedNode = cache.load_item(file_index).unwrap();

    assert_eq!(node.hnsw_level, deserialized.hnsw_level);
    assert!(deserialized.get_parent().is_invalid());
    assert!(deserialized.get_child().is_invalid());
    assert_eq!(deserialized.get_neighbors().len(), 0);
}

#[test]
fn test_merged_node_with_neighbors_serialization() {
    let root_version_id = Hash::from(0);
    let node = MergedNode::new(HNSWLevel(2));

    let neighbor1 = LazyItem::from_data(1.into(), 1, MergedNode::new(HNSWLevel(1)));
    let neighbor2 = LazyItem::from_data(2.into(), 2, MergedNode::new(HNSWLevel(1)));
    node.add_ready_neighbor(
        neighbor1,
        MetricResult::CosineSimilarity(CosineSimilarity(0.9)),
    );
    node.add_ready_neighbor(
        neighbor2,
        MetricResult::CosineSimilarity(CosineSimilarity(0.9)),
    );

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = node.serialize(bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: MergedNode = cache.load_item(file_index).unwrap();

    let original_neighbors = node.get_neighbors();
    let deserialized_neighbors = deserialized.get_neighbors();

    // Additional checks
    assert_eq!(original_neighbors.len(), deserialized_neighbors.len());

    for (original, deserialized) in original_neighbors.iter().zip(deserialized_neighbors.iter()) {
        match (original, deserialized) {
            (
                EagerLazyItem(
                    original_cs,
                    LazyItem::Valid {
                        data: Some(mut original_arc),
                        version_id: original_version_id,
                        version_number: original_version_number,
                        ..
                    },
                ),
                EagerLazyItem(
                    deserialized_cs,
                    LazyItem::Valid {
                        data: Some(mut deserialized_arc),
                        version_id: deserialized_version_id,
                        version_number: deserialized_version_number,
                        ..
                    },
                ),
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();

                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
                assert_eq!(original_cs, deserialized_cs);
                assert_eq!(original_version_id, deserialized_version_id);
                assert_eq!(original_version_number, deserialized_version_number);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

#[test]
fn test_merged_node_with_parent_child_serialization() {
    let root_version_id = Hash::from(0);
    let node = MergedNode::new(HNSWLevel(2));
    let parent = LazyItem::new(1.into(), 1, MergedNode::new(HNSWLevel(3)));
    let child = LazyItem::new(2.into(), 2, MergedNode::new(HNSWLevel(1)));

    node.set_parent(parent);
    node.set_child(child);

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = node.serialize(bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: MergedNode = cache.load_item(file_index).unwrap();

    assert!(matches!(
        deserialized.get_parent().item.get(),
        LazyItem::Valid {
            data: Some(_),
            version_number: 1,
            ..
        }
    ));
    assert!(matches!(
        deserialized.get_child().item.get(),
        LazyItem::Valid {
            data: Some(_),
            version_number: 2,
            ..
        }
    ));
}

#[test]
fn test_lazy_item_with_versions_serialization() {
    let temp_dir = tempdir().unwrap();
    let env = Arc::new(
        Environment::new()
            .set_max_dbs(2)
            .set_map_size(10485760) // 10MB
            .open(temp_dir.as_ref())
            .unwrap(),
    );
    let vcs = Arc::new(VersionControl::new(env).unwrap());

    let v0_hash = vcs.generate_hash("main", 0.into()).unwrap();
    let node_v0 = LazyItem::new(v0_hash, 0, MergedNode::new(HNSWLevel(2)));

    let v1_hash = vcs.add_next_version("main").unwrap();
    let node_v1 = LazyItem::new(v1_hash, 1, MergedNode::new(HNSWLevel(2)));
    node_v0.add_version(node_v1);

    let v2_hash = vcs.add_next_version("main").unwrap();
    let node_v2 = LazyItem::new(v2_hash, 2, MergedNode::new(HNSWLevel(2)));
    node_v0.add_version(node_v2);

    let bufmans = Arc::new(BufferManagerFactory::new(
        temp_dir.as_ref().into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));
    let cache = get_cache(bufmans.clone());
    let bufman = bufmans.get(&v0_hash).unwrap();
    let cursor = bufman.open_cursor().unwrap();

    let offset = node_v0.serialize(bufmans, v0_hash, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: v0_hash,
    };

    let deserialized: LazyItem<MergedNode> = cache.load_item(file_index).unwrap();

    assert_eq!(
        node_v0.get_versions().unwrap().len(),
        deserialized.get_versions().unwrap().len()
    );
    assert_eq!(
        node_v0.get_current_version(),
        deserialized.get_current_version(),
    );
    assert_eq!(
        node_v0.get_current_version_number(),
        deserialized.get_current_version_number()
    );
}

#[test]
fn test_lazy_item_cyclic_serialization() {
    let root_version = Hash::from(0);
    let node0 = LazyItem::new(root_version, 0, MergedNode::new(HNSWLevel(2)));
    let node1 = LazyItem::new(1.into(), 1, MergedNode::new(HNSWLevel(2)));

    node0
        .get_lazy_data()
        .unwrap()
        .get()
        .set_parent(node1.clone());
    node1
        .get_lazy_data()
        .unwrap()
        .get()
        .set_child(node0.clone());

    let lazy_ref = LazyItemRef::from_lazy(node0.clone());

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version);

    let offset = lazy_ref.serialize(bufmans, root_version, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: LazyItem<MergedNode> = cache.load_item(file_index).unwrap();

    let mut parent_ref = deserialized.get_lazy_data().unwrap().get_parent();

    // Deserialize the parent
    if let LazyItem::Valid {
        data: Some(mut parent_arc),
        version_number,
        ..
    } = parent_ref.item.get().clone()
    {
        assert_eq!(version_number, 1);
        let parent = parent_arc.get();

        let mut child_ref = parent.get_child();
        let child = child_ref.item.get();

        assert!(matches!(
            child,
            LazyItem::Valid {
                data: None,
                version_number: 0,
                ..
            }
        ));
    } else {
        panic!("Expected lazy load for parent");
    }
}

#[test]
fn test_lazy_item_complex_cyclic_serialization() {
    let root_version_id = Hash::from(0);
    let mut node0 = ArcShift::new(MergedNode::new(HNSWLevel(2)));
    let mut node1 = ArcShift::new(MergedNode::new(HNSWLevel(2)));
    let mut node2 = ArcShift::new(MergedNode::new(HNSWLevel(2)));

    let lazy1 = LazyItem::from_arcshift(root_version_id, 0, node0.clone());
    let lazy2 = LazyItem::from_arcshift(1.into(), 1, node1.clone());
    let lazy3 = LazyItem::from_arcshift(2.into(), 2, node2.clone());

    node0.get().set_parent(lazy2.clone());
    node1.get().set_child(lazy1.clone());
    node1.get().set_parent(lazy3.clone());
    node2.get().set_child(lazy2.clone());

    node0.get().add_ready_neighbor(
        LazyItem::from_arcshift(2.into(), 2, node2),
        MetricResult::CosineSimilarity(CosineSimilarity(0.9)),
    );

    let lazy_ref = LazyItemRef::from_lazy(lazy1);

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = lazy_ref
        .serialize(bufmans, root_version_id, cursor)
        .unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: LazyItemRef<MergedNode> = cache.clone().load_item(file_index).unwrap();

    let mut deserialized_data_arc = deserialized.get_data().unwrap();
    let deserialized_data = deserialized_data_arc.get();

    let mut parent_ref = deserialized_data.get_parent();
    let parent = parent_ref.item.get();

    // Deserialize the parent
    if let LazyItem::Valid {
        data: Some(mut parent_arc),
        ..
    } = parent.clone()
    {
        let parent = parent_arc.get();

        let mut child_ref = parent.get_child();
        let child = child_ref.item.get();

        let mut grand_parent_ref = parent.get_parent();

        let grand_parent = grand_parent_ref.item.get();

        if let LazyItem::Valid {
            data: None,
            file_index,
            ..
        } = &child
        {
            let file_index = file_index.clone().get().clone().unwrap();
            let _: LazyItemRef<MergedNode> = cache.load_item(file_index).unwrap();
        } else {
            panic!("Deserialization mismatch");
        }

        assert!(matches!(
            grand_parent,
            LazyItem::Valid { data: Some(_), .. }
        ));
    } else {
        panic!("Deserialization Error");
    }
}

#[test]
fn test_lazy_item_set_linked_chunk_serialization() {
    let root_version_id = Hash::from(0);
    let lazy_items = LazyItemSet::new();
    for i in 1..13 {
        lazy_items.insert(LazyItem::from_data(
            i.into(),
            i as u16,
            MergedNode::new(HNSWLevel(2)),
        ));
    }

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = lazy_items
        .serialize(bufmans, root_version_id, cursor)
        .unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: LazyItemSet<MergedNode> = cache.load_item(file_index).unwrap();

    assert_eq!(lazy_items.len(), deserialized.len());
    for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
        match (original, deserialized) {
            (
                LazyItem::Valid {
                    data: Some(mut original_arc),
                    version_id: original_version_id,
                    version_number: original_version_number,
                    ..
                },
                LazyItem::Valid {
                    data: Some(mut deserialized_arc),
                    version_id: deserialized_version_id,
                    version_number: deserialized_version_number,
                    ..
                },
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();

                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
                assert_eq!(original_version_id, deserialized_version_id);
                assert_eq!(original_version_number, deserialized_version_number);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

#[test]
fn test_eager_lazy_item_set_linked_chunk_serialization() {
    let root_version_id = Hash::from(0);
    let lazy_items = EagerLazyItemSet::new();
    for i in 1..13 {
        lazy_items.insert(EagerLazyItem(
            3.4,
            LazyItem::from_data(i.into(), i as u16, MergedNode::new(HNSWLevel(2))),
        ));
    }

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = lazy_items
        .serialize(bufmans, root_version_id, cursor)
        .unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: EagerLazyItemSet<MergedNode, f32> = cache.load_item(file_index).unwrap();

    assert_eq!(lazy_items.len(), deserialized.len());
    for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
        match (original, deserialized) {
            (
                EagerLazyItem(
                    original_data,
                    LazyItem::Valid {
                        data: Some(mut original_arc),
                        version_id: original_version_id,
                        version_number: original_version_number,
                        ..
                    },
                ),
                EagerLazyItem(
                    deserialized_data,
                    LazyItem::Valid {
                        data: Some(mut deserialized_arc),
                        version_id: deserialized_version_id,
                        version_number: deserialized_version_number,
                        ..
                    },
                ),
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();

                assert_eq!(original_data, deserialized_data);
                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
                assert_eq!(original_version_id, deserialized_version_id);
                assert_eq!(original_version_number, deserialized_version_number);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

fn validate_lazy_item_versions(
    cache: Arc<NodeRegistry>,
    lazy_item: LazyItem<MergedNode>,
    version_number: u16,
) {
    let versions = lazy_item.get_versions().unwrap();

    for i in 0..versions.len() {
        let version = versions.get(i).unwrap();
        let version = if version.get_lazy_data().is_none() {
            let file_index = version.get_file_index().unwrap();
            cache.clone().load_item(file_index).unwrap()
        } else {
            version
        };

        let current_version_number = version.get_current_version_number();

        assert_eq!(current_version_number - version_number, 4_u16.pow(i as u32));
        validate_lazy_item_versions(cache.clone(), version, current_version_number);
    }
}

#[test]
fn test_lazy_item_with_versions_serialization_and_validation() {
    let temp_dir = tempdir().unwrap();
    let env = Arc::new(
        Environment::new()
            .set_max_dbs(2)
            .set_map_size(10485760) // 10MB
            .open(temp_dir.as_ref())
            .unwrap(),
    );
    let vcs = Arc::new(VersionControl::new(env).unwrap());

    let v0_hash = vcs.generate_hash("main", Version::from(0)).unwrap();
    let root = LazyItem::new(v0_hash, 0, MergedNode::new(HNSWLevel(0)));

    let bufmans = Arc::new(BufferManagerFactory::new(
        temp_dir.as_ref().into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));
    let cache = get_cache(bufmans.clone());
    let bufman = bufmans.get(&v0_hash).unwrap();
    let cursor = bufman.open_cursor().unwrap();

    for i in 1..=100 {
        let hash = vcs.add_next_version("main").unwrap();
        let next_version = LazyItem::new(hash, i, MergedNode::new(HNSWLevel(0)));
        root.add_version(next_version);
    }

    validate_lazy_item_versions(cache.clone(), root.clone(), 0);

    let offset = root.serialize(bufmans.clone(), v0_hash, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: v0_hash,
    };
    bufman.close_cursor(cursor).unwrap();
    bufmans.flush_all().unwrap();

    let deserialized: LazyItem<MergedNode> = cache.clone().load_item(file_index).unwrap();

    validate_lazy_item_versions(cache, deserialized.clone(), 0);
    assert_eq!(deserialized.get_versions().unwrap().len(), 4);
}

#[test]
fn test_lazy_item_with_versions_multiple_serialization() {
    let temp_dir = tempdir().unwrap();
    let env = Arc::new(
        Environment::new()
            .set_max_dbs(2)
            .set_map_size(10485760) // 10MB
            .open(temp_dir.as_ref())
            .unwrap(),
    );
    let vcs = Arc::new(VersionControl::new(env).unwrap());

    let v0_hash = vcs.generate_hash("main", Version::from(0)).unwrap();
    let root = LazyItem::new(v0_hash, 0, MergedNode::new(HNSWLevel(0)));

    let bufmans = Arc::new(BufferManagerFactory::new(
        temp_dir.as_ref().into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));
    let cache = get_cache(bufmans.clone());
    let bufman = bufmans.get(&v0_hash).unwrap();
    let cursor = bufman.open_cursor().unwrap();

    for i in 1..26 {
        let hash = vcs.add_next_version("main").unwrap();
        let next_version = LazyItem::new(hash, i, MergedNode::new(HNSWLevel(0)));
        root.add_version(next_version);
    }

    validate_lazy_item_versions(cache.clone(), root.clone(), 0);
    root.serialize(bufmans.clone(), v0_hash, cursor).unwrap();
    // 1, 4, 16
    assert_eq!(root.get_versions().unwrap().len(), 3);

    for i in 26..51 {
        let hash = vcs.add_next_version("main").unwrap();
        let next_version = LazyItem::new(hash, i, MergedNode::new(HNSWLevel(0)));
        root.add_version(next_version);
    }

    validate_lazy_item_versions(cache.clone(), root.clone(), 0);
    root.set_versions_persistence(true);
    root.serialize(bufmans.clone(), v0_hash, cursor).unwrap();
    // 1, 4, 16
    assert_eq!(root.get_versions().unwrap().len(), 3);

    for i in 51..76 {
        let hash = vcs.add_next_version("main").unwrap();
        let next_version = LazyItem::new(hash, i, MergedNode::new(HNSWLevel(0)));
        root.add_version(next_version);
    }

    validate_lazy_item_versions(cache.clone(), root.clone(), 0);
    root.set_versions_persistence(true);

    root.serialize(bufmans.clone(), v0_hash, cursor).unwrap();
    // 1, 4, 16, 64
    assert_eq!(root.get_versions().unwrap().len(), 4);

    for i in 76..101 {
        let hash = vcs.add_next_version("main").unwrap();
        let next_version = LazyItem::new(hash, i, MergedNode::new(HNSWLevel(0)));
        root.add_version(next_version);
    }
    validate_lazy_item_versions(cache.clone(), root.clone(), 0);
    root.set_versions_persistence(true);

    let offset = root.serialize(bufmans.clone(), v0_hash, cursor).unwrap();
    // 1, 4, 16, 64
    assert_eq!(root.get_versions().unwrap().len(), 4);
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: v0_hash,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: LazyItem<MergedNode> = cache.clone().load_item(file_index).unwrap();
    // 1, 4, 16, 64
    assert_eq!(deserialized.get_versions().unwrap().len(), 4);

    validate_lazy_item_versions(cache, deserialized, 0);
}

#[test]
fn test_version_hashing_function_uniqueness() {
    let branch_id = BranchId::new("main");
    let mut versions = HashSet::new();

    for i in 0..1000 {
        let version_hash = VersionHash::new(branch_id, Version::from(i));
        versions.insert(version_hash.calculate_hash());
        // simulate some processing
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    assert_eq!(versions.iter().count(), 1000);
}

#[test]
fn test_storage_serialization() {
    let storages = [
        Storage::UnsignedByte {
            mag: 10,
            quant_vec: vec![0, 1, 4],
        },
        Storage::SubByte {
            mag: 34,
            quant_vec: vec![vec![55, 35], vec![56, 23]],
            resolution: 2,
        },
        Storage::HalfPrecisionFP {
            mag: 4234.34,
            quant_vec: vec![f16::from_f32(534.324), f16::from_f32(6453.3)],
        },
    ];
    let (bufmans, cache, bufman, cursor, _dir) = setup_test(&1.into());
    bufman.close_cursor(cursor).unwrap();

    for (version, storage) in storages.into_iter().enumerate() {
        let version_id = Hash::from(version as u32);
        let bufman = bufmans.get(&version_id).unwrap();
        let cursor = bufman.open_cursor().unwrap();
        let offset = storage
            .serialize(bufmans.clone(), version_id, cursor)
            .unwrap();
        let file_index = FileIndex::Valid {
            offset: FileOffset(offset),
            version_number: version as u16,
            version_id,
        };
        let deserialized: Storage = cache.clone().load_item(file_index).unwrap();

        assert_eq!(deserialized, storage);
    }
}

#[test]
fn test_lazy_item_vec_serialization() {
    let root_version_id = Hash::from(0);
    let lazy_items = LazyItemVec::new();

    lazy_items.push(LazyItem::from_data(
        1.into(),
        1,
        MergedNode::new(HNSWLevel(2)),
    ));
    lazy_items.push(LazyItem::from_data(
        2.into(),
        2,
        MergedNode::new(HNSWLevel(2)),
    ));

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = lazy_items
        .serialize(bufmans, root_version_id, cursor)
        .unwrap();
    bufman.close_cursor(cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };

    let deserialized: LazyItemSet<MergedNode> = cache.load_item(file_index).unwrap();

    assert_eq!(lazy_items.len(), deserialized.len());
    for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
        match (original, deserialized) {
            (
                LazyItem::Valid {
                    data: Some(mut original_arc),
                    ..
                },
                LazyItem::Valid {
                    data: Some(mut deserialized_arc),
                    ..
                },
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();
                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

#[test]
fn test_lazy_item_vec_linked_chunk_serialization() {
    let root_version_id = Hash::from(0);
    let lazy_items = LazyItemVec::new();
    for i in 1..13 {
        lazy_items.push(LazyItem::from_data(
            i.into(),
            i as u16,
            MergedNode::new(HNSWLevel(2)),
        ));
    }

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = lazy_items
        .serialize(bufmans, root_version_id, cursor)
        .unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: LazyItemVec<MergedNode> = cache.load_item(file_index).unwrap();

    assert_eq!(lazy_items.len(), deserialized.len());
    for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
        match (original, deserialized) {
            (
                LazyItem::Valid {
                    data: Some(mut original_arc),
                    ..
                },
                LazyItem::Valid {
                    data: Some(mut deserialized_arc),
                    ..
                },
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();
                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}
