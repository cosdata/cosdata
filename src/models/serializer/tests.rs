use crate::distance::cosine::CosineSimilarity;
use crate::models::identity_collections::IdentityMapKey;
use crate::models::lazy_load::*;
use crate::models::serializer::*;
use crate::models::types::*;
use crate::models::versioning::BranchId;
use crate::models::versioning::VersionHash;
use crate::models::versioning::{Version, VersionControl};
use arcshift::ArcShift;
use lmdb::Environment;
use std::io::Cursor;
use std::sync::Arc;
use tempfile::tempdir;

fn get_cache<R: Read + Seek>(reader: R) -> Arc<NodeRegistry<R>> {
    Arc::new(NodeRegistry::new(1000, reader))
}

#[test]
fn test_lazy_item_serialization() {
    let node = MergedNode::new(HNSWLevel(2));
    let lazy_item = LazyItemRef::new(1.into(), node);

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_item.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 1.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: LazyItemRef<MergedNode> = cache.load_item(file_index).unwrap();
    let mut deserialized_arc = deserialized.item.clone();
    let mut original_arc = lazy_item.item.clone();

    match (original_arc.get(), deserialized_arc.get()) {
        (
            LazyItem::Valid {
                data: Some(original),
                ..
            },
            LazyItem::Valid {
                data: Some(deserialized),
                ..
            },
        ) => {
            let mut original_arc = original.clone();
            let mut deserialized_arc = deserialized.clone();
            let original = original_arc.get();
            let deserialized = deserialized_arc.get();
            assert_eq!(original.hnsw_level, deserialized.hnsw_level);
        }
        _ => panic!("Deserialization mismatch"),
    }
}

#[test]
fn test_eager_lazy_item_serialization() {
    let item = EagerLazyItem(10.5, LazyItem::new(1.into(), MergedNode::new(HNSWLevel(2))));

    let mut writer = Cursor::new(Vec::new());
    let offset = item.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 1.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: EagerLazyItem<MergedNode, f32> = cache.load_item(file_index).unwrap();

    assert_eq!(item.0, deserialized.0);

    if let (Some(mut node_arc), Some(mut deserialized_arc)) =
        (item.1.get_data(), deserialized.1.get_data())
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
    let lazy_items = LazyItemSet::new();

    lazy_items.insert(LazyItem::from_data(1.into(), MergedNode::new(HNSWLevel(2))));
    lazy_items.insert(LazyItem::from_data(2.into(), MergedNode::new(HNSWLevel(2))));

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_items.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 0.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
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
fn test_eager_lazy_item_set_serialization() {
    let lazy_items = EagerLazyItemSet::new();
    lazy_items.insert(EagerLazyItem(
        1.0,
        LazyItem::from_data(1.into(), MergedNode::new(HNSWLevel(2))),
    ));
    lazy_items.insert(EagerLazyItem(
        2.5,
        LazyItem::from_data(2.into(), MergedNode::new(HNSWLevel(2))),
    ));

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_items.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 0.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: EagerLazyItemSet<MergedNode, f32> = cache.load_item(file_index).unwrap();

    assert_eq!(lazy_items.len(), deserialized.len());
    for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
        match (original, deserialized) {
            (
                EagerLazyItem(
                    original_data,
                    LazyItem::Valid {
                        data: Some(mut original_arc),
                        ..
                    },
                ),
                EagerLazyItem(
                    deserialized_data,
                    LazyItem::Valid {
                        data: Some(mut deserialized_arc),
                        ..
                    },
                ),
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();
                assert_eq!(original_data, deserialized_data);
                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

#[test]
fn test_merged_node_acyclic_serialization() {
    let node = MergedNode::new(HNSWLevel(2));

    let mut writer = Cursor::new(Vec::new());
    let offset = node.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 0.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: MergedNode = cache.load_item(file_index).unwrap();

    assert_eq!(node.hnsw_level, deserialized.hnsw_level);
    assert!(deserialized.get_parent().is_invalid());
    assert!(deserialized.get_child().is_invalid());
    assert_eq!(deserialized.get_neighbors().len(), 0);
}

#[test]
fn test_merged_node_with_neighbors_serialization() {
    let node = MergedNode::new(HNSWLevel(2));

    let neighbor1 = LazyItem::from_data(2.into(), MergedNode::new(HNSWLevel(1)));
    let neighbor2 = LazyItem::from_data(3.into(), MergedNode::new(HNSWLevel(1)));
    node.add_ready_neighbor(
        neighbor1,
        MetricResult::CosineSimilarity(CosineSimilarity(0.9)),
    );
    node.add_ready_neighbor(
        neighbor2,
        MetricResult::CosineSimilarity(CosineSimilarity(0.9)),
    );

    let mut writer = Cursor::new(Vec::new());
    let offset = node.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 0.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
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
                        ..
                    },
                ),
                EagerLazyItem(
                    deserialized_cs,
                    LazyItem::Valid {
                        data: Some(mut deserialized_arc),
                        ..
                    },
                ),
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();

                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
                assert_eq!(original_cs, deserialized_cs);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

#[test]
fn test_merged_node_with_parent_child_serialization() {
    let node = MergedNode::new(HNSWLevel(2));
    let parent = LazyItem::new(2.into(), MergedNode::new(HNSWLevel(3)));
    let child = LazyItem::new(3.into(), MergedNode::new(HNSWLevel(1)));

    node.set_parent(parent);
    node.set_child(child);

    let mut writer = Cursor::new(Vec::new());
    let offset = node.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 0.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: MergedNode = cache.load_item(file_index).unwrap();

    assert!(matches!(
        deserialized.get_parent().item.get(),
        LazyItem::Valid { data: Some(_), .. }
    ));
    assert!(matches!(
        deserialized.get_child().item.get(),
        LazyItem::Valid { data: Some(_), .. }
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
    let branch_id = BranchId::new("main");

    let v0_hash = vcs.generate_hash("main", 0.into()).unwrap();
    let node_v0 = LazyItem::new(v0_hash, MergedNode::new(HNSWLevel(2)));

    let v1_hash = vcs.add_next_version("main").unwrap();
    let node_v1 = LazyItem::new(v1_hash, MergedNode::new(HNSWLevel(2)));
    node_v0.add_version(branch_id, 1, node_v1).unwrap();

    let v2_hash = vcs.add_next_version("main").unwrap();
    let node_v2 = LazyItem::new(v2_hash, MergedNode::new(HNSWLevel(2)));
    node_v0.add_version(branch_id, 2, node_v2).unwrap();

    let mut writer = Cursor::new(Vec::new());
    let offset = node_v0.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 0.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: LazyItem<MergedNode> = cache.load_item(file_index).unwrap();

    assert_eq!(
        node_v0.get_versions().unwrap().len(),
        deserialized.get_versions().unwrap().len()
    );
}

#[test]
fn test_lazy_item_cyclic_serialization() {
    let node1 = LazyItem::new(1.into(), MergedNode::new(HNSWLevel(2)));
    let node2 = LazyItem::new(2.into(), MergedNode::new(HNSWLevel(2)));

    node1.get_data().unwrap().get().set_parent(node2.clone());
    node2.get_data().unwrap().get().set_child(node1.clone());

    let lazy_ref = LazyItemRef::from_lazy(node1.clone());

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_ref.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 1.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: LazyItem<MergedNode> = cache.load_item(file_index).unwrap();

    let mut parent_ref = deserialized.get_data().unwrap().get_parent();

    // Deserialize the parent
    if let LazyItem::Valid {
        data: Some(mut parent_arc),
        ..
    } = parent_ref.item.get().clone()
    {
        let parent = parent_arc.get();

        let mut child_ref = parent.get_child();
        let child = child_ref.item.get();

        assert!(matches!(child, LazyItem::Valid { data: None, .. }));
    } else {
        panic!("Expected lazy load for parent");
    }
}

#[test]
fn test_lazy_item_complex_cyclic_serialization() {
    let mut node1 = ArcShift::new(MergedNode::new(HNSWLevel(2)));
    let mut node2 = ArcShift::new(MergedNode::new(HNSWLevel(2)));
    let mut node3 = ArcShift::new(MergedNode::new(HNSWLevel(2)));

    let lazy1 = LazyItem::from_arcshift(1.into(), node1.clone());
    let lazy2 = LazyItem::from_arcshift(2.into(), node2.clone());
    let lazy3 = LazyItem::from_arcshift(3.into(), node3.clone());

    node1.get().set_parent(lazy2.clone());
    node2.get().set_child(lazy1.clone());
    node2.get().set_parent(lazy3.clone());
    node3.get().set_child(lazy2.clone());

    node1.get().add_ready_neighbor(
        LazyItem::from_arcshift(3.into(), node3),
        MetricResult::CosineSimilarity(CosineSimilarity(0.9)),
    );

    let lazy_ref = LazyItemRef::from_lazy(lazy1);

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_ref.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 1.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
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
    let lazy_items = LazyItemSet::new();
    for i in 1..13 {
        lazy_items.insert(LazyItem::from_data(i.into(), MergedNode::new(HNSWLevel(2))));
    }

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_items.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 0.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
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
fn test_eager_lazy_item_set_linked_chunk_serialization() {
    let lazy_items = EagerLazyItemSet::new();
    for i in 1..13 {
        lazy_items.insert(EagerLazyItem(
            3.4,
            LazyItem::from_data(i.into(), MergedNode::new(HNSWLevel(2))),
        ));
    }

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_items.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: 0.into(),
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: EagerLazyItemSet<MergedNode, f32> = cache.load_item(file_index).unwrap();

    assert_eq!(lazy_items.len(), deserialized.len());
    for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
        match (original, deserialized) {
            (
                EagerLazyItem(
                    original_data,
                    LazyItem::Valid {
                        data: Some(mut original_arc),
                        ..
                    },
                ),
                EagerLazyItem(
                    deserialized_data,
                    LazyItem::Valid {
                        data: Some(mut deserialized_arc),
                        ..
                    },
                ),
            ) => {
                let original = original_arc.get();
                let deserialized = deserialized_arc.get();
                assert_eq!(original_data, deserialized_data);
                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }
}

fn validate_lazy_item_versions<R: Read + Seek>(
    vcs: Arc<VersionControl>,
    cache: Arc<NodeRegistry<R>>,
    lazy_item: LazyItem<MergedNode>,
    current_version_hash: VersionHash,
) {
    let versions = lazy_item.get_versions().unwrap();

    for i in 0.. {
        let key = IdentityMapKey::Int(i);
        if let Some(version) = versions.get(&key) {
            let version = if version.get_data().is_none() {
                let file_index = version.get_file_index().unwrap();
                cache.clone().load_item(file_index).unwrap()
            } else {
                version
            };

            let version_hash = vcs
                .get_version_hash(&version.get_current_version())
                .unwrap()
                .unwrap();

            assert_eq!(
                *version_hash.version - *current_version_hash.version,
                4_u32.pow(i)
            );
            validate_lazy_item_versions(vcs.clone(), cache.clone(), version, version_hash);
        } else {
            break;
        }
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
    let branch_id = BranchId::new("main");

    let v0_hash = vcs.generate_hash("main", Version::from(0)).unwrap();
    let root = LazyItem::new(v0_hash, MergedNode::new(HNSWLevel(0)));

    for i in 0..100 {
        let hash = vcs.add_next_version("main").unwrap();
        let next_version = LazyItem::new(hash, MergedNode::new(HNSWLevel(0)));
        root.add_version(branch_id, i + 1, next_version).unwrap();
    }

    let root_version_hash = vcs.get_version_hash(&v0_hash).unwrap().unwrap();
    let empty_reader = Cursor::new(Vec::new());
    let empty_cache = get_cache(empty_reader);
    validate_lazy_item_versions(
        vcs.clone(),
        empty_cache,
        root.clone(),
        root_version_hash.clone(),
    );

    let mut writer = Cursor::new(Vec::new());
    let offset = root.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version: v0_hash,
    };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: LazyItem<MergedNode> = cache.clone().load_item(file_index).unwrap();

    validate_lazy_item_versions(vcs, cache, deserialized, root_version_hash);
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
