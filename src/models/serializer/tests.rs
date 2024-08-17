use crate::models::lazy_load::*;
use crate::models::serializer::*;
use crate::models::types::*;
use arcshift::ArcShift;
use std::io::Cursor;
use std::sync::Arc;

fn get_cache<R: Read + Seek>(reader: R) -> Arc<NodeRegistry<R>> {
    Arc::new(NodeRegistry::new(1000, reader))
}

#[test]
fn test_lazy_item_serialization() {
    let node = MergedNode::new(2);
    let lazy_item = LazyItemRef::new(1, node);

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_item.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 1 };

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
    let item = EagerLazyItem(10.5, LazyItem::new(1, MergedNode::new(2)));

    let mut writer = Cursor::new(Vec::new());
    let offset = item.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 1 };

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
    lazy_items.insert(LazyItem::from_data(1, MergedNode::new(2)));
    lazy_items.insert(LazyItem::from_data(2, MergedNode::new(2)));

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_items.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 0 };

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
        LazyItem::from_data(1, MergedNode::new(2)),
    ));
    lazy_items.insert(EagerLazyItem(
        2.5,
        LazyItem::from_data(2, MergedNode::new(2)),
    ));

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_items.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 0 };

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
    let node = MergedNode::new(2);

    let mut writer = Cursor::new(Vec::new());
    let offset = node.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 0 };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: MergedNode = cache.load_item(file_index).unwrap();

    assert_eq!(node.hnsw_level, deserialized.hnsw_level);
    assert!(deserialized.get_parent().is_invalid());
    assert!(deserialized.get_child().is_invalid());
    assert_eq!(deserialized.get_neighbors().len(), 0);
    assert_eq!(deserialized.get_versions().len(), 0);
}

#[test]
fn test_merged_node_with_neighbors_serialization() {
    let node = MergedNode::new(2);

    let neighbor1 = LazyItem::from_data(2, MergedNode::new(1));
    let neighbor2 = LazyItem::from_data(3, MergedNode::new(1));
    node.add_ready_neighbor(neighbor1, 0.9);
    node.add_ready_neighbor(neighbor2, 0.8);

    let mut writer = Cursor::new(Vec::new());
    let offset = node.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 0 };

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
    let node = MergedNode::new(2);
    let parent = LazyItem::new(2, MergedNode::new(3));
    let child = LazyItem::new(3, MergedNode::new(1));

    node.set_parent(parent);
    node.set_child(child);

    let mut writer = Cursor::new(Vec::new());
    let offset = node.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 0 };

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
fn test_merged_node_with_versions_serialization() {
    let node = Arc::new(MergedNode::new(2));
    let version1 = ArcShift::new(MergedNode::new(2));
    let version2 = ArcShift::new(MergedNode::new(2));

    node.add_version(2, version1);
    node.add_version(3, version2);

    let mut writer = Cursor::new(Vec::new());
    let offset = node.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 0 };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: MergedNode = cache.load_item(file_index).unwrap();

    assert_eq!(node.get_versions().len(), deserialized.get_versions().len());
}

#[test]
fn test_merged_node_cyclic_serialization() {
    let node1 = LazyItem::new(1, MergedNode::new(2));
    let node2 = LazyItem::new(2, MergedNode::new(2));

    node1.get_data().unwrap().get().set_parent(node2.clone());
    node2.get_data().unwrap().get().set_child(node1.clone());

    let lazy_ref = LazyItemRef::from_lazy(node1.clone());

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_ref.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 1 };

    let reader = Cursor::new(writer.into_inner());
    let cache = get_cache(reader);
    let deserialized: MergedNode = cache.load_item(file_index).unwrap();

    let mut parent_ref = deserialized.get_parent();

    // Deserialize the parent
    if let LazyItem::Valid {
        data: Some(mut parent_arc),
        ..
    } = parent_ref.item.get().clone()
    {
        let parent = parent_arc.get();

        let mut child_ref = parent.get_child();
        let child = child_ref.item.get();

        assert!(matches!(child, LazyItem::Valid { data: Some(_), .. }));
    } else {
        panic!("Expected lazy load for parent");
    }
}

#[test]
fn test_merged_node_complex_cyclic_serialization() {
    let mut node1 = ArcShift::new(MergedNode::new(2));
    let mut node2 = ArcShift::new(MergedNode::new(2));
    let mut node3 = ArcShift::new(MergedNode::new(2));

    let lazy1 = LazyItem::from_arcshift(1, node1.clone());
    let lazy2 = LazyItem::from_arcshift(2, node2.clone());
    let lazy3 = LazyItem::from_arcshift(3, node3.clone());

    node1.get().set_parent(lazy2.clone());
    node2.get().set_child(lazy1.clone());
    node2.get().set_parent(lazy3.clone());
    node3.get().set_child(lazy2.clone());
    node1
        .get()
        .add_ready_neighbor(LazyItem::from_arcshift(3, node3), 0.9);

    let lazy_ref = LazyItemRef::from_lazy(lazy1);

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_ref.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 1 };

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
            let _: MergedNode = cache.load_item(file_index).unwrap();
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
        lazy_items.insert(LazyItem::from_data(i, MergedNode::new(2)));
    }

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_items.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 0 };

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
            LazyItem::from_data(i, MergedNode::new(2)),
        ));
    }

    let mut writer = Cursor::new(Vec::new());
    let offset = lazy_items.serialize(&mut writer).unwrap();
    let file_index = FileIndex::Valid { offset, version: 0 };

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
