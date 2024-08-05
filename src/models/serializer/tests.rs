#[cfg(test)]
mod tests {
    use crate::models::chunked_list::*;
    use crate::models::serializer::*;
    use crate::models::types::*;
    use std::io::Cursor;
    use std::sync::Arc;
    use std::sync::RwLock;

    // Helper function to create a sample VectorQt
    fn sample_vector_qt() -> VectorQt {
        VectorQt::UnsignedByte {
            mag: 100,
            quant_vec: vec![1, 2, 3, 4, 5],
        }
    }

    fn get_cache<R: Read + Seek>(reader: R) -> Arc<NodeRegistry<R>> {
        Arc::new(NodeRegistry::new(1000, reader))
    }

    // Helper function to create a simple MergedNode
    fn simple_merged_node(version_id: VersionId, hnsw_level: HNSWLevel) -> MergedNode {
        MergedNode::new(version_id, hnsw_level)
    }

    #[test]
    fn test_lazy_item_serialization() {
        let node = simple_merged_node(1, 2);
        let lazy_item = LazyItemRef::new(node);

        let mut writer = Cursor::new(Vec::new());
        let offset = lazy_item.serialize(&mut writer).unwrap();

        let reader = Cursor::new(writer.into_inner());
        let cache = get_cache(reader);
        let deserialized: LazyItemRef<MergedNode> = cache.load_item(offset).unwrap();
        let deserialized_guard = deserialized.item.read().unwrap();
        let original_guard = lazy_item.item.read().unwrap();

        match (&*original_guard, &*deserialized_guard) {
            (
                LazyItem {
                    data: Some(original),
                    ..
                },
                LazyItem {
                    data: Some(deserialized),
                    ..
                },
            ) => {
                let original_guard = original.read().unwrap();
                let deserialized_guard = deserialized.read().unwrap();
                assert_eq!(original_guard.version_id, deserialized_guard.version_id);
                assert_eq!(original_guard.hnsw_level, deserialized_guard.hnsw_level);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }

    #[test]
    fn test_lazy_items_serialization() {
        let mut lazy_items = LazyItems::new();
        lazy_items.push(LazyItem {
            data: Some(Item::new(simple_merged_node(1, 2))),
            offset: None,
            decay_counter: 0,
        });
        lazy_items.push(LazyItem {
            data: Some(Item::new(simple_merged_node(2, 2))),
            offset: None,
            decay_counter: 0,
        });

        let mut writer = Cursor::new(Vec::new());
        let offset = lazy_items.serialize(&mut writer).unwrap();

        let reader = Cursor::new(writer.into_inner());
        let cache = get_cache(reader);
        let mut deserialized: LazyItems<MergedNode> = cache.load_item(offset).unwrap();

        assert_eq!(lazy_items.len(), deserialized.len());
        for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
            match (original, deserialized) {
                (
                    LazyItem {
                        data: Some(mut original),
                        ..
                    },
                    LazyItem {
                        data: Some(mut deserialized),
                        ..
                    },
                ) => {
                    let original_guard = original.get();
                    let deserialized_guard = deserialized.get();
                    assert_eq!(original_guard.version_id, deserialized_guard.version_id);
                    assert_eq!(original_guard.hnsw_level, deserialized_guard.hnsw_level);
                }
                _ => panic!("Deserialization mismatch"),
            }
        }
    }

    #[test]
    fn test_merged_node_acyclic_serialization() {
        let node = simple_merged_node(1, 2);

        let mut writer = Cursor::new(Vec::new());
        let offset = node.serialize(&mut writer).unwrap();

        let reader = Cursor::new(writer.into_inner());
        let cache = get_cache(reader);
        let deserialized: MergedNode = cache.load_item(offset).unwrap();

        assert_eq!(node.version_id, deserialized.version_id);
        assert_eq!(node.hnsw_level, deserialized.hnsw_level);
        assert!(deserialized.get_parent().is_none());
        assert!(deserialized.get_child().is_none());
        assert_eq!(deserialized.get_neighbors().len(), 0);
        assert_eq!(deserialized.get_versions().len(), 0);
    }

    #[test]
    fn test_merged_node_with_neighbors_serialization() {
        let node = MergedNode::new(1, 2);

        let neighbor1 = LazyItem::with_data(MergedNode::new(2, 1));
        let neighbor2 = LazyItem::with_data(MergedNode::new(3, 1));
        node.add_ready_neighbor(neighbor1, 0.9);
        node.add_ready_neighbor(neighbor2, 0.8);

        let mut writer = Cursor::new(Vec::new());
        let offset = node.serialize(&mut writer).unwrap();

        let reader = Cursor::new(writer.into_inner());
        let cache = get_cache(reader);
        let deserialized: MergedNode = cache.load_item(offset).unwrap();

        let original_neighbors = node.get_neighbors();
        let deserialized_neighbors = deserialized.get_neighbors();

        // Additional checks
        assert_eq!(original_neighbors.len(), deserialized_neighbors.len());

        for (original, deserialized) in original_neighbors.iter().zip(deserialized_neighbors.iter())
        {
            match (original, deserialized) {
                (
                    LazyItem {
                        data: Some(mut original),
                        ..
                    },
                    LazyItem {
                        data: Some(mut deserialized),
                        ..
                    },
                ) => {
                    let original_guard = original.get();
                    let deserialized_guard = deserialized.get();
                    let mut original_node_data = original_guard.node.data.clone().unwrap();
                    let original_node_data_guard = original_node_data.get();
                    let mut deserialized_node_data = deserialized_guard.node.data.clone().unwrap();
                    let deserialized_node_data_guard = deserialized_node_data.get();

                    assert_eq!(
                        original_node_data_guard.version_id,
                        deserialized_node_data_guard.version_id
                    );
                    assert_eq!(
                        original_node_data_guard.hnsw_level,
                        deserialized_node_data_guard.hnsw_level
                    );
                    assert_eq!(
                        original_guard.cosine_similarity,
                        deserialized_guard.cosine_similarity
                    );
                }
                _ => panic!("Deserialization mismatch"),
            }
        }
    }

    #[test]
    fn test_merged_node_with_parent_child_serialization() {
        let mut node = MergedNode::new(1, 2);
        let parent = LazyItemRef::new(MergedNode::new(2, 3));
        let child = LazyItemRef::new(MergedNode::new(3, 1));

        // TODO: take a look later
        node.set_parent(Some(parent));
        node.set_child(Some(child));

        let mut writer = Cursor::new(Vec::new());
        let offset = node.serialize(&mut writer).unwrap();

        let reader = Cursor::new(writer.into_inner());
        let cache = get_cache(reader);
        let deserialized: MergedNode = cache.load_item(offset).unwrap();

        assert!(matches!(
            deserialized.get_parent().unwrap().item.get(),
            LazyItem {
                offset: Some(_),
                data: Some(_),
                ..
            }
        ));
        assert!(matches!(
            deserialized.get_child().unwrap().item.get(),
            LazyItem {
                offset: Some(_),
                data: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn test_merged_node_with_versions_serialization() {
        let node = Arc::new(MergedNode::new(1, 2));
        let version1 = Arc::new(RwLock::new(MergedNode::new(2, 2)));
        let version2 = Arc::new(RwLock::new(MergedNode::new(3, 2)));

        node.add_version(version1);
        node.add_version(version2);

        let mut writer = Cursor::new(Vec::new());
        let offset = node.serialize(&mut writer).unwrap();

        let reader = Cursor::new(writer.into_inner());
        let cache = get_cache(reader);
        let deserialized: MergedNode = cache.load_item(offset).unwrap();

        assert_eq!(node.get_versions().len(), deserialized.get_versions().len());
    }

    #[test]
    fn test_merged_node_cyclic_serialization() {
        let mut node1 = Item::new(MergedNode::new(1, 2));
        let mut node2 = Item::new(MergedNode::new(2, 2));
        println!("Created node1 with id: 1 and node2 with id: 2");

        let lazy1 = LazyItemRef::from_item(node1.clone());
        let lazy2 = LazyItemRef::from_item(node2.clone());

        node1.get().set_parent(Some(lazy2.clone()));
        node2.get().set_child(Some(lazy1.clone()));
        println!("Set cyclic references: node1's parent is node2, node2's child is node1");

        let mut writer = Cursor::new(Vec::new());
        let offset = lazy1.serialize(&mut writer).unwrap();
        println!("Serialized lazy1 to writer. Offset: {}", offset);

        let reader = Cursor::new(writer.into_inner());

        let cache = get_cache(reader);

        let deserialized: MergedNode = cache.load_item(offset).unwrap();
        println!(
            "Deserialized MergedNode from cache. Node : {}",
            deserialized
        );

        let parent = deserialized.get_parent().unwrap();
        println!("Got parent from deserialized node: {:?}", parent);

        let parent_guard = parent.item.read().unwrap();

        // Deserialize the parent
        if let LazyItem {
            offset: Some(parent_offset),
            data: Some(parent),
            ..
        } = &*parent_guard
        {
            println!("Parent is a LazyItem : {:?} ", parent);

            let parent_guard = parent.read().unwrap();

            let child = parent_guard.get_child().unwrap();
            println!("Got child from parent: {:?}", child);

            let child_guard = child.item.read().unwrap();

            match &*child_guard {
                LazyItem {
                    data: Some(_),
                    offset: Some(child_offset),
                    ..
                } => println!(
                    "Child is a LazyItem with data and offset: {:?}",
                    child_offset
                ),
                _ => println!("Unexpected child state"),
            }

            assert!(matches!(
                &*child_guard,
                LazyItem {
                    data: Some(_),
                    offset: Some(_),
                    ..
                }
            ));
            println!("Assertion passed: child is a LazyItem with no data and Some offset");
        } else {
            println!("Parent is not in the expected state");
            panic!("Expected lazy load for parent");
        }

        println!("Test completed successfully");
    }

    #[test]
    fn test_merged_node_complex_cyclic_serialization() {
        let node1 = Arc::new(RwLock::new(MergedNode::new(1, 2)));
        let node2 = Arc::new(RwLock::new(MergedNode::new(2, 2)));
        let node3 = Arc::new(RwLock::new(MergedNode::new(3, 2)));

        let lazy1 = LazyItemRef::new_with_lock(node1.clone());
        let lazy2 = LazyItemRef::new_with_lock(node2.clone());
        let lazy3 = LazyItemRef::new_with_lock(node3.clone());

        node1.write().unwrap().set_parent(Some(lazy2.clone()));
        node2.write().unwrap().set_child(Some(lazy1.clone()));
        node2.write().unwrap().set_parent(Some(lazy3.clone()));
        node3.write().unwrap().set_child(Some(lazy2.clone()));
        node1.write().unwrap().add_ready_neighbor(
            LazyItem {
                data: Some(node3),
                offset: None,
                decay_counter: 0,
            },
            0.9,
        );

        let mut writer = Cursor::new(Vec::new());
        let offset = lazy1.serialize(&mut writer).unwrap();
        // Print the contents of the cursor
        let serialized_data = writer.clone().into_inner();
        println!("Serialized data: {:?}", serialized_data);
        let reader = Cursor::new(writer.into_inner());
        println!("Reader created");

        let cache = get_cache(reader);
        println!("Cache created");

        let deserialized: LazyItemRef<MergedNode> = cache.load_item(offset).unwrap();
        println!("Deserialized: {:?}", deserialized);

        let deserialized_guard = deserialized.item.read().unwrap();
        println!("Deserialized guard: {:?}", deserialized_guard);

        let deserialized_data = deserialized_guard.data.clone().unwrap();
        println!("Deserialized data: {:?}", deserialized_data);

        let deserialized_data_guard = deserialized_data.read().unwrap();
        println!("Deserialized data guard: {:?}", deserialized_data_guard);

        println!(
            "Number of neighbors: {}",
            deserialized_data_guard.get_neighbors().len()
        );

        let parent = deserialized_data_guard.get_parent().unwrap();
        println!("Parent: {:?}", parent);

        let parent_guard = parent.item.read().unwrap();
        println!("Parent guard: {:?}", parent_guard);

        // Deserialize the parent
        if let LazyItem {
            data: Some(parent), ..
        } = &*parent_guard
        {
            println!("Parent data found");
            let parent_guard = parent.read().unwrap();
            println!("Parent data guard: {:?}", parent_guard);

            let child = parent_guard.get_child().unwrap();
            println!("Child: {:?}", child);

            let child_guard = child.item.read().unwrap();
            println!("Child guard: {:?}", child_guard);

            let grand_parent = parent_guard.get_parent().unwrap();
            println!("Grand parent: {:?}", grand_parent);

            let grand_parent_guard = grand_parent.item.read().unwrap();
            println!("Grand parent guard: {:?}", grand_parent_guard);

            println!(
                "Child guard matches: {}",
                matches!(
                    &*child_guard,
                    LazyItem {
                        data: Some(_),
                        offset: Some(_),
                        ..
                    }
                )
            );

            println!(
                "Grand parent guard matches: {}",
                matches!(
                    &*grand_parent_guard,
                    LazyItem {
                        data: Some(_),
                        offset: Some(_),
                        ..
                    }
                )
            );
        } else {
            println!("Deserialization Error: Parent data not found");
            panic!("Deserialization Error");
        }
    }

    #[test]
    fn test_lazy_items_linked_chunk_serialization() {
        let lazy_items = LazyItems::new();
        for i in 1..13 {
            lazy_items.push(LazyItem::with_data(simple_merged_node(i, 2)));
        }

        let mut writer = Cursor::new(Vec::new());
        let offset = lazy_items.serialize(&mut writer).unwrap();

        let reader = Cursor::new(writer.into_inner());
        let cache = get_cache(reader);
        let deserialized: LazyItems<MergedNode> = cache.load_item(offset).unwrap();

        assert_eq!(lazy_items.len(), deserialized.len());
        for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
            match (original, deserialized) {
                (
                    LazyItem {
                        data: Some(original),
                        ..
                    },
                    LazyItem {
                        data: Some(deserialized),
                        ..
                    },
                ) => {
                    let original_guard = original.read().unwrap();
                    let deserialized_guard = deserialized.read().unwrap();
                    assert_eq!(original_guard.version_id, deserialized_guard.version_id);
                    assert_eq!(original_guard.hnsw_level, deserialized_guard.hnsw_level);
                }
                _ => panic!("Deserialization mismatch"),
            }
        }
    }
}
