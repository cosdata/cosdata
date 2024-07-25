#[cfg(test)]
mod tests {
    use crate::models::chunked_list::*;
    use crate::models::serializer::*;
    use crate::models::types::*;
    use std::io::Cursor;
    use std::sync::Arc;

    // Helper function to create a sample VectorQt
    fn sample_vector_qt() -> VectorQt {
        VectorQt::UnsignedByte {
            mag: 100,
            quant_vec: vec![1, 2, 3, 4, 5],
        }
    }
    // Helper function to create a simple MergedNode
    fn simple_merged_node(version_id: VersionId, hnsw_level: HNSWLevel) -> MergedNode {
        MergedNode::new(version_id, hnsw_level)
    }

    #[test]
    fn test_lazy_item_serialization() {
        let node = Arc::new(simple_merged_node(1, 2));
        let lazy_item = LazyItem::Ready(node, Some(0));

        let mut writer = Cursor::new(Vec::new());
        let offset = lazy_item.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let deserialized = LazyItem::<MergedNode>::deserialize(&mut reader, offset).unwrap();

        match (lazy_item, deserialized) {
            (LazyItem::Ready(original, _), LazyItem::Ready(deserialized, _)) => {
                assert_eq!(original.version_id, deserialized.version_id);
                assert_eq!(original.hnsw_level, deserialized.hnsw_level);
            }
            _ => panic!("Deserialization mismatch"),
        }
    }

    #[test]
    fn test_lazy_items_serialization() {
        let mut lazy_items = LazyItems::new();
        lazy_items.push(LazyItem::Ready(Arc::new(simple_merged_node(1, 2)), Some(0)));
        lazy_items.push(LazyItem::LazyLoad(10));
        lazy_items.push(LazyItem::Null);

        let mut writer = Cursor::new(Vec::new());
        let offset = lazy_items.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let deserialized = LazyItems::<MergedNode>::deserialize(&mut reader, offset).unwrap();

        assert_eq!(lazy_items.len(), deserialized.len());
        for (original, deserialized) in lazy_items.iter().zip(deserialized.iter()) {
            match (original, deserialized) {
                (LazyItem::Ready(orig, _), LazyItem::Ready(des, _)) => {
                    assert_eq!(orig.version_id, des.version_id);
                    assert_eq!(orig.hnsw_level, des.hnsw_level);
                }
                (LazyItem::LazyLoad(_), LazyItem::LazyLoad(_)) => {}
                (LazyItem::Null, LazyItem::Null) => {}
                _ => panic!("Deserialization mismatch"),
            }
        }
    }

    #[test]
    fn test_merged_node_acyclic_serialization() {
        let node = simple_merged_node(1, 2);

        let mut writer = Cursor::new(Vec::new());
        let offset = node.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let deserialized = MergedNode::deserialize(&mut reader, offset).unwrap();

        assert_eq!(node.version_id, deserialized.version_id);
        assert_eq!(node.hnsw_level, deserialized.hnsw_level);
        assert!(matches!(deserialized.get_parent(), LazyItem::Null));
        assert!(matches!(deserialized.get_child(), LazyItem::Null));
        assert_eq!(deserialized.get_neighbors().len(), 0);
        assert_eq!(deserialized.get_versions().len(), 0);
    }

    #[test]
    fn test_merged_node_with_neighbors_serialization() {
        let node = MergedNode::new(1, 2);

        let neighbor1 = Arc::new(MergedNode::new(2, 1));
        let neighbor2 = Arc::new(MergedNode::new(3, 1));
        node.add_ready_neighbor(neighbor1, 0.9);
        node.add_ready_neighbor(neighbor2, 0.8);

        let mut writer = Cursor::new(Vec::new());
        let offset = node.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let deserialized = MergedNode::deserialize(&mut reader, offset).unwrap();

        assert_eq!(
            node.get_neighbors().len(),
            deserialized.get_neighbors().len()
        );

        // Additional checks
        let original_neighbors = node.get_neighbors();
        let deserialized_neighbors = deserialized.get_neighbors();

        for (original, deserialized) in original_neighbors.iter().zip(deserialized_neighbors.iter())
        {
            match (original, deserialized) {
                (LazyItem::Ready(orig, _), LazyItem::Ready(des, _)) => {
                    assert_eq!(orig.node.version_id, des.node.version_id);
                    assert_eq!(orig.node.hnsw_level, des.node.hnsw_level);
                    assert_eq!(orig.cosine_similarity, des.cosine_similarity);
                }
                (LazyItem::LazyLoad(_), LazyItem::LazyLoad(_)) => {}
                _ => panic!("Neighbor deserialization mismatch"),
            }
        }
    }

    #[test]
    fn test_merged_node_with_parent_child_serialization() {
        let node = Arc::new(MergedNode::new(1, 2));
        let parent = Arc::new(MergedNode::new(2, 3));
        let child = Arc::new(MergedNode::new(3, 1));

        node.set_parent(parent.clone());
        node.set_child(child.clone());

        let mut writer = Cursor::new(Vec::new());
        let offset = node.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let deserialized = MergedNode::deserialize(&mut reader, offset).unwrap();

        assert!(matches!(deserialized.get_parent(), LazyItem::LazyLoad(_)));
        assert!(matches!(deserialized.get_child(), LazyItem::LazyLoad(_)));
    }

    #[test]
    fn test_merged_node_with_versions_serialization() {
        let node = Arc::new(MergedNode::new(1, 2));
        let version1 = Arc::new(MergedNode::new(2, 2));
        let version2 = Arc::new(MergedNode::new(3, 2));

        node.add_version(version1);
        node.add_version(version2);

        let mut writer = Cursor::new(Vec::new());
        let offset = node.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let deserialized = MergedNode::deserialize(&mut reader, offset).unwrap();

        assert_eq!(node.get_versions().len(), deserialized.get_versions().len());
    }

    #[test]
    fn test_merged_node_cyclic_serialization() {
        let node1 = Arc::new(MergedNode::new(1, 2));
        let node2 = Arc::new(MergedNode::new(2, 2));

        node1.set_parent(node2.clone());
        node2.set_child(node1.clone());

        let mut writer = Cursor::new(Vec::new());
        let offset = node1.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let deserialized = MergedNode::deserialize(&mut reader, offset).unwrap();

        assert!(matches!(deserialized.get_parent(), LazyItem::LazyLoad(_)));

        // Deserialize the parent
        if let LazyItem::LazyLoad(parent_offset) = deserialized.get_parent() {
            let parent = MergedNode::deserialize(&mut reader, parent_offset).unwrap();
            assert!(matches!(parent.get_child(), LazyItem::LazyLoad(_)));
        } else {
            panic!("Expected LazyLoad for parent");
        }
    }

    #[test]
    fn test_merged_node_complex_cyclic_serialization() {
        let node1 = Arc::new(MergedNode::new(1, 2));
        let node2 = Arc::new(MergedNode::new(2, 2));
        let node3 = Arc::new(MergedNode::new(3, 2));

        node1.set_parent(node2.clone());
        node2.set_child(node1.clone());
        node2.set_parent(node3.clone());
        node3.set_child(node2.clone());
        node1.add_ready_neighbor(node3.clone(), 0.9);

        let mut writer = Cursor::new(Vec::new());
        let offset = node1.serialize(&mut writer).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let deserialized = MergedNode::deserialize(&mut reader, offset).unwrap();

        assert!(matches!(deserialized.get_parent(), LazyItem::LazyLoad(_)));
        assert_eq!(deserialized.get_neighbors().len(), 1);

        // Deserialize the parent
        if let LazyItem::LazyLoad(parent_offset) = deserialized.get_parent() {
            let parent = MergedNode::deserialize(&mut reader, parent_offset).unwrap();
            assert!(matches!(parent.get_child(), LazyItem::LazyLoad(_)));
            assert!(matches!(parent.get_parent(), LazyItem::LazyLoad(_)));
        } else {
            panic!("Expected LazyLoad for parent");
        }
    }
}
