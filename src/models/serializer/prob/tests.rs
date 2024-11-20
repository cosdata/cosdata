use std::sync::Arc;

use arcshift::ArcShift;
use tempfile::{tempdir, TempDir};

use crate::models::{
    buffered_io::{BufferManager, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::FileIndex,
    prob_lazy_load::lazy_item::{ProbLazyItem, ProbLazyItemState},
    prob_node::ProbNode,
    types::{BytesToRead, FileOffset, HNSWLevel, PropState},
    versioning::Hash,
};

use super::ProbSerialize;

fn get_cache(bufmans: Arc<BufferManagerFactory>) -> Arc<ProbCache> {
    Arc::new(ProbCache::new(1000, bufmans))
}

fn setup_test(
    root_version: &Hash,
) -> (
    Arc<BufferManagerFactory>,
    Arc<ProbCache>,
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
    let node = ProbNode::new(
        HNSWLevel(2),
        ArcShift::new(PropState::Pending((FileOffset(0), BytesToRead(0)))),
        None,
        None,
        8
    );
    let root_version_number = 0;
    let root_version_id = Hash::from(0);
    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let lazy_item = ProbLazyItem::new(node, root_version_id, root_version_number);

    let offset = lazy_item
        .serialize(bufmans.clone(), root_version_id, cursor)
        .unwrap();
    bufman.close_cursor(cursor).unwrap();

    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: root_version_number,
        version_id: root_version_id,
    };

    let deserialized: Arc<ProbLazyItem<ProbNode>> = cache.load_item(file_index).unwrap();

    match (&*lazy_item.get_state(), &*deserialized.get_state()) {
        (
            ProbLazyItemState::Ready {
                data: original_data,
                version_id: original_version_id,
                version_number: original_version_number,
                ..
            },
            ProbLazyItemState::Ready {
                data: deserialized_data,
                version_id: deserialized_version_id,
                version_number: deserialized_version_number,
                ..
            },
        ) => {
            assert_eq!(original_data.hnsw_level, deserialized_data.hnsw_level);
            assert_eq!(original_version_number, deserialized_version_number);
            assert_eq!(original_version_id, deserialized_version_id);
        }
        _ => panic!("Deserialization mismatch"),
    }
}

#[test]
fn test_prob_node_acyclic_serialization() {
    let root_version_id = Hash::from(0);
    let node = ProbNode::new(
        HNSWLevel(2),
        ArcShift::new(PropState::Pending((FileOffset(0), BytesToRead(0)))),
        None,
        None,
        8
    );

    let (bufmans, cache, bufman, cursor, _temp_dir) = setup_test(&root_version_id);

    let offset = node.serialize(bufmans, root_version_id, cursor).unwrap();
    let file_index = FileIndex::Valid {
        offset: FileOffset(offset),
        version_number: 0,
        version_id: root_version_id,
    };
    bufman.close_cursor(cursor).unwrap();

    let deserialized: ProbNode = cache.load_item(file_index).unwrap();

    assert_eq!(node.hnsw_level, deserialized.hnsw_level);
    assert!(deserialized.get_parent().is_none());
    assert!(deserialized.get_child().is_none());
    assert_eq!(deserialized.get_neighbors().len(), 0);
}
