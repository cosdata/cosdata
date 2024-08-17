use super::cache_loader::NodeRegistry;
use super::common::WaCustomError;
use super::lazy_load::{FileIndex, LazyItem, SyncPersist};
use super::types::{HNSWLevel, MergedNode, NodeProp, VectorId};
use crate::models::custom_buffered_writer::*;
use crate::models::serializer::CustomSerialize;
use arcshift::ArcShift;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

pub fn read_node_from_file<R: Read + Seek>(
    file_index: FileIndex,
    cache: Arc<NodeRegistry<R>>,
) -> std::io::Result<MergedNode> {
    // Deserialize the MergedNode using the FileIndex
    let node: MergedNode = cache.load_item(file_index.clone())?;

    // Pretty print the node
    match file_index {
        FileIndex::Valid { offset, version } => {
            println!(
                "Read MergedNode from offset: {}, version: {}",
                offset, version
            );
        }
        FileIndex::Invalid => {
            println!("Attempted to read MergedNode with an invalid FileIndex");
        }
    }

    // You might want to add more detailed printing here, depending on what information
    // you want to see about the node
    // println!("{:#?}", node);

    Ok(node)
}
pub fn write_node_to_file(
    lazy_item: &LazyItem<MergedNode>,
    writer: &mut CustomBufferedWriter,
    file_index: Option<FileIndex>,
) -> Result<FileIndex, WaCustomError> {
    let mut node_arc = lazy_item
        .get_data()
        .ok_or(WaCustomError::LazyLoadingError("node in null".to_string()))?;
    let node = node_arc.get();

    match file_index {
        Some(FileIndex::Valid { offset, version }) => {
            println!(
                "About to write at offset {}, version {}, node: {:#?}",
                offset, version, node
            );
            writer
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| WaCustomError::FsError(e.to_string()))?;
        }
        Some(FileIndex::Invalid) | None => {
            println!("About to write node at the end of file: {:#?}", node);
            writer
                .seek(SeekFrom::End(0))
                .map_err(|e| WaCustomError::FsError(e.to_string()))?;
        }
    }

    let new_offset = node
        .serialize(writer)
        .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

    // Create and return the new FileIndex
    let new_file_index = FileIndex::Valid {
        offset: new_offset,
        version: match file_index {
            Some(FileIndex::Valid { version, .. }) => version,
            _ => lazy_item.get_current_version(),
        },
    };

    Ok(new_file_index)
}

pub fn persist_node_update_loc(
    ver_file: &mut CustomBufferedWriter,
    node: &mut ArcShift<LazyItem<MergedNode>>,
) -> Result<(), WaCustomError> {
    let lazy_item = node.get();
    let current_file_index = lazy_item.get_file_index();

    // Write the node to file
    let new_file_index = write_node_to_file(node.get(), ver_file, current_file_index)?;

    // Update the file index in the lazy item
    node.rcu(|lazy_item| {
        let updated_item = lazy_item.clone();
        updated_item.set_file_index(Some(new_file_index));
        updated_item
    });

    Ok(())
}
//
pub fn load_vector_id_lsmdb(_level: HNSWLevel, _vector_id: VectorId) -> LazyItem<MergedNode> {
    LazyItem::Invalid
}

pub fn load_neighbor_persist_ref(_level: HNSWLevel, _node_file_ref: u32) -> Option<MergedNode> {
    None
}
pub fn write_prop_to_file(prop: &NodeProp, mut file: &File) -> (u32, u32) {
    let mut prop_bytes = Vec::new();
    //let result = encode(&prop);
    let result = serde_cbor::to_vec(&prop).unwrap();

    prop_bytes.extend_from_slice(result.as_ref());

    file.write_all(&prop_bytes)
        .expect("Failed to write to file");
    let offset = file.metadata().unwrap().len() - prop_bytes.len() as u64;
    (offset as u32, prop_bytes.len() as u32)
}
