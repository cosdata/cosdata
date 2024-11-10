use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::cache_loader::NodeRegistry;
use super::common::WaCustomError;
use super::lazy_load::{FileIndex, LazyItem, SyncPersist};
use super::types::{BytesToRead, FileOffset, HNSWLevel, MergedNode, NodeProp, VectorId};
use crate::models::serializer::CustomSerialize;
use crate::storage::Storage;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

pub fn read_node_from_file(
    file_index: FileIndex,
    cache: Arc<NodeRegistry>,
) -> Result<MergedNode, BufIoError> {
    // Deserialize the MergedNode using the FileIndex
    let node: MergedNode = cache.load_item(file_index.clone())?;

    // Pretty print the node
    match file_index {
        FileIndex::Valid {
            offset, version_id, ..
        } => {
            println!(
                "Read MergedNode from offset: {}, version: {}",
                offset.0, *version_id
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
    bufmans: Arc<BufferManagerFactory>,
) -> Result<(), WaCustomError> {
    let file_index = lazy_item.get_file_index();
    let version = lazy_item.get_current_version();
    let bufman = bufmans.get(&version)?;
    let cursor = bufman.open_cursor()?;

    match file_index {
        Some(FileIndex::Valid {
            offset: FileOffset(offset),
            version_id,
            ..
        }) => {
            println!(
                "About to write at offset {}, version {}",
                offset, *version_id
            );
            bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
        }
        Some(FileIndex::Invalid) | None => {
            println!("About to write node at the end of file");
            bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
        }
    }

    lazy_item.serialize(bufmans, version, cursor)?;

    bufman.close_cursor(cursor)?;

    Ok(())
}
//
pub fn load_vector_id_lsmdb(_level: HNSWLevel, _vector_id: VectorId) -> LazyItem<MergedNode> {
    LazyItem::Invalid
}

pub fn load_neighbor_persist_ref(_level: HNSWLevel, _node_file_ref: u32) -> Option<MergedNode> {
    None
}

#[derive(Debug, Clone, Serialize)]
pub struct NodePropSerialize<'a> {
    pub id: &'a VectorId,
    pub value: Arc<Storage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodePropDeserialize {
    pub id: VectorId,
    pub value: Arc<Storage>,
}

pub fn write_prop_to_file(
    id: &VectorId,
    value: Arc<Storage>,
    mut file: &File,
) -> Result<(FileOffset, BytesToRead), WaCustomError> {
    let prop = NodePropSerialize { id, value };
    let prop_bytes =
        serde_cbor::to_vec(&prop).map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

    let offset = file
        .seek(SeekFrom::End(0))
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;

    file.write_all(&prop_bytes)
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;

    Ok((
        FileOffset(offset as u32),
        BytesToRead(prop_bytes.len() as u32),
    ))
}

pub fn read_prop_from_file(
    (offset, bytes_to_read): (FileOffset, BytesToRead),
    mut file: &File,
) -> Result<NodeProp, WaCustomError> {
    let mut bytes = vec![0u8; bytes_to_read.0 as usize];
    file.seek(SeekFrom::Start(offset.0 as u64))
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;
    file.read_exact(&mut bytes)
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let prop: NodePropDeserialize = serde_cbor::from_slice(&bytes)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    Ok(NodeProp {
        id: prop.id,
        value: prop.value,
        location: (offset, bytes_to_read),
    })
}
