use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::common::WaCustomError;
use super::lazy_load::SyncPersist;
use super::prob_node::SharedNode;
use super::serializer::prob::ProbSerialize;
use super::types::{BytesToRead, FileOffset, NodeProp, VectorId};
use super::versioning::Hash;
use crate::storage::Storage;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::Arc;

// pub fn read_node_from_file(
//     file_index: FileIndex,
//     cache: Arc<NodeRegistry>,
// ) -> Result<MergedNode, BufIoError> {
//     // Deserialize the MergedNode using the FileIndex
//     let node: MergedNode = cache.load_item(file_index.clone())?;

//     // Pretty print the node
//     match file_index {
//         FileIndex::Valid {
//             offset, version_id, ..
//         } => {
//             println!(
//                 "Read MergedNode from offset: {}, version: {}",
//                 offset.0, *version_id
//             );
//         }
//         FileIndex::Invalid => {
//             println!("Attempted to read MergedNode with an invalid FileIndex");
//         }
//     }

//     // You might want to add more detailed printing here, depending on what information
//     // you want to see about the node
//     // println!("{:#?}", node);

//     Ok(node)
// }
pub fn write_node_to_file(
    lazy_item: SharedNode,
    bufmans: &BufferManagerFactory<Hash>,
) -> Result<u32, WaCustomError> {
    let lazy_item_ref = unsafe { &*lazy_item };
    let version = lazy_item_ref.get_current_version();
    let bufman = bufmans.get(version)?;
    let cursor = bufman.open_cursor()?;

    lazy_item_ref.set_persistence(true);
    let offset = lazy_item.serialize(bufmans, version, cursor)?;

    bufman.close_cursor(cursor)?;

    Ok(offset)
}
// #[allow(dead_code)]
// pub fn load_vector_id_lsmdb(
//     _level: HNSWLevel,
//     _vector_id: VectorId,
// ) -> Option<ProbLazyItem<MergedNode>> {
//     None
// }

// #[allow(dead_code)]
// pub fn load_neighbor_persist_ref(_level: HNSWLevel, _node_file_ref: u32) -> Option<MergedNode> {
//     None
// }

#[derive(Debug, Clone, Serialize)]
pub struct NodePropSerialize<'a> {
    pub id: &'a VectorId,
    pub value: Arc<Storage>,
}

#[allow(dead_code)]
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
    file: &mut File,
) -> Result<NodeProp, BufIoError> {
    let mut bytes = vec![0u8; bytes_to_read.0 as usize];
    file.seek(SeekFrom::Start(offset.0 as u64))?;
    file.read_exact(&mut bytes)?;

    let prop: NodePropDeserialize = serde_cbor::from_slice(&bytes)
        .map_err(|e| BufIoError::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))?;

    Ok(NodeProp {
        id: prop.id,
        value: prop.value,
        location: (offset, bytes_to_read),
    })
}
