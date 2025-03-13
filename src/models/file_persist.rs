use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::common::WaCustomError;
use super::prob_node::SharedNode;
use super::serializer::hnsw::HNSWIndexSerialize;
use super::types::{BytesToRead, FileOffset, NodeProp, VectorId};
use super::versioning::Hash;
use crate::storage::Storage;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::Arc;

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn write_node_to_file(
    lazy_item: SharedNode,
    bufmans: &BufferManagerFactory<Hash>,
    level_0_bufmans: &BufferManagerFactory<Hash>,
    version: Hash,
) -> Result<u32, WaCustomError> {
    let lazy_item_ref = unsafe { &*lazy_item };
    let is_level_0 = lazy_item_ref.is_level_0;
    let (bufman, bufmans) = if is_level_0 {
        (level_0_bufmans.get(version)?, level_0_bufmans)
    } else {
        (bufmans.get(version)?, bufmans)
    };
    let cursor = bufman.open_cursor()?;
    let offset = lazy_item.serialize(bufmans, version, cursor)?;
    bufman.close_cursor(cursor)?;

    Ok(offset)
}

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
    file: &mut File,
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
