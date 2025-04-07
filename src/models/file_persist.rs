use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::common::WaCustomError;
use super::prob_node::SharedNode;
use super::serializer::hnsw::HNSWIndexSerialize;
use super::types::{
    BytesToRead, FileOffset, Metadata, MetadataId, NodePropMetadata, NodePropValue, VectorId,
};
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
struct NodePropValueSerialize<'a> {
    pub id: &'a VectorId,
    pub value: Arc<Storage>,
}

#[derive(Debug, Clone, Deserialize)]
struct NodePropValueDeserialize {
    pub id: VectorId,
    pub value: Arc<Storage>,
}

pub fn write_prop_value_to_file(
    id: &VectorId,
    value: Arc<Storage>,
    file: &mut File,
) -> Result<(FileOffset, BytesToRead), WaCustomError> {
    let prop = NodePropValueSerialize { id, value };
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

pub fn read_prop_value_from_file(
    (offset, bytes_to_read): (FileOffset, BytesToRead),
    file: &mut File,
) -> Result<NodePropValue, BufIoError> {
    let mut bytes = vec![0u8; bytes_to_read.0 as usize];
    file.seek(SeekFrom::Start(offset.0 as u64))?;
    file.read_exact(&mut bytes)?;

    let prop: NodePropValueDeserialize = serde_cbor::from_slice(&bytes)
        .map_err(|e| BufIoError::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))?;

    Ok(NodePropValue {
        id: prop.id,
        vec: prop.value,
        location: (offset, bytes_to_read),
    })
}

#[derive(Deserialize, Serialize)]
struct NodePropMetadataSerde {
    pub id: MetadataId,
    pub vec: Arc<Metadata>,
}

pub fn write_prop_metadata_to_file(
    id: &MetadataId,
    vec: Arc<Metadata>,
    file: &mut File,
) -> Result<(FileOffset, BytesToRead), WaCustomError> {
    let prop_metadata = NodePropMetadataSerde {
        id: id.clone(),
        vec: vec.clone(),
    };
    let prop_bytes = serde_cbor::to_vec(&prop_metadata)
        .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

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

pub fn read_prop_metadata_from_file(
    (offset, bytes_to_read): (FileOffset, BytesToRead),
    file: &mut File,
) -> Result<NodePropMetadata, BufIoError> {
    let mut bytes = vec![0u8; bytes_to_read.0 as usize];
    file.seek(SeekFrom::Start(offset.0 as u64))?;
    file.read_exact(&mut bytes)?;

    let prop_metadata: NodePropMetadataSerde = serde_cbor::from_slice(&bytes)
        .map_err(|e| BufIoError::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))?;

    Ok(NodePropMetadata {
        id: prop_metadata.id,
        vec: prop_metadata.vec,
        location: (offset, bytes_to_read),
    })
}
