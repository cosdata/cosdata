use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::common::WaCustomError;
use super::prob_node::SharedNode;
use super::serializer::hnsw::HNSWIndexSerialize;
use super::types::{
    BytesToRead, FileOffset, InternalId, Metadata, NodePropMetadata, NodePropValue,
};
use crate::indexes::hnsw::offset_counter::IndexFileId;
use crate::storage::Storage;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::Arc;

pub fn write_node_to_file(
    lazy_item: SharedNode,
    bufmans: &BufferManagerFactory<IndexFileId>,
    file_id: IndexFileId,
) -> Result<u32, BufIoError> {
    let bufman = bufmans.get(file_id)?;
    let cursor = bufman.open_cursor()?;
    let offset = lazy_item.serialize(&bufman, cursor)?;
    bufman.close_cursor(cursor)?;
    Ok(offset)
}

#[derive(Debug, Clone, Serialize)]
struct NodePropValueSerialize<'a> {
    pub id: &'a InternalId,
    pub value: &'a Storage,
}

#[derive(Debug, Deserialize)]
struct NodePropValueDeserialize {
    pub id: InternalId,
    pub value: Storage,
}

pub fn write_prop_value_to_file(
    id: &InternalId,
    value: &Storage,
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
        vec: Arc::new(prop.value),
        location: (offset, bytes_to_read),
    })
}

#[derive(Deserialize, Serialize)]
struct NodePropMetadataSerde {
    pub vec: Arc<Metadata>,
}

pub fn write_prop_metadata_to_file(
    vec: Arc<Metadata>,
    file: &mut File,
) -> Result<(FileOffset, BytesToRead), WaCustomError> {
    let prop_metadata = NodePropMetadataSerde { vec: vec.clone() };
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
        vec: prop_metadata.vec,
        location: (offset, bytes_to_read),
    })
}
