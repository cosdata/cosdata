use super::buffered_io::BufIoError;
use super::cache_loader::HNSWIndexCache;
use super::common::WaCustomError;
use super::prob_node::{SharedLatestNode, SharedNode};
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

pub fn write_lazy_item_latest_ptr_to_file(
    cache: &HNSWIndexCache,
    lazy_item_ptr: SharedLatestNode,
    file_id: IndexFileId,
) -> Result<u32, BufIoError> {
    let bufman = cache.bufmans.get(file_id)?;
    let cursor = bufman.open_cursor()?;
    let latest_version_links_cursor = cache.latest_version_links_bufman.open_cursor()?;
    let offset = lazy_item_ptr.serialize(
        &bufman,
        &cache.latest_version_links_bufman,
        cursor,
        latest_version_links_cursor,
    )?;
    cache
        .latest_version_links_bufman
        .close_cursor(latest_version_links_cursor)?;
    bufman.close_cursor(cursor)?;
    Ok(offset)
}

pub fn write_lazy_item_to_file(
    cache: &HNSWIndexCache,
    lazy_item: SharedNode,
    file_id: IndexFileId,
) -> Result<u32, BufIoError> {
    let bufman = cache.bufmans.get(file_id)?;
    let cursor = bufman.open_cursor()?;
    let latest_version_links_cursor = cache.latest_version_links_bufman.open_cursor()?;
    let offset = lazy_item.serialize(
        &bufman,
        &cache.latest_version_links_bufman,
        cursor,
        latest_version_links_cursor,
    )?;
    cache
        .latest_version_links_bufman
        .close_cursor(latest_version_links_cursor)?;
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
    pub replica_id: InternalId,
    pub vec: Arc<Metadata>,
}

pub fn write_prop_metadata_to_file(
    replica_id: InternalId,
    vec: Arc<Metadata>,
    file: &mut File,
) -> Result<(FileOffset, BytesToRead), WaCustomError> {
    let prop_metadata = NodePropMetadataSerde {
        replica_id,
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
        replica_id: prop_metadata.replica_id,
        vec: prop_metadata.vec,
        location: (offset, bytes_to_read),
    })
}
