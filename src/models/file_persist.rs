use super::chunked_list::LazyItem;
use super::common::{tuple_to_string, WaCustomError};
use super::types::{
    HNSWLevel, MergedNode, Neighbour, NodeFileRef, NodeProp, VectorId, VectorQt, VectorStore,
    VersionId,
};
use crate::models::custom_buffered_writer::*;
use crate::models::serializer::*;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::sync::{Arc, RwLock};

// pub type FileOffset = u32;
// pub type BytesToRead = u32;

// pub type PropPersistRef = (FileOffset, BytesToRead);

pub fn read_node_from_file(file: &mut File, offset: u32) -> std::io::Result<MergedNode> {
    // Seek to the specified offset
    file.seek(SeekFrom::Start(offset as u64))?;

    // Deserialize the NodePersist from the current position
    let node = MergedNode::deserialize(file, offset)?;

    // Pretty print the node
    println!("Read NodePersist from offset {}:", offset);
    println!("{}", node);

    Ok(node)
}

pub fn write_node_update(
    ver_file: &mut CustomBufferedWriter,
    nprst: &mut MergedNode,
    current_location: Option<u32>,
) -> Result<u64, WaCustomError> {
    if let Some(loc) = current_location {
        Ok(write_node_to_file_at_offset(nprst, ver_file, loc) as u64)
    } else {
        Ok(write_node_to_file(nprst, ver_file) as u64)
    }
}

pub fn persist_node_update_loc(
    ver_file: &mut CustomBufferedWriter,
    node: &mut LazyItem<MergedNode>,
) -> Result<(), WaCustomError> {
    match node {
        LazyItem::Ready(arc_node, loc) => {
            let mut_node = Arc::make_mut(arc_node);
            // Get a mutable reference to the MergedNode
            let current_location = loc;

            let file_loc = write_node_update(ver_file, mut_node, loc)?;
            Ok(())
        }
        LazyItem::LazyLoad(offset) => {
            // If it's a lazy load, we might want to load the node first
            Err(WaCustomError::LazyLoadingError(
                "Cannot update location of LazyLoad node".to_string(),
            ))
        }
        LazyItem::Null => Err(WaCustomError::LazyLoadingError(
            "Cannot update location of Null node".to_string(),
        )),
    }
}

pub fn write_node_to_file(node: &mut MergedNode, writer: &mut CustomBufferedWriter) -> u32 {
    // Assume CustomBufferWriter already handles seeking to the end
    // Serialize
    let result = node.serialize(writer);
    let offset = result.expect("Failed to serialize NodePersist & write to file");
    offset as u32
}

pub fn write_node_to_file_at_offset(
    node: &mut MergedNode,
    writer: &mut CustomBufferedWriter,
    offset: u32,
) -> u32 {
    // Seek to the specified offset before writing
    writer
        .seek(SeekFrom::Start(offset as u64))
        .expect("Failed to seek in file");
    // Serialize
    let result = node.serialize(writer);
    let offset = result.expect("Failed to serialize NodePersist & write to file");
    offset as u32
}
//
pub fn load_vector_id_lsmdb(level: HNSWLevel, vector_id: VectorId) -> Option<MergedNode> {
    return None;
}

pub fn load_neighbor_persist_ref(level: HNSWLevel, node_file_ref: u32) -> Option<MergedNode> {
    return None;
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
