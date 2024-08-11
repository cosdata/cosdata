use super::cache_loader::NodeRegistry;
use super::chunked_list::LazyItem;
use super::common::WaCustomError;
use super::types::{FileOffset, HNSWLevel, Item, MergedNode, NodeProp, VectorId};
use crate::models::custom_buffered_writer::*;
use crate::models::serializer::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

// pub type BytesToRead = u32;

// pub type PropPersistRef = (FileOffset, BytesToRead);

pub fn read_node_from_file<R: Read + Seek>(
    offset: FileOffset,
    cache: Arc<NodeRegistry<R>>,
) -> std::io::Result<MergedNode> {
    // Seek to the specified offset
    // file.seek(SeekFrom::Start(offset as u64))?;

    // Deserialize the NodePersist from the current position
    let node: MergedNode = cache.load_item(offset)?;

    // Pretty print the node
    println!("Read NodePersist from offset {}:", offset.0);
    println!("{}", node);

    Ok(node)
}

pub fn write_node_update(
    ver_file: &mut CustomBufferedWriter,
    nprst: Item<MergedNode>,
    current_location: Option<FileOffset>,
) -> Result<FileOffset, WaCustomError> {
    if let Some(loc) = current_location {
        Ok(write_node_to_file_at_offset(nprst, ver_file, loc))
    } else {
        Ok(write_node_to_file(nprst, ver_file))
    }
}

pub fn persist_node_update_loc(
    ver_file: &mut CustomBufferedWriter,
    node: &mut LazyItem<MergedNode>,
) -> Result<(), WaCustomError> {
    let Some(data) = &node.data else {
        return Err(WaCustomError::LazyLoadingError("data is None".to_string()));
    };
    let file_loc = write_node_update(ver_file, data.clone(), node.offset)?;
    node.offset = Some(file_loc);
    Ok(())
}

pub fn write_node_to_file(node: Item<MergedNode>, writer: &mut CustomBufferedWriter) -> FileOffset {
    // Assume CustomBufferWriter already handles seeking to the end
    // Serialize
    let result = node.read().unwrap().serialize(writer);
    let offset = result.expect("Failed to serialize NodePersist & write to file");
    FileOffset(offset)
}

pub fn write_node_to_file_at_offset(
    node: Item<MergedNode>,
    writer: &mut CustomBufferedWriter,
    offset: FileOffset,
) -> FileOffset {
    // Seek to the specified offset before writing
    writer
        .seek(SeekFrom::Start(offset.0 as u64))
        .expect("Failed to seek in file");
    // Serialize
    let result = node.read().unwrap().serialize(writer);
    let offset = result.expect("Failed to serialize NodePersist & write to file");
    FileOffset(offset)
}
//
pub fn load_vector_id_lsmdb(level: HNSWLevel, vector_id: VectorId) -> Option<LazyItem<MergedNode>> {
    None
}

pub fn load_neighbor_persist_ref(
    level: HNSWLevel,
    node_file_ref: FileOffset,
) -> Option<MergedNode> {
    None
}
pub fn write_prop_to_file(prop: &NodeProp, mut file: &File) -> (FileOffset, u32) {
    let mut prop_bytes = Vec::new();
    //let result = encode(&prop);
    let result = serde_cbor::to_vec(&prop).unwrap();

    prop_bytes.extend_from_slice(result.as_ref());

    file.write_all(&prop_bytes)
        .expect("Failed to write to file");
    let offset = file.metadata().unwrap().len() - prop_bytes.len() as u64;
    (FileOffset(offset as u32), prop_bytes.len() as u32)
}
