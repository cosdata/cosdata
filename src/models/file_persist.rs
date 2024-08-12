use super::cache_loader::NodeRegistry;
use super::common::WaCustomError;
use super::lazy_load::LazyItem;
use super::types::{HNSWLevel, Item, MergedNode, NodeProp, VectorId};
use crate::models::custom_buffered_writer::*;
use crate::models::serializer::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

pub fn read_node_from_file<R: Read + Seek>(
    offset: FileOffset,
    cache: Arc<NodeRegistry<R>>,
) -> std::io::Result<MergedNode> {
    // Seek to the specified offset
    // file.seek(SeekFrom::Start(offset as u64))?;

    // Deserialize the NodePersist from the current position
    let node: MergedNode = cache.load_item(offset)?;

    // Pretty print the node
    println!("Read NodePersist from offset {}:", offset);
    // println!("{}", node);

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
    mut node: Item<LazyItem<MergedNode>>,
) -> Result<(), WaCustomError> {
    let LazyItem::Valid {
        data: Some(data), ..
    } = node.get()
    else {
        return Err(WaCustomError::LazyLoadingError("data is None".to_string()));
    };
    let file_loc = write_node_update(ver_file, data.clone(), node.get_offset())?;
    node.rcu(|node| {
        let node = node.clone();
        node.set_offset(Some(file_loc as u32));
        node
    });
    Ok(())
}

pub fn write_node_to_file(mut node: Item<MergedNode>, writer: &mut CustomBufferedWriter) -> u32 {
    println!("about to write node: {}", node.get());
    // Assume CustomBufferWriter already handles seeking to the end
    // Serialize
    let result = node.get().serialize(writer);
    let offset = result.expect("Failed to serialize NodePersist & write to file");
    FileOffset(offset)
}

pub fn write_node_to_file_at_offset(
    mut node: Item<MergedNode>,
    writer: &mut CustomBufferedWriter,
    offset: u32,
) -> u32 {
    println!("write_node_to_file_at_offset");
    // Seek to the specified offset before writing
    writer
        .seek(SeekFrom::Start(offset.0 as u64))
        .expect("Failed to seek in file");
    // Serialize
    let result = node.get().serialize(writer);
    let offset = result.expect("Failed to serialize NodePersist & write to file");
    FileOffset(offset)
}
//
pub fn load_vector_id_lsmdb(_level: HNSWLevel, _vector_id: VectorId) -> LazyItem<MergedNode> {
    LazyItem::Invalid
}

pub fn load_neighbor_persist_ref(_level: HNSWLevel, _node_file_ref: u32) -> Option<MergedNode> {
    None
}
pub fn write_prop_to_file(prop: &NodeProp, mut file: &File) -> (FileOffset, BytesToRead) {
    let mut prop_bytes = Vec::new();
    //let result = encode(&prop);
    let result = serde_cbor::to_vec(&prop).unwrap();

    prop_bytes.extend_from_slice(result.as_ref());

    file.write_all(&prop_bytes)
        .expect("Failed to write to file");
    let offset = file.metadata().unwrap().len() - prop_bytes.len() as u64;
    (
        FileOffset(offset as u32),
        BytesToRead(prop_bytes.len() as u32),
    )
}
