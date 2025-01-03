use super::CustomSerialize;
use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::NodeRegistry,
    cuckoo_filter_tree::CuckooFilterTreeNode,
    lazy_load::FileIndex,
    types::FileOffset,
    versioning::Hash,
};
use cuckoofilter::{CuckooFilter, ExportedCuckooFilter};
use std::collections::HashSet;
use std::collections::VecDeque;
use std::io::SeekFrom;
use std::sync::Arc;

fn push_nodes_to_queue<'a>(
    node: Option<&'a CuckooFilterTreeNode>,
    queue: &mut VecDeque<Option<&'a CuckooFilterTreeNode>>,
) {
    match node {
        None => {
            queue.push_back(None);
        }
        Some(node) => {
            queue.push_back(Some(node));
            push_nodes_to_queue(node.left.as_deref(), queue);
            push_nodes_to_queue(node.right.as_deref(), queue);
        }
    }
}

fn reconstruct_tree(
    queue: &mut VecDeque<Option<CuckooFilterTreeNode>>,
) -> Option<Box<CuckooFilterTreeNode>> {
    if queue.is_empty() {
        return None;
    }

    let mut node = queue.pop_front().unwrap();
    match node {
        None => None,
        Some(mut node) => {
            node.left = reconstruct_tree(queue);
            node.right = reconstruct_tree(queue);
            Some(Box::new(node))
        }
    }
}

impl CustomSerialize for CuckooFilterTreeNode {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        // Push nodes in preorder traversal to a queue
        let mut queue: VecDeque<Option<&CuckooFilterTreeNode>> = VecDeque::new();
        push_nodes_to_queue(Some(self), &mut queue);
        let number_of_nodes = queue.len();

        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)? as u32;

        // Serialize the number of nodes in the queue
        // Null nodes are also included in the count
        bufman.write_u32_with_cursor(cursor, number_of_nodes as u32)?;

        // Serialize each node in the queue
        for node in queue {
            match node {
                None => {
                    // Serialize a u32::MAX to indicate a null node
                    bufman.write_u32_with_cursor(cursor, u32::MAX)?;
                }
                Some(node) => {
                    // Serialize the node
                    // Write node metadata
                    bufman.write_u32_with_cursor(cursor, node.index as u32)?;
                    bufman.write_f32_with_cursor(cursor, node.range_min)?;
                    bufman.write_f32_with_cursor(cursor, node.range_max)?;

                    // Serialize the Cuckoo filter by first converting it to a struct ExportedCuckooFilter
                    // We first serialize the length of the byte array, then the byte array itself
                    let exported_cuckoo_filter = node.filter.export();
                    bufman.write_u32_with_cursor(cursor, exported_cuckoo_filter.length as u32)?;
                    for value in exported_cuckoo_filter.values {
                        bufman.write_u8_with_cursor(cursor, value)?;
                    }
                }
            }
        }

        Ok(start_offset)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Ok(CuckooFilterTreeNode::new(0, 0.0, 0.0)),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                ..
            } => {
                let bufman = bufmans.get(version_id)?;

                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

                // Read the number of nodes and create a queue of that size
                let number_of_nodes = bufman.read_u32_with_cursor(cursor)?;
                let mut queue: VecDeque<Option<CuckooFilterTreeNode>> =
                    VecDeque::with_capacity(number_of_nodes as usize);

                for _ in 0..number_of_nodes {
                    // Read node metadata
                    // index = u32::MAX indicates a dummy node
                    let index = bufman.read_u32_with_cursor(cursor)?;
                    if index == u32::MAX {
                        queue.push_back(None);
                        continue;
                    }

                    let range_min = bufman.read_f32_with_cursor(cursor)?;
                    let range_max = bufman.read_f32_with_cursor(cursor)?;

                    // Deserialize the cuckoo filter's length and values
                    let exported_cuckoo_filter_length = bufman.read_u32_with_cursor(cursor)?;
                    let mut exported_cuckoo_filter_values =
                        vec![0u8; exported_cuckoo_filter_length as usize];
                    for i in 0..exported_cuckoo_filter_length {
                        exported_cuckoo_filter_values[i as usize] =
                            bufman.read_u8_with_cursor(cursor)?;
                    }
                    let exported_cuckoo_filter = ExportedCuckooFilter {
                        length: exported_cuckoo_filter_length as usize,
                        values: exported_cuckoo_filter_values,
                    };

                    // Convert the ExportedCuckooFilter to a CuckooFilter
                    let cuckoo_filter = CuckooFilter::from(exported_cuckoo_filter);

                    // Reconstruct the node
                    let mut node = CuckooFilterTreeNode::new(index as usize, range_min, range_max);
                    node.filter = cuckoo_filter;

                    // Push the node to the queue
                    queue.push_back(Some(node));
                }

                // Reconstruct the tree from the queue
                let root = reconstruct_tree(&mut queue).unwrap();
                Ok(*root)
            }
        }
    }
}
