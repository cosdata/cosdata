use super::common::{tuple_to_string, WaCustomError};
use super::types::{
    HNSWLevel, NeighbourRef, Node, NodeFileRef, NodeProp, NodeRef, VectorId, VectorQt, VectorStore,
    VersionId,
};
//use serde::{Deserialize, Serialize};
//use serde_cbor;
use crate::models::serializer::*;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{Seek, SeekFrom, Write};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
//start
pub type NodePersistRef = u32; // (offset )
pub type PropPersistRef = (u32, u32); // (offset , bytes length)

#[derive(Clone, Copy)]
pub struct NeighbourPersist {
    pub node: NodePersistRef,
    pub cosine_similarity: f32,
}

#[derive(Clone)]
pub enum VersionRef {
    Reference(Box<Versions>),
    Invalid(u32),
}

#[derive(Clone)]
pub struct Versions {
    pub versions: [NodePersistRef; 4],
    pub next: VersionRef,
}

pub struct NodePersist {
    pub version_id: u32, // Assuming VersionId is a type alias for u32
    pub prop_location: PropPersistRef,
    pub hnsw_level: u8, // Assuming HNSWLevel is a type alias for u8
    pub version_ref: VersionRef,
    pub neighbors: [NeighbourPersist; 10], // Bounded array of size 10
    pub parent: Option<NodePersistRef>,
    pub child: Option<NodePersistRef>,
}

impl NodePersist {
    pub fn new(
        version_id: VersionId,
        prop_location: PropPersistRef,
        hnsw_level: HNSWLevel,
        version_ref: VersionRef,
        neighbors: Vec<NeighbourPersist>,
        parent: Option<NodePersistRef>,
        child: Option<NodePersistRef>,
    ) -> NodePersist {
        let mut fixed_neighbors = [NeighbourPersist {
            node: 0,
            cosine_similarity: 0.0,
        }; 10];
        for (index, neighbor) in neighbors.iter().enumerate().take(10) {
            fixed_neighbors[index] = *neighbor;
        }
        NodePersist {
            version_id,
            prop_location,
            hnsw_level,
            version_ref,
            neighbors: fixed_neighbors,
            parent,
            child,
        }
    }
}

pub fn persist_node_update_loc(
    wal_file: Arc<File>,
    node: NodeRef,
    hnsw_level: HNSWLevel,
) -> Result<(), WaCustomError> {
    // Lock the Mutex to access the neighbors
    println!(" For node {} ", node);
    let neighbors_lock = node
        .neighbors
        .read()
        .map_err(|_| WaCustomError::MutexPoisoned("convert_node_to_node_persist".to_owned()))?;

    // Convert neighbors from NodeRef to NodePersistRef
    let mut fixed_neighbors = [NeighbourPersist {
        node: (0),
        cosine_similarity: 0.0,
    }; 10];
    for (index, neighbor) in neighbors_lock.iter().enumerate().take(10) {
        fixed_neighbors[index] = match neighbor {
            NeighbourRef::Ready {
                node: nodex,
                cosine_similarity,
            } => match nodex.get_location() {
                Some(loca) => NeighbourPersist {
                    node: loca,
                    cosine_similarity: *cosine_similarity,
                },
                None => {
                    println!(" issue in node location {} ", nodex);
                    return Err(WaCustomError::InvalidLocationNeighborEncountered(
                        "neighbours loop".to_owned(),
                        nodex.prop.id.clone(),
                    ));
                }
            },
            NeighbourRef::Pending(x) => {
                return Err(WaCustomError::PendingNeighborEncountered(x.to_string()));
            }
        };
    }

    // Convert parent and child
    let parent = node
        .get_parent()
        .and_then(|p| p.get_location())
        .unwrap_or(0);
    let child = node.get_child().and_then(|c| c.get_location()).unwrap_or(0);

    let mut nprst = NodePersist {
        hnsw_level,
        neighbors: fixed_neighbors,
        parent: Some(parent),
        child: Some(child),
        prop_location: node.get_prop_location().unwrap_or((0, 0)),
        version_ref: VersionRef::Invalid(0),
        version_id: node.version_id + 1,
    };

    let mut location = node.location.write().unwrap();
    if let Some(loc) = *location {
        let file_loc = write_node_to_file_at_offset(&mut nprst, &wal_file, loc.into());
        *location = Some(file_loc);
    } else {
        let file_loc = write_node_to_file(&mut nprst, &wal_file);
        *location = Some(file_loc);
    }

    //TODO: update the previous node_persist with the new location in its next field
    // only needs to update the tuple
    Ok(())
}

// end

pub fn map_node_persist_ref_to_node(
    vec_store: VectorStore,
    node_ref: NodePersistRef,
    cosine_similarity: f32,
    vec_level: HNSWLevel,
    vec_id: VectorId,
) -> NeighbourRef {
    // logic to map NodePersistRef to Node
    //
    match load_neighbor_persist_ref(vec_level, node_ref) {
        Some(nodex) => {
            return NeighbourRef::Ready {
                node: nodex,
                cosine_similarity,
            }
        }
        None => return NeighbourRef::Pending(node_ref),
    };
}

pub fn load_node_from_node_persist(
    vec_store: VectorStore,
    node_persist: NodePersist,
    persist_loc: NodeFileRef,
    prop: Arc<NodeProp>,
) -> NodeRef {
    // Convert neighbors from NodePersistRef to NeighbourRef
    let neighbors_result: Vec<NeighbourRef> = node_persist
        .neighbors
        .iter()
        .filter_map(|nref| {
            if nref.node != 0 {
                Some(map_node_persist_ref_to_node(
                    vec_store.clone(),
                    nref.node,
                    nref.cosine_similarity,
                    node_persist.hnsw_level,
                    prop.id.clone(),
                ))
            } else {
                None
            }
        })
        .collect();
    // Wrap neighbors in Arc<Mutex<Vec<NeighbourRef>>>
    let neighbors = Arc::new(RwLock::new(neighbors_result));

    // Convert parent and child
    let parent = if let Some(parent_ref) = node_persist.parent {
        load_neighbor_persist_ref(node_persist.hnsw_level, node_persist.parent.unwrap())
    } else {
        None
    };
    let parent = Arc::new(RwLock::new(parent));

    let child = if let Some(child_ref) = node_persist.child {
        load_neighbor_persist_ref(node_persist.hnsw_level, node_persist.child.unwrap())
    } else {
        None
    };
    let child = Arc::new(RwLock::new(child));

    // Create and return NodeRef
    Arc::new(Node {
        prop,
        location: Arc::new(RwLock::new(Some(persist_loc))),
        prop_location: Arc::new(RwLock::new(Some(node_persist.prop_location))),
        neighbors,
        parent,
        child,
        //previous: Some(persist_loc),
        version_id: node_persist.version_id,
    })
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

// fn write_to_end_of_file(file_path: &str, data: &[u8]) -> std::io::Result<(u64, usize)> {
//     let mut file = OpenOptions::new().append(true).open(file_path)?;
//     let offset = file.seek(SeekFrom::End(0))?;
//     file.write_all(data)?;
//     Ok((offset, data.len()))
// }

pub fn write_node_to_file(node: &mut NodePersist, mut file: &File) -> u32 {
    file.seek(SeekFrom::End(0)).expect("Seek failed"); // Explicitly move to the end

    // Serialize
    let result = node.serialize(&mut file);

    let offset = result.expect("Failed to serialize NodePersist & write to file");
    offset as u32
}

pub fn write_node_to_file_at_offset(node: &mut NodePersist, mut file: &File, offset: u64) -> u32 {
    // Seek to the specified offset before writing
    file.seek(SeekFrom::Start(offset))
        .expect("Failed to seek in file");

    // Serialize
    let result = node.serialize(&mut file);

    let offset = result.expect("Failed to serialize NodePersist & write to file");
    offset as u32
}

pub fn load_vector_id_lsmdb(level: HNSWLevel, vector_id: VectorId) -> Option<NodeRef> {
    return None;
}

pub fn load_neighbor_persist_ref(level: HNSWLevel, node_file_ref: u32) -> Option<NodeRef> {
    return None;
}
