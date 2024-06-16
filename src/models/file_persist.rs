use super::common::WaCustomError;
use super::types::{
    HNSWLevel, NeighbourRef, Node, NodeProp, NodeRef, VectorId, VectorQt, VectorStore,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_cbor;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::io::{Seek, SeekFrom, Write};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

// Assuming the fixed size for neighbors and quant_vec
const MAX_NEIGHBORS: usize = 10; // Adjust as needed
const MAX_QUANT_VEC: usize = 10; // Adjust as needed

// persist structures

type NodePersistRef = (u32, u32); // (file_number, offset)

#[derive(Clone, Serialize, Deserialize)]
pub struct NeighbourPersist {
    pub node: NodePersistRef,
    pub cosine_similarity: f32,
}
#[derive(Serialize, Deserialize)]
pub struct NodePersist {
    // prop is not serialized in this context
    #[serde(skip_serializing)]
    pub prop: NodePersistProp,
    pub hnsw_level: HNSWLevel,
    pub location: NodePersistRef,
    pub neighbors: Vec<NeighbourPersist>,
    pub parent: Option<NodePersistRef>,
    pub child: Option<NodePersistRef>,
}

#[derive(Serialize, Deserialize)]
pub struct NodePersistProp {
    pub id: VectorId,
    pub vector: Arc<VectorQt>,
}

impl NodePersistProp {
    pub fn new(id: VectorId, vector: Arc<VectorQt>) -> NodePersistProp {
        NodePersistProp { id, vector }
    }
}

impl NodePersist {
    pub fn new(
        prop: NodePersistProp,
        hnsw_level: HNSWLevel,
        location: NodePersistRef,
        neighbors: Vec<NeighbourPersist>,
        parent: Option<NodePersistRef>,
        child: Option<NodePersistRef>,
    ) -> NodePersist {
        NodePersist {
            prop,
            hnsw_level,
            location,
            neighbors,
            parent,
            child,
        }
    }
}

pub fn convert_node_to_node_persist(
    node: NodeRef,
    prop: NodePersistProp,
    hnsw_level: HNSWLevel,
) -> Result<NodePersist, WaCustomError> {
    // Lock the Mutex to access the neighbors
    let neighbors_lock = node
        .neighbors
        .read()
        .map_err(|_| WaCustomError::PendingNeighborEncountered("Mutex poisoned".to_owned()))?;

    // Convert neighbors from NodeRef to NodePersistRef
    let neighbors: Result<Vec<NeighbourPersist>, _> = neighbors_lock
        .iter()
        .map(|neighbor| match neighbor {
            NeighbourRef::Ready {
                node: nodex,
                cosine_similarity,
            } => Ok(NeighbourPersist {
                node: nodex.location,
                cosine_similarity: *cosine_similarity,
            }),
            NeighbourRef::Pending(_) => Err(WaCustomError::PendingNeighborEncountered(
                "Pending neighbor encountered".to_owned(),
            )),
        })
        .collect();

    // Handle the Result of the neighbors conversion
    let neighbors = neighbors?;

    // Convert parent and child
    let parent = node
        .parent
        .read()
        .unwrap()
        .as_ref()
        .map(|parent_node| parent_node.location);
    let child = node
        .child
        .read()
        .unwrap()
        .as_ref()
        .map(|child_node| child_node.location);

    // Create NodePersist
    Ok(NodePersist {
        prop,
        hnsw_level,
        location: node.location,
        neighbors,
        parent,
        child,
    })
}

pub fn map_node_persist_ref_to_node(
    vec_store: VectorStore,
    node_ref: NodePersistRef,
    cosine_similarity: f32,
    vec_level: HNSWLevel,
    vec_id: VectorId,
) -> NeighbourRef {
    // logic to map NodePersistRef to Node
    //
    match vec_store.cache.get(&(vec_level, vec_id)) {
        Some(nodex) => {
            return NeighbourRef::Ready {
                node: nodex.value().clone(),
                cosine_similarity,
            }
        }
        None => return NeighbourRef::Pending(node_ref),
    };
}

pub fn load_node_from_node_persist(vec_store: VectorStore, node_persist: NodePersist) -> NodeRef {
    // Convert NodePersistProp to NodeProp
    let prop = NodeProp::new(node_persist.prop.id, node_persist.prop.vector.clone());

    // Convert neighbors from NodePersistRef to NeighbourRef
    let neighbors_result: Vec<NeighbourRef> = node_persist
        .neighbors
        .iter()
        .map(|nref| {
            map_node_persist_ref_to_node(
                vec_store.clone(),
                nref.node,
                nref.cosine_similarity,
                node_persist.hnsw_level,
                prop.id.clone(),
            )
        })
        .collect();

    // Wrap neighbors in Arc<Mutex<Vec<NeighbourRef>>>
    let neighbors = Arc::new(RwLock::new(neighbors_result));

    // Convert parent and child
    let parent = if let Some(parent_ref) = node_persist.parent {
        vec_store
            .cache
            .get(&(node_persist.hnsw_level, prop.id.clone()))
            .map(|node| node.value().clone())
    } else {
        None
    };
    let parent = Arc::new(RwLock::new(parent));

    let child = if let Some(child_ref) = node_persist.child {
        vec_store
            .cache
            .get(&(node_persist.hnsw_level, prop.id.clone()))
            .map(|node| node.value().clone())
    } else {
        None
    };
    let child = Arc::new(RwLock::new(child));

    // Create and return NodeRef
    Arc::new(Node {
        prop,
        location: node_persist.location,
        neighbors,
        parent,
        child,
    })
}

pub fn write_prop_to_file(prop: &NodePersistProp, filename: &str) -> (u32, u32) {
    let mut prop_bytes = Vec::new();
    //let result = encode(&prop);
    let result = serde_cbor::to_vec(&prop).unwrap();

    prop_bytes.extend_from_slice(result.as_ref());

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(filename)
        .expect("Failed to open file for writing");
    file.write_all(&prop_bytes)
        .expect("Failed to write to file");
    let offset = file.metadata().unwrap().len() - prop_bytes.len() as u64;
    (offset as u32, prop_bytes.len() as u32)
}

fn write_to_end_of_file(file_path: &str, data: &[u8]) -> std::io::Result<(u64, usize)> {
    let mut file = OpenOptions::new().append(true).open(file_path)?;
    let offset = file.seek(SeekFrom::End(0))?;
    file.write_all(data)?;
    Ok((offset, data.len()))
}

pub fn write_node_to_file(node: &mut NodePersist, filename: &str) -> (u32, u32) {
    // Check if neighbors vector needs padding
    let pad_size = MAX_NEIGHBORS.saturating_sub(node.neighbors.len());

    // Create a vector of dummy entries
    let dummy = NeighbourPersist {
        node: (0, 0),
        cosine_similarity: -999.0,
    };
    let mut padding = vec![dummy; pad_size];

    // Combine neighbors and padding
    let mut neighbors = node.neighbors.clone();
    neighbors.append(&mut padding);

    node.neighbors = neighbors;
    let mut node_bytes = Vec::new();
    let result = serde_cbor::to_vec(&node);
    if let Err(err) = result {
        panic!("Failed to CBOR encode NodePersist: {}", err);
    }
    node_bytes.extend_from_slice(result.unwrap().as_ref());

    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(filename)
        .expect("Failed to open file for writing");
    file.write_all(&node_bytes)
        .expect("Failed to write to file");
    let offset = file.metadata().unwrap().len() - node_bytes.len() as u64;
    (offset as u32, node_bytes.len() as u32)
}

pub fn write_node_to_file_at_offset(
    node: &mut NodePersist,
    filename: &str,
    offset: u64,
) -> (u32, u32) {
    // Check if neighbors vector needs padding
    let pad_size = MAX_NEIGHBORS.saturating_sub(node.neighbors.len());

    // Create a vector of dummy entries
    let dummy = NeighbourPersist {
        node: (0, 0),
        cosine_similarity: -999.0,
    };
    let mut padding = vec![dummy; pad_size];

    // Combine neighbors and padding
    let mut neighbors = node.neighbors.clone();
    neighbors.append(&mut padding);

    node.neighbors = neighbors;

    let mut node_bytes = Vec::new();
    let result = serde_cbor::to_vec(&node);
    if let Err(err) = result {
        panic!("Failed to CBOR encode NodePersist: {}", err);
    }
    node_bytes.extend_from_slice(result.unwrap().as_ref());

    let mut file = OpenOptions::new()
        .write(true)
        .open(filename)
        .expect("Failed to open file for writing");

    // Seek to the specified offset before writing
    file.seek(SeekFrom::Start(offset as u64))
        .expect("Failed to seek in file");

    file.write_all(&node_bytes)
        .expect("Failed to write to file");
    let written_bytes = node_bytes.len() as u32;
    (offset as u32, written_bytes)
}
