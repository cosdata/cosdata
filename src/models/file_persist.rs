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
use std::sync::{Arc, Mutex, OnceLock};

// Assuming the fixed size for neighbors and quant_vec
const MAX_NEIGHBORS: usize = 10; // Adjust as needed
const MAX_QUANT_VEC: usize = 10; // Adjust as needed

// persist structures

type NodePersistRef = (u32, u32); // (file_number, offset)

#[derive(Clone, Serialize, Deserialize)]
struct NeighbourPersist {
    node: NodePersistRef,
    cosine_similarity: f32,
}
#[derive(Serialize, Deserialize)]
struct NodePersist {
    // prop is not serialized in this context
    #[serde(skip_serializing)]
    prop: NodePersistProp,
    hnsw_level: HNSWLevel,
    location: NodePersistRef,
    neighbors: Vec<NeighbourPersist>,
}

#[derive(Serialize, Deserialize)]
struct NodePersistProp {
    id: VectorId,
    vector: VectorQt,
}

impl NodePersistProp {
    fn new(id: VectorId, vector: VectorQt) -> NodePersistProp {
        NodePersistProp { id, vector }
    }
}

impl NodePersist {
    fn new(
        prop: NodePersistProp,
        hnsw_level: HNSWLevel,
        location: NodePersistRef,
        neighbors: Vec<NeighbourPersist>,
    ) -> NodePersist {
        NodePersist {
            prop,
            hnsw_level,
            location,
            neighbors,
        }
    }
}

fn convert_node_to_node_persist(
    node: NodeRef,
    hnsw_level: HNSWLevel,
) -> Result<NodePersist, WaCustomError> {
    // Extract data from the Node
    let prop = NodePersistProp::new(node.prop.id.clone(), node.prop.value.clone());

    // Convert neighbors from NodeRef to NodePersistRef
    let neighbors: Result<Vec<_>, _> = node
        .neighbors
        .iter()
        .map(|neighbor| match neighbor {
            NeighbourRef::Done {
                node,
                cosine_similarity,
            } => Ok(NeighbourPersist {
                node: node.location,
                cosine_similarity: cosine_similarity.clone(),
            }),
            NeighbourRef::Pending(xxx) => Err(WaCustomError::PendingNeighborEncountered(
                "Pending neighbor encountered".to_owned(),
            )),
        })
        .collect();

    // Handle the Result of the neighbors conversion
    let neighbors = neighbors?;

    // Create NodePersist
    Ok(NodePersist {
        prop,
        hnsw_level,
        location: node.location,
        neighbors,
    })
}

fn map_node_persist_ref_to_node(
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
            return NeighbourRef::Done {
                node: nodex.value().clone(),
                cosine_similarity,
            }
        }
        None => return NeighbourRef::Pending(node_ref),
    };
}

fn load_node_from_node_persist(vec_store: VectorStore, node_persist: NodePersist) -> NodeRef {
    // Convert NodePersistProp to NodeProp
    let prop = NodeProp::new(node_persist.prop.id, node_persist.prop.vector.clone());

    // Convert neighbors from NodePersistRef to NodeRef using the mapping function
    let neighbors_result: Vec<_> = node_persist
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

    let neighbors = neighbors_result;

    // Create Node
    Arc::new(Node {
        prop,
        location: node_persist.location,
        neighbors,
    })
}

fn write_prop_to_file(prop: &NodePersistProp, filename: &str) -> (usize, usize) {
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
    (offset as usize, prop_bytes.len())
}

fn write_to_end_of_file(file_path: &str, data: &[u8]) -> std::io::Result<(u64, usize)> {
    let mut file = OpenOptions::new().append(true).open(file_path)?;
    let offset = file.seek(SeekFrom::End(0))?;
    file.write_all(data)?;
    Ok((offset, data.len()))
}

fn aawrite_node_to_file(node: &NodePersist, filename: &str) -> (usize, usize) {
    let mut node_bytes = Vec::new();
    let result = serde_cbor::to_vec(node);
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
    (offset as usize, node_bytes.len())
}

fn write_node_to_file(node: &mut NodePersist, filename: &str) -> (usize, usize) {
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
    (offset as usize, node_bytes.len())
}
