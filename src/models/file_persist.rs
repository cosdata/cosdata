use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

use super::common::WaCustomError;
use super::types::{HNSWLevel, NeighbourRef, Node, NodeProp, NodeRef, VectorId, VectorQt, VectorStore};

// persist structures

type NodePersistRef = (u32, u32); // (file_number, offset)

#[derive(Serialize, Deserialize)]
struct NodePersist {
    prop: NodePersistProp,
    hnsw_level: HNSWLevel,
    location: NodePersistRef,
    neighbors: Vec<NodePersistRef>,
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
        neighbors: Vec<NodePersistRef>,
    ) -> NodePersist {
        NodePersist {
            prop,
            hnsw_level,
            location,
            neighbors,
        }
    }
}

fn convert_node_to_node_persist(node: NodeRef, hnsw_level: HNSWLevel) -> NodePersist {
    // Extract data from the Node
    let prop = NodePersistProp::new(node.prop.id.clone(), node.prop.value.clone());

    // Convert neighbors from NodeRef to NodePersistRef
    let neighbors = node
        .neighbors
        .iter()
        .map(|neighbor| neighbor.location)
        .collect();

    // Create NodePersist
    NodePersist {
        prop,
        hnsw_level,
        location: node.location,
        neighbors,
    }
}

fn map_node_persist_ref_to_node(
    vec_store: VectorStore,
    node_ref: NodePersistRef,
    vec_level: HNSWLevel,
) -> NeighbourRef {
    // logic to map NodePersistRef to Node
    //
    match vec_store.cache.get(&(vec_level, node_ref)) {
        Some(node) => return NeighbourRef::Done(node.value().clone()),
        None => return NeighbourRef::Pending(node_ref),
    };
}

fn load_node_from_node_persist(
    vec_store: VectorStore,
    node_persist: NodePersist,
) -> NodeRef {
    // Convert NodePersistProp to NodeProp
    let prop = NodeProp::new(node_persist.prop.id, node_persist.prop.vector.clone());

    // Convert neighbors from NodePersistRef to NodeRef using the mapping function
    let neighbors_result: Vec<_> = node_persist
        .neighbors
        .iter()
        .map(|&nref| map_node_persist_ref_to_node(vec_store.clone(), nref, node_persist.hnsw_level))
        .collect();

    let neighbors = neighbors_result;

    // Create Node
    Arc::new(Node {
        prop,
        location: node_persist.location,
        neighbors,
    })
}
