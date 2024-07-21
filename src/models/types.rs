use crate::models::common::*;
use crate::models::versioning::VersionHash;
use bincode;
use dashmap::DashMap;
use http::Version;
use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::fs::*;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
pub type HNSWLevel = u8;

pub type NodeRef = Arc<Node>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeighbourRef {
    Ready {
        node: NodeRef,
        cosine_similarity: f32,
    },
    Pending(NodeFileRef),
}

pub type NodeFileRef = u32; // (offset)
pub type PropFileRef = (u32, u32); // (offset, size)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProp {
    pub id: VectorId,
    pub value: Arc<VectorQt>,
}

pub type VersionId = u16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub version_id: VersionId,
    pub prop: Arc<NodeProp>,
    pub location: Arc<RwLock<Option<NodeFileRef>>>,
    pub prop_location: Arc<RwLock<Option<PropFileRef>>>,
    pub neighbors: Arc<RwLock<Vec<NeighbourRef>>>,
    pub parent: Arc<RwLock<Option<NodeRef>>>,
    pub child: Arc<RwLock<Option<NodeRef>>>,
}

impl NodeProp {
    pub fn new(id: VectorId, value: Arc<VectorQt>) -> Arc<NodeProp> {
        Arc::new(NodeProp { id, value })
    }
}

impl Node {
    pub fn new(
        prop: Arc<NodeProp>,
        loc: Option<NodeFileRef>,
        prop_loc: Option<PropFileRef>,
        version_id: VersionId,
    ) -> NodeRef {
        Arc::new(Node {
            prop,
            location: Arc::new(RwLock::new(loc)),
            prop_location: Arc::new(RwLock::new(prop_loc)),
            neighbors: Arc::new(RwLock::new(Vec::new())),
            parent: Arc::new(RwLock::new(None)),
            child: Arc::new(RwLock::new(None)),
            version_id,
        })
    }

    pub fn add_ready_neighbor(&self, neighbor: NodeRef, cosine_similarity: f32) {
        let mut neighbors = self.neighbors.write().unwrap();
        neighbors.push(NeighbourRef::Ready {
            node: neighbor,
            cosine_similarity,
        });
    }

    pub fn add_ready_neighbors(&self, neighbors_list: Vec<(NodeRef, f32)>) {
        let mut neighbors = self.neighbors.write().unwrap();
        for (neighbor, cosine_similarity) in neighbors_list.iter() {
            neighbors.push(NeighbourRef::Ready {
                node: neighbor.clone(),
                cosine_similarity: *cosine_similarity,
            });
        }
    }

    pub fn get_neighbors(&self) -> Vec<NeighbourRef> {
        let neighbors = self.neighbors.read().unwrap();
        neighbors.clone()
    }

    pub fn set_parent(&self, parent: NodeRef) {
        let mut parent_lock = self.parent.write().unwrap();
        *parent_lock = Some(parent);
    }

    pub fn set_child(&self, child: NodeRef) {
        let mut child_lock = self.child.write().unwrap();
        *child_lock = Some(child);
    }

    pub fn get_parent(&self) -> Option<NodeRef> {
        let parent_lock = self.parent.read().unwrap();
        parent_lock.clone()
    }

    pub fn get_child(&self) -> Option<NodeRef> {
        let child_lock = self.child.read().unwrap();
        child_lock.clone()
    }

    pub fn set_location(&self, new_location: NodeFileRef) {
        let mut location_write = self.location.write().unwrap();
        *location_write = Some(new_location);
    }

    pub fn get_location(&self) -> Option<NodeFileRef> {
        let location_read = self.location.read().unwrap();
        *location_read
    }

    pub fn set_prop_location(&self, new_location: PropFileRef) {
        let mut location_write = self.prop_location.write().unwrap();
        *location_write = Some(new_location);
    }

    pub fn get_prop_location(&self) -> Option<PropFileRef> {
        let location_read = self.prop_location.read().unwrap();
        *location_read
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node {{ id: {:?},", self.prop.id)?; // Include self ID

        // Get references to inner data with locking (assuming RAII pattern)
        let parent_id = self
            .parent
            .read()
            .unwrap()
            .as_ref()
            .map(|p| p.prop.id.clone());
        let child_id = self
            .child
            .read()
            .unwrap()
            .as_ref()
            .map(|c| c.prop.id.clone());
        let neighbor_ids = self
            .neighbors
            .read()
            .unwrap()
            .iter()
            .filter_map(|n| match n {
                NeighbourRef::Ready { node, .. } => Some(node.prop.id.clone()),
                _ => None,
            })
            .collect::<Vec<VectorId>>();
        let location = self.location.read().unwrap();
        // Write data using write! or formatting options
        write!(
            f,
            " parent: {:?}, child: {:?}, neighbors: {:?}, location: {:?}",
            parent_id, child_id, neighbor_ids, location
        )?;

        write!(f, " }}")
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorQt {
    UnsignedByte {
        mag: u32,
        quant_vec: Vec<u8>,
    },
    SubByte {
        mag: u32,
        quant_vec: Vec<Vec<u32>>,
        resolution: u8,
    },
}

impl VectorQt {
    pub fn unsigned_byte(vec: &[f32]) -> Self {
        let quant_vec = simp_quant(vec);
        let mag = mag_square_u8(&quant_vec);
        Self::UnsignedByte { mag, quant_vec }
    }

    pub fn sub_byte(vec: &[f32], resolution: u8) -> Self {
        let quant_vec = quantize_to_u32_bits(vec, resolution);
        let mag = 0; //implement a proper magnitude calculation
        Self::SubByte {
            mag,
            quant_vec,
            resolution,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorId {
    Str(String),
    Int(i32),
}

// Implementing the std::fmt::Display trait for VectorId
impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorId::Str(s) => write!(f, "{}", s),
            VectorId::Int(i) => write!(f, "{}", i),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorTreeNode {
    pub vector_list: Arc<VectorQt>,
    pub neighbors: Vec<(VectorId, f32)>,
}

impl VectorTreeNode {
    // Serialize the VectorTreeNode to a byte vector
    pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let serialized = bincode::serialize(self)?;
        Ok(serialized)
    }

    // Deserialize a byte vector to a VectorTreeNode
    pub fn deserialize(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let deserialized = bincode::deserialize(bytes)?;
        Ok(deserialized)
    }
}

pub type SizeBytes = u32;

// needed to flatten and get uniques
pub type ExecQueueUpdate = Arc<DashMap<(HNSWLevel, VectorId), (NodeRef, SizeBytes)>>;

#[derive(Debug, Clone)]
pub struct MetaDb {
    pub env: Arc<Environment>,
    pub db: Arc<Database>,
}
#[derive(Debug, Clone)]
pub struct VectorStore {
    pub exec_queue_nodes: ExecQueueUpdate,
    pub max_cache_level: u8,
    pub database_name: String,
    pub root_vec: NodeRef,
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub quant_dim: usize,
    pub prop_file: Arc<File>,
    pub version_lmdb: MetaDb,
    pub current_version: Arc<RwLock<Option<VersionHash>>>,
}
impl VectorStore {
    // Get method
    pub fn get_current_version(
        &self,
    ) -> Result<
        Option<VersionHash>,
        std::sync::PoisonError<std::sync::RwLockReadGuard<'_, Option<VersionHash>>>,
    > {
        self.current_version.read().map(|guard| guard.clone())
    }

    // Set method
    pub fn set_current_version(
        &self,
        new_version: Option<VersionHash>,
    ) -> Result<(), std::sync::PoisonError<std::sync::RwLockWriteGuard<'_, Option<VersionHash>>>>
    {
        let mut write_guard = self.current_version.write()?;
        *write_guard = new_version;
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct VectorEmbedding {
    pub raw_vec: Arc<VectorQt>,
    pub hash_vec: VectorId,
}
