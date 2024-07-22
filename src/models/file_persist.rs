use super::common::{tuple_to_string, WaCustomError};
use super::types::{
    HNSWLevel, MergedNode, Neighbour, NodeFileRef, NodeProp, VectorId, VectorQt, VectorStore,
    VersionId,
};
use crate::models::serializer::*;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::sync::{Arc, RwLock};

pub type FileOffset = u32;
pub type BytesToRead = u32;

#[derive(Debug, Clone)]
pub enum NodePersistRef {
    Reference(Box<NodePersist>),
    DerefPending(FileOffset),
    Invalid,
}

pub type PropPersistRef = (FileOffset, BytesToRead);

#[derive(Debug, Clone)]
pub struct NeighbourPersist {
    pub node: NodePersistRef,
    pub cosine_similarity: f32,
}

#[derive(Debug, Clone)]
pub enum VersionRef {
    Reference(Box<Versions>),
    Invalid,
}

#[derive(Debug, Clone)]
pub struct Versions {
    pub versions: [NodePersistRef; 4],
    pub next: VersionRef,
}

#[derive(Debug, Clone)]
pub struct NodePersist {
    pub version_id: VersionId,
    pub prop_location: PropPersistRef,
    pub hnsw_level: HNSWLevel,
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
        // Create a vector with default values
        let mut fixed_neighbors = vec![
            NeighbourPersist {
                node: NodePersistRef::Invalid,
                cosine_similarity: 0.0,
            };
            10
        ];

        // Copy over the provided neighbors
        for (index, neighbor) in neighbors.into_iter().enumerate().take(10) {
            fixed_neighbors[index] = neighbor;
        }

        // Convert the vector to an array
        let fixed_neighbors: [NeighbourPersist; 10] = fixed_neighbors.try_into().unwrap();

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

use std::fmt;

impl fmt::Display for NodePersist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NodePersist {{")?;
        writeln!(f, "    version_id: {},", self.version_id)?;
        writeln!(f, "    prop_location: {:?},", self.prop_location)?;
        writeln!(f, "    hnsw_level: {},", self.hnsw_level)?;
        writeln!(f, "    version_ref: {:?},", self.version_ref)?;
        writeln!(f, "    neighbors: [")?;
        for (i, neighbor) in self.neighbors.iter().enumerate() {
            writeln!(
                f,
                "        {}: {{ node: {:?}, cosine_similarity: {} }},",
                i, neighbor.node, neighbor.cosine_similarity
            )?;
        }
        writeln!(f, "    ],")?;
        writeln!(f, "    parent: {:?},", self.parent)?;
        writeln!(f, "    child: {:?}", self.child)?;
        write!(f, "}}")
    }
}

pub fn read_node_from_file(file: &mut File, offset: u32) -> std::io::Result<NodePersist> {
    // Seek to the specified offset
    file.seek(SeekFrom::Start(offset as u64))?;

    // Deserialize the NodePersist from the current position
    let node = NodePersist::deserialize(file, offset)?;

    // Pretty print the node
    println!("Read NodePersist from offset {}:", offset);
    println!("{}", node);

    Ok(node)
}
// end
pub fn persist_node_update_loc(
    wal_file: Arc<File>,
    node: Arc<Node>,
    hnsw_level: HNSWLevel,
    create_vrefs_flag: bool,
) -> Result<(), WaCustomError> {
    println!(" For node {} ", node);
    let neighbors_lock = node
        .neighbors
        .read()
        .map_err(|_| WaCustomError::MutexPoisoned("convert_node_to_node_persist".to_owned()))?;

    // Create a vector to hold neighbors
    let mut fixed_neighbors = Vec::with_capacity(10);

    for neighbor in neighbors_lock.iter().take(10) {
        match neighbor {
            NeighbourRef::Ready {
                node: nodex,
                cosine_similarity,
            } => {
                if let Some(loca) = nodex.get_location() {
                    fixed_neighbors.push(NeighbourPersist {
                        node: NodePersistRef::DerefPending(loca),
                        cosine_similarity: *cosine_similarity,
                    });
                } else {
                    println!(" issue in node location {} ", nodex);
                    return Err(WaCustomError::InvalidLocationNeighborEncountered(
                        "neighbours loop".to_owned(),
                        nodex.prop.id.clone(),
                    ));
                }
            }
            NeighbourRef::Pending(x) => {
                return Err(WaCustomError::PendingNeighborEncountered(x.to_string()));
            }
        };
    }

    // Pad the vector with default values if needed
    while fixed_neighbors.len() < 10 {
        fixed_neighbors.push(NeighbourPersist {
            node: NodePersistRef::Invalid,
            cosine_similarity: 0.0,
        });
    }

    // Convert the vector to an array
    let fixed_neighbors: [NeighbourPersist; 10] = fixed_neighbors.try_into().unwrap();

    // Convert parent and child
    let parent = node
        .get_parent()
        .and_then(|p| p.get_location())
        .map(NodePersistRef::DerefPending);
    let child = node
        .get_child()
        .and_then(|c| c.get_location())
        .map(NodePersistRef::DerefPending);

    let vref = if create_vrefs_flag {
        VersionRef::Reference(Box::new(Versions {
            versions: [
                NodePersistRef::Invalid,
                NodePersistRef::Invalid,
                NodePersistRef::Invalid,
                NodePersistRef::Invalid,
            ],
            next: VersionRef::Invalid,
        }))
    } else {
        VersionRef::Invalid
    };

    let mut nprst = NodePersist {
        hnsw_level,
        neighbors: fixed_neighbors,
        parent,
        child,
        prop_location: node.get_prop_location().unwrap_or((0, 0)),
        version_ref: vref,
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

    Ok(())
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
