use super::common::{tuple_to_string, WaCustomError};
use super::types::{
    HNSWLevel, NeighbourRef, Node, NodeFileRef, NodeProp, NodeRef, VectorId, VectorQt, VectorStore,
    VersionId,
};
use crate::models::cos_buffered_writer::*;
use crate::models::dry_run_writer::DryRunWriter;
use crate::models::serializer::*;
use std::cell::RefCell;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::rc::Rc;
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
        // Create a vector with default values
        let mut fixed_neighbors = vec![
            NeighbourPersist {
                node: NodePersistRef::Invalid,
                cosine_similarity: 999.999,
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

impl NodePersist {
    pub fn calculate_serialized_size(&self) -> std::io::Result<u64> {
        let mut writer = DryRunWriter::new();
        self.serialize(&mut writer)?;
        Ok(writer.bytes_written())
    }
}

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
pub fn prepare_node_update(
    node: NodeRef,
    hnsw_level: HNSWLevel,
    create_vrefs_flag: bool,
) -> Result<NodePersist, WaCustomError> {
    println!(" For node {} ", node);
    let neighbors_lock = node
        .neighbors
        .read()
        .map_err(|_| WaCustomError::MutexPoisoned("convert_node_to_node_persist".to_owned()))?;

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

    while fixed_neighbors.len() < 10 {
        fixed_neighbors.push(NeighbourPersist {
            node: NodePersistRef::Invalid,
            cosine_similarity: 999.999,
        });
    }

    let fixed_neighbors: [NeighbourPersist; 10] = fixed_neighbors.try_into().unwrap();

    let parent = node
        .get_parent()
        .and_then(|p| p.get_location())
        .map(NodePersistRef::DerefPending);
    let child = node
        .get_child()
        .and_then(|c| c.get_location())
        .map(NodePersistRef::DerefPending);

    let version_ref = if create_vrefs_flag {
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

    let nprst = NodePersist {
        hnsw_level,
        neighbors: fixed_neighbors,
        parent,
        child,
        prop_location: node.get_prop_location().unwrap_or((0, 0)),
        version_ref,
        version_id: node.version_id + 1,
    };

    Ok(nprst)
}

pub fn write_node_update(
    ver_file: &mut CustomBufferedWriter,
    mut nprst: NodePersist,
    current_location: Option<u64>,
) -> Result<u64, WaCustomError> {
    if let Some(loc) = current_location {
        Ok(write_node_to_file_at_offset(&mut nprst, ver_file, loc) as u64)
    } else {
        Ok(write_node_to_file(&mut nprst, ver_file) as u64)
    }
}

pub fn persist_node_update_loc(
    ver_file: &mut CustomBufferedWriter,
    node: NodeRef,
    hnsw_level: HNSWLevel,
    create_vrefs_flag: bool,
) -> Result<(), WaCustomError> {
    let nprst = prepare_node_update(node.clone(), hnsw_level, create_vrefs_flag)?;

    let mut location = node.location.write().unwrap();
    let current_location = location.map(u64::from);

    let file_loc = write_node_update(ver_file, nprst, current_location)?;
    *location = Some(file_loc as u32);

    Ok(())
}
// pub fn map_node_persist_ref_to_node(
//     vec_store: VectorStore,
//     node_ref: NodePersistRef,
//     cosine_similarity: f32,
//     vec_level: HNSWLevel,
//     vec_id: VectorId,
// ) -> NeighbourRef {
//     // logic to map NodePersistRef to Node
//     //
//     match load_neighbor_persist_ref(vec_level, node_ref) {
//         Some(nodex) => {
//             return NeighbourRef::Ready {
//                 node: nodex,
//                 cosine_similarity,
//             }
//         }
//         None => return NeighbourRef::Pending(node_ref),
//     };
// }

// pub fn load_node_from_node_persist(
//     vec_store: VectorStore,
//     node_persist: NodePersist,
//     persist_loc: NodeFileRef,
//     prop: Arc<NodeProp>,
// ) -> NodeRef {
//     // Convert neighbors from NodePersistRef to NeighbourRef
//     let neighbors_result: Vec<NeighbourRef> = node_persist
//         .neighbors
//         .iter()
//         .filter_map(|nref| {
//             if nref.node != 0 {
//                 Some(map_node_persist_ref_to_node(
//                     vec_store.clone(),
//                     nref.node,
//                     nref.cosine_similarity,
//                     node_persist.hnsw_level,
//                     prop.id.clone(),
//                 ))
//             } else {
//                 None
//             }
//         })
//         .collect();
//     // Wrap neighbors in Arc<Mutex<Vec<NeighbourRef>>>
//     let neighbors = Arc::new(RwLock::new(neighbors_result));

//     // Convert parent and child
//     let parent = if let Some(parent_ref) = node_persist.parent {
//         load_neighbor_persist_ref(node_persist.hnsw_level, node_persist.parent.unwrap())
//     } else {
//         None
//     };
//     let parent = Arc::new(RwLock::new(parent));

//     let child = if let Some(child_ref) = node_persist.child {
//         load_neighbor_persist_ref(node_persist.hnsw_level, node_persist.child.unwrap())
//     } else {
//         None
//     };
//     let child = Arc::new(RwLock::new(child));

//     // Create and return NodeRef
//     Arc::new(Node {
//         prop,
//         location: Arc::new(RwLock::new(Some(persist_loc))),
//         prop_location: Arc::new(RwLock::new(Some(node_persist.prop_location))),
//         neighbors,
//         parent,
//         child,
//         //previous: Some(persist_loc),
//         version_id: node_persist.version_id,
//     })
// }

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

pub fn write_node_to_file(node: &mut NodePersist, writer: &mut CustomBufferedWriter) -> u32 {
    // Assume CustomBufferWriter already handles seeking to the end
    // Serialize
    let result = node.serialize(writer);
    let offset = result.expect("Failed to serialize NodePersist & write to file");
    offset as u32
}

pub fn write_node_to_file_at_offset(
    node: &mut NodePersist,
    writer: &mut CustomBufferedWriter,
    offset: u64,
) -> u32 {
    // Seek to the specified offset before writing
    writer
        .seek(SeekFrom::Start(offset))
        .expect("Failed to seek in file");
    // Serialize
    let result = node.serialize(writer);
    let offset = result.expect("Failed to serialize NodePersist & write to file");
    offset as u32
}

pub fn load_vector_id_lsmdb(level: HNSWLevel, vector_id: VectorId) -> Option<NodeRef> {
    return None;
}

pub fn load_neighbor_persist_ref(level: HNSWLevel, node_file_ref: u32) -> Option<NodeRef> {
    return None;
}
