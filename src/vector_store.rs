use crate::distance::DistanceFunction;
use crate::models::common::*;
use crate::models::custom_buffered_writer::CustomBufferedWriter;
use crate::models::file_persist::*;
use crate::models::lazy_load::*;
use crate::models::meta_persist::*;
use crate::models::types::*;
use crate::storage::Storage;
use arcshift::ArcShift;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use lmdb::Transaction;
use lmdb::WriteFlags;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use smallvec::SmallVec;
use std::array::TryFromSliceError;
use std::collections::HashSet;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::sync::Arc;

pub fn ann_search(
    vec_store: Arc<VectorStore>,
    vector_emb: VectorEmbedding,
    cur_entry: LazyItem<MergedNode>,
    cur_level: i8,
) -> Result<Option<Vec<(LazyItem<MergedNode>, f32)>>, WaCustomError> {
    if cur_level == -1 {
        return Ok(Some(vec![]));
    }

    let fvec = vector_emb.raw_vec.clone();
    let mut skipm = HashSet::new();
    skipm.insert(vector_emb.hash_vec.clone());

    let mut cur_node_arc = match cur_entry.clone() {
        LazyItem::Valid {
            data: Some(node), ..
        } => node,
        LazyItem::Valid {
            data: None,
            mut file_index,
            ..
        } => {
            if let Some(file_index) = file_index.get() {
                return Err(WaCustomError::LazyLoadingError(format!(
                    "Node at offset {} needs to be loaded",
                    file_index
                )));
            } else {
                return Err(WaCustomError::NodeError(
                    "Current entry is null".to_string(),
                ));
            }
        }
        _ => {
            return Err(WaCustomError::NodeError(
                "Current entry is null".to_string(),
            ))
        }
    };

    let cur_node = cur_node_arc.get();

    let mut prop_arc = cur_node.prop.clone();
    let prop_state = prop_arc.get();

    let node_prop = match &*prop_state {
        PropState::Ready(prop) => prop,
        PropState::Pending(_) => {
            return Err(WaCustomError::NodeError(
                "Node prop is in pending state".to_string(),
            ))
        }
    };

    let z = traverse_find_nearest(
        vec_store.clone(),
        cur_entry.clone(),
        fvec.clone(),
        vector_emb.hash_vec.clone(),
        0,
        &mut skipm,
        cur_level,
        false,
    )?;

    let dist = vec_store
        .distance_metric
        .calculate(&fvec, &node_prop.value)?;

    let z = if z.is_empty() {
        vec![(cur_entry.clone(), dist)]
    } else {
        z
    };

    let result = ann_search(
        vec_store.clone(),
        vector_emb.clone(),
        z[0].0.clone(),
        cur_level - 1,
    )?;

    Ok(add_option_vecs(&result, &Some(z)))
}

pub fn vector_fetch(
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Result<Vec<Option<(VectorId, Vec<(VectorId, f32)>)>>, WaCustomError> {
    let mut results = Vec::new();

    for lev in 0..vec_store.max_cache_level {
        let maybe_res = load_vector_id_lsmdb(lev, vector_id.clone());
        let neighbors = match maybe_res {
            LazyItem::Valid {
                data: Some(vth), ..
            } => {
                let mut vth = vth.clone();
                let nes: Vec<(VectorId, f32)> = vth
                    .get()
                    .neighbors
                    .iter()
                    .filter_map(|ne| match ne.1.clone() {
                        LazyItem::Valid {
                            data: Some(node), ..
                        } => get_vector_id_from_node(node.clone().get()).map(|id| (id, ne.0)),
                        LazyItem::Valid {
                            data: None,
                            mut file_index,
                            ..
                        } => {
                            if let Some(xloc) = file_index.get() {
                                match load_neighbor_from_db(xloc.clone(), &vec_store) {
                                    Ok(Some(info)) => Some(info),
                                    Ok(None) => None,
                                    Err(e) => {
                                        eprintln!("Error loading neighbor: {}", e);
                                        None
                                    }
                                }
                            } else {
                                None
                                // NonePaste, drop, or click to add files Create pull request
                            }
                        }
                        _ => None,
                    })
                    .collect();
                Some((vector_id.clone(), nes))
            }
            LazyItem::Valid {
                data: None,
                mut file_index,
                ..
            } => {
                if let Some(xloc) = file_index.get() {
                    match load_node_from_persist(xloc.clone(), &vec_store) {
                        Ok(Some((id, neighbors))) => Some((id, neighbors)),
                        Ok(None) => None,
                        Err(e) => {
                            eprintln!("Error loading vector: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        results.push(neighbors);
    }
    Ok(results)
}
fn load_node_from_persist(
    offset: FileIndex,
    vec_store: &Arc<VectorStore>,
) -> Result<Option<(VectorId, Vec<(VectorId, f32)>)>, WaCustomError> {
    // Placeholder function to load vector from database
    // TODO: Implement actual database loading logic
    Err(WaCustomError::LazyLoadingError(
        "Not implemented".to_string(),
    ))
}

// fn get_neighbor_info(nbr: &Neighbour) -> Option<(VectorId, f32)> {
//     let Some(node) = nbr.node.data.clone() else {
//         eprintln!("Neighbour node not initialized");
//         return None;
//     };
//     let guard = node.read().unwrap();

//     let prop_state = match guard.prop.read() {
//         Ok(guard) => guard,
//         Err(e) => {
//             eprintln!("Lock error when reading prop: {}", e);
//             return None;
//         }
//     };

//     match &*prop_state {
//         PropState::Ready(node_prop) => Some((node_prop.id.clone(), nbr.cosine_similarity)),
//         PropState::Pending(_) => {
//             eprintln!("Encountered pending prop state");
//             None
//         }
//     }
// }

fn get_vector_id_from_node(node: &MergedNode) -> Option<VectorId> {
    let mut prop_arc = node.prop.clone();
    match prop_arc.get() {
        PropState::Ready(node_prop) => Some(node_prop.id.clone()),
        PropState::Pending(_) => None,
    }
}

fn load_neighbor_from_db(
    offset: FileIndex,
    vec_store: &Arc<VectorStore>,
) -> Result<Option<(VectorId, f32)>, WaCustomError> {
    // Placeholder function to load neighbor from database
    // TODO: Implement actual database loading logic
    Err(WaCustomError::LazyLoadingError(
        "Not implemented".to_string(),
    ))
}

pub fn write_embedding<W: Write + Seek>(
    writer: &mut W,
    emb: &VectorEmbedding,
) -> Result<u32, WaCustomError> {
    // TODO: select a better value for `N` (number of bytes to pre-allocate)
    let serialized = rkyv::to_bytes::<_, 256>(emb)
        .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

    let len = serialized.len() as u32;

    let start = writer
        .stream_position()
        .map_err(|e| WaCustomError::FsError(e.to_string()))? as u32;

    writer
        .write_u32::<LittleEndian>(len)
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;

    writer
        .write_all(&serialized)
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;

    Ok(start)
}

fn read_embedding<R: Read + Seek>(
    reader: &mut R,
    offset: u32,
) -> Result<(VectorEmbedding, u32), WaCustomError> {
    reader
        .seek(SeekFrom::Start(offset as u64))
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let len = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let mut buf = vec![0; len as usize];

    reader
        .read_exact(&mut buf)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let emb = unsafe { rkyv::from_bytes_unchecked(&buf) }.map_err(|e| {
        WaCustomError::DeserializationError(format!("Failed to deserialize VectorEmbedding: {}", e))
    })?;

    let next = reader
        .stream_position()
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))? as u32;

    Ok((emb, next))
}

pub fn insert_embedding(
    vec_store: Arc<VectorStore>,
    emb: &VectorEmbedding,
) -> Result<(), WaCustomError> {
    let env = vec_store.lmdb.env.clone();
    let embedding_db = vec_store.lmdb.embeddings_db.clone();
    let metadata_db = vec_store.lmdb.metadata_db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let count_unindexed = match txn.get(*metadata_db, &"count_unindexed") {
        Ok(bytes) => {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            u32::from_le_bytes(bytes)
        }
        Err(lmdb::Error::NotFound) => 0,
        Err(err) => return Err(WaCustomError::DatabaseError(err.to_string())),
    };

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open("vec_raw.0")
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let offset = write_embedding(&mut file, emb)?.to_le_bytes();

    txn.put(
        *embedding_db,
        &emb.hash_vec.to_string(),
        &offset,
        WriteFlags::empty(),
    )
    .map_err(|e| WaCustomError::DatabaseError(format!("Failed to put data: {}", e)))?;

    txn.put(
        *metadata_db,
        &"count_unindexed",
        &(count_unindexed + 1).to_le_bytes(),
        WriteFlags::empty(),
    )
    .map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to update `count_unindexed`: {}", e))
    })?;

    txn.commit().map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
    })?;

    Ok(())
}

pub fn index_embeddings(
    vec_store: Arc<VectorStore>,
    upload_process_batch_size: usize,
) -> Result<(), WaCustomError> {
    let env = vec_store.lmdb.env.clone();
    let metadata_db = vec_store.lmdb.metadata_db.clone();

    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let mut count_indexed = match txn.get(*metadata_db, &"count_indexed") {
        Ok(bytes) => {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            u32::from_le_bytes(bytes)
        }
        Err(lmdb::Error::NotFound) => 0,
        Err(err) => return Err(WaCustomError::DatabaseError(err.to_string())),
    };

    let mut count_unindexed = match txn.get(*metadata_db, &"count_unindexed") {
        Ok(bytes) => {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            u32::from_le_bytes(bytes)
        }
        Err(lmdb::Error::NotFound) => 0,
        Err(err) => return Err(WaCustomError::DatabaseError(err.to_string())),
    };

    let next_file_offset = match txn.get(*metadata_db, &"next_file_offset") {
        Ok(bytes) => {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            u32::from_le_bytes(bytes)
        }
        Err(lmdb::Error::NotFound) => 0,
        Err(err) => return Err(WaCustomError::DatabaseError(err.to_string())),
    };

    txn.abort();

    let mut file = OpenOptions::new()
        .read(true)
        .open("vec_raw.0")
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let metadata = file
        .metadata()
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;
    let len = metadata.len() as u32;

    let mut i = next_file_offset;
    let mut embeddings = Vec::new();

    // `file` is not thread safe, so we have to collect all the embeddings in the current thread
    while i < len {
        let (embedding, next) = read_embedding(&mut file, i)?;
        embeddings.push(embedding);
        i = next;

        if embeddings.len() == upload_process_batch_size || i == len {
            // TODO: handle the errors
            let results: Vec<()> = embeddings
                .into_par_iter()
                .map(|embedding| {
                    let lp = &vec_store.levels_prob;
                    let iv = get_max_insert_level(rand::random::<f32>().into(), lp.clone());

                    index_embedding(
                        vec_store.clone(),
                        None,
                        embedding,
                        vec_store.root_vec.item.clone().get().clone(),
                        vec_store.max_cache_level.try_into().unwrap(),
                        iv.try_into().unwrap(),
                    )
                    .expect("index_embedding failed");
                })
                .collect();

            embeddings = Vec::new();

            let batch_size = results.len() as u32;
            count_indexed += batch_size;
            count_unindexed -= batch_size;

            let mut txn = env.begin_rw_txn().map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e))
            })?;

            txn.put(
                *metadata_db,
                &"count_indexed",
                &count_indexed.to_le_bytes(),
                WriteFlags::empty(),
            )
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to update `count_indexed`: {}", e))
            })?;

            txn.put(
                *metadata_db,
                &"count_unindexed",
                &count_unindexed.to_le_bytes(),
                WriteFlags::empty(),
            )
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to update `count_unindexed`: {}", e))
            })?;

            txn.put(
                *metadata_db,
                &"next_file_offset",
                &i.to_le_bytes(),
                WriteFlags::empty(),
            )
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to update `next_file_offset`: {}", e))
            })?;

            txn.commit().map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
            })?;
        }
    }

    Ok(())
}

pub fn index_embedding(
    vec_store: Arc<VectorStore>,
    parent: Option<LazyItem<MergedNode>>,
    vector_emb: VectorEmbedding,
    cur_entry: LazyItem<MergedNode>,
    cur_level: i8,
    max_insert_level: i8,
) -> Result<(), WaCustomError> {
    if cur_level == -1 {
        return Ok(());
    }

    let fvec = vector_emb.raw_vec.clone();
    let mut skipm = HashSet::new();
    skipm.insert(vector_emb.hash_vec.clone());

    let mut cur_node_arc = match cur_entry.clone() {
        LazyItem::Valid {
            data: Some(node), ..
        } => node,
        LazyItem::Valid {
            data: None,
            mut file_index,
            ..
        } => {
            if let Some(offset) = file_index.get() {
                return Err(WaCustomError::LazyLoadingError(format!(
                    "Node at offset {} needs to be loaded",
                    offset
                )));
            } else {
                return Err(WaCustomError::NodeError(
                    "Current entry is null".to_string(),
                ));
            }
        }
        _ => {
            return Err(WaCustomError::NodeError(
                "Current entry is null".to_string(),
            ))
        }
    };

    let cur_node = cur_node_arc.get();

    let mut prop_arc = cur_node.prop.clone();
    let prop_state = prop_arc.get();

    let node_prop = match &*prop_state {
        PropState::Ready(prop) => prop,
        PropState::Pending(_) => {
            return Err(WaCustomError::NodeError(
                "Node prop is in pending state".to_string(),
            ))
        }
    };

    let z = traverse_find_nearest(
        vec_store.clone(),
        cur_entry.clone(),
        fvec.clone(),
        vector_emb.hash_vec.clone(),
        0,
        &mut skipm,
        cur_level,
        true,
    )?;

    let dist = vec_store
        .distance_metric
        .calculate(&fvec, &node_prop.value)?;

    let z = if z.is_empty() {
        vec![(cur_entry.clone(), dist)]
    } else {
        z
    };

    let z_clone: Vec<_> = z.iter().map(|(first, _)| first.clone()).collect();

    if cur_level <= max_insert_level {
        let parent = insert_node_create_edges(
            vec_store.clone(),
            parent,
            fvec,
            vector_emb.hash_vec.clone(),
            z,
            cur_level,
        )
        .expect("Failed insert_node_create_edges");
        index_embedding(
            vec_store.clone(),
            Some(parent),
            vector_emb.clone(),
            z_clone[0].clone(),
            cur_level - 1,
            max_insert_level,
        )?;
    } else {
        index_embedding(
            vec_store.clone(),
            None,
            vector_emb.clone(),
            z_clone[0].clone(),
            cur_level - 1,
            max_insert_level,
        )?;
    }

    Ok(())
}

pub fn queue_node_prop_exec(
    lznode: LazyItem<MergedNode>,
    prop_file: Arc<File>,
    vec_store: Arc<VectorStore>,
) -> Result<(), WaCustomError> {
    let (mut node_arc, location) = match &lznode {
        LazyItem::Valid {
            data: Some(node),
            file_index,
            ..
        } => (node.clone(), file_index.clone().get().clone()),
        LazyItem::Valid {
            data: None,
            file_index,
            ..
        } => {
            if let Some(offset) = file_index.clone().get().clone() {
                return Err(WaCustomError::LazyLoadingError(format!(
                    "Node at offset {} needs to be loaded",
                    offset
                )));
            } else {
                return Err(WaCustomError::NodeError("Node is null".to_string()));
            }
        }
        _ => return Err(WaCustomError::NodeError("Node is null".to_string())),
    };

    let node = node_arc.get();
    let mut prop_arc = node.prop.clone();

    let prop_state = prop_arc.get();

    if let PropState::Ready(node_prop) = &*prop_state {
        let prop_location = write_prop_to_file(node_prop, &prop_file);
        node.set_prop_location(prop_location);
    } else {
        return Err(WaCustomError::NodeError(
            "Node prop is not ready".to_string(),
        ));
    }

    // Set persistence flag for the main node
    node.set_persistence(true);

    for neighbor in node.neighbors.iter() {
        if let LazyItem::Valid {
            data: Some(mut neighbor_arc),
            ..
        } = neighbor.1
        {
            let neighbor = neighbor_arc.get();
            neighbor.set_persistence(true);
        }
    }

    // Add the node to exec_queue_nodes
    let mut exec_queue = vec_store.exec_queue_nodes.clone();
    println!("queue length before {}", exec_queue.get().len());
    exec_queue
        .transactional_update(|queue| {
            let mut new_queue = queue.clone();
            new_queue.push(ArcShift::new(lznode.clone()));
            new_queue
        })
        .unwrap();
    println!("queue length after {}", exec_queue.get().len());

    Ok(())
}

pub fn link_prev_version(prev_loc: Option<u32>, offset: u32) {
    // todo , needs to happen in file persist
}
pub fn auto_commit_transaction(
    vec_store: Arc<VectorStore>,
    buf_writer: &mut CustomBufferedWriter,
) -> Result<(), WaCustomError> {
    // Retrieve exec_queue_nodes from vec_store
    let mut exec_queue_nodes_arc = vec_store.exec_queue_nodes.clone();
    let exec_queue_nodes = exec_queue_nodes_arc.get();

    // Iterate through the exec_queue_nodes and persist each node
    for node in exec_queue_nodes.iter() {
        println!("auto_commit_txn");
        persist_node_update_loc(buf_writer, node)?;
    }

    exec_queue_nodes_arc.update(Vec::new());

    // Update version
    let ver = vec_store
        .get_current_version()
        .expect("No current version found");
    let new_ver = ver.version + 1;
    let vec_hash =
        store_current_version(vec_store.clone(), "main".to_string(), new_ver).map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to store current version: {:?}", e))
        })?;

    vec_store.set_current_version(Some(vec_hash));

    Ok(())
}

fn insert_node_create_edges(
    vec_store: Arc<VectorStore>,
    parent: Option<LazyItem<MergedNode>>,
    fvec: Arc<Storage>,
    hs: VectorId,
    nbs: Vec<(LazyItem<MergedNode>, f32)>,
    cur_level: i8,
) -> Result<LazyItem<MergedNode>, WaCustomError> {
    let node_prop = NodeProp {
        id: hs.clone(),
        value: fvec.clone(),
        location: None,
    };
    let mut nn = ArcShift::new(MergedNode::new(0, cur_level as u8)); // Assuming MergedNode::new exists
    nn.get().set_prop_ready(Arc::new(node_prop));

    nn.get().add_ready_neighbors(nbs.clone());

    for (nbr1, cs) in nbs.into_iter() {
        if let LazyItem::Valid {
            data: Some(mut nbr1_node),
            ..
        } = nbr1.clone()
        {
            let mut neighbor_list: Vec<(LazyItem<MergedNode>, f32)> = nbr1_node
                .get()
                .neighbors
                .iter()
                .map(|nbr2| (nbr2.1, nbr2.0))
                .collect();

            neighbor_list.push((LazyItem::from_arcshift(nn.clone()), cs));

            neighbor_list
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            neighbor_list.truncate(20);

            nbr1_node.get().add_ready_neighbors(neighbor_list);
        }
    }
    println!("insert node create edges, queuing nodes");
    let lz_item = LazyItem::from_arcshift(nn);
    if let Some(parent) = parent {
        lz_item.get_data().unwrap().set_parent(parent.clone());
        parent.get_data().unwrap().set_child(lz_item.clone());
    }
    queue_node_prop_exec(lz_item.clone(), vec_store.prop_file.clone(), vec_store)?;

    Ok(lz_item)
}

fn traverse_find_nearest(
    vec_store: Arc<VectorStore>,
    vtm: LazyItem<MergedNode>,
    fvec: Arc<Storage>,
    hs: VectorId,
    hops: u8,
    skipm: &mut HashSet<VectorId>,
    cur_level: i8,
    skip_hop: bool,
) -> Result<Vec<(LazyItem<MergedNode>, f32)>, WaCustomError> {
    let mut tasks: SmallVec<[Vec<(LazyItem<MergedNode>, f32)>; 24]> = SmallVec::new();

    let mut node_arc = match vtm.clone() {
        LazyItem::Valid {
            data: Some(node), ..
        } => node,
        LazyItem::Valid {
            data: None,
            mut offset,
            ..
        } => {
            if let Some(offset) = offset.get() {
                return Err(WaCustomError::LazyLoadingError(format!(
                    "Node at offset {} needs to be loaded",
                    offset
                )));
            } else {
                return Err(WaCustomError::NodeError(
                    "Current entry is null".to_string(),
                ));
            }
        }
        _ => {
            return Err(WaCustomError::NodeError(
                "Current entry is null".to_string(),
            ))
        }
    };

    let node = node_arc.get();

    for (index, nref) in node.neighbors.iter().enumerate() {
        if let Some(mut neighbor_arc) = nref.1.get_data() {
            let neighbor = neighbor_arc.get();
            let mut prop_arc = neighbor.prop.clone();
            let prop_state = prop_arc.get();

            let node_prop = match prop_state {
                PropState::Ready(prop) => prop.clone(),
                PropState::Pending(loc) => {
                    return Err(WaCustomError::NodeError(format!(
                        "Neighbor prop is in pending state at loc: {:?}",
                        loc
                    )))
                }
            };

            let nb = node_prop.id.clone();

            if index % 2 != 0 && skip_hop && index > 4 {
                continue;
            }

            let vec_store = vec_store.clone();
            let fvec = fvec.clone();
            let hs = hs.clone();

            if skipm.insert(nb.clone()) {
                let dist = vec_store
                    .distance_metric
                    .calculate(&fvec, &node_prop.value)?;

                let full_hops = 30;
                if hops <= tapered_total_hops(full_hops, cur_level as u8, vec_store.max_cache_level)
                {
                    let mut z = traverse_find_nearest(
                        vec_store.clone(),
                        nref.1.clone(),
                        fvec.clone(),
                        hs.clone(),
                        hops + 1,
                        skipm,
                        cur_level,
                        skip_hop,
                    )?;
                    z.push((nref.1.clone(), dist));
                    tasks.push(z);
                } else {
                    tasks.push(vec![(nref.1.clone(), dist)]);
                }
            }
        }
    }

    let mut nn: Vec<_> = tasks.into_iter().flatten().collect();
    nn.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut seen = HashSet::new();
    nn.retain(|(lazy_node, _)| {
        if let LazyItem::Valid {
            data: Some(node_arc),
            ..
        } = &lazy_node
        {
            let mut node_arc = node_arc.clone();
            let node = node_arc.get();
            let mut prop_arc = node.prop.clone();
            let prop_state = prop_arc.get();
            if let PropState::Ready(node_prop) = &*prop_state {
                seen.insert(node_prop.id.clone())
            } else {
                false
            }
        } else {
            false
        }
    });

    Ok(nn.into_iter().take(5).collect())
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, sync::Arc};

    use rand::{distributions::Uniform, rngs::ThreadRng, thread_rng, Rng};

    use crate::{
        models::types::{VectorEmbedding, VectorId},
        quantization::{scalar::ScalarQuantization, Quantization, StorageType},
    };

    use super::{read_embedding, write_embedding};

    fn get_random_embedding(rng: &mut ThreadRng) -> VectorEmbedding {
        let range = Uniform::new(-1.0, 1.0);

        let vector: Vec<f32> = (0..rng.gen_range(100..200))
            .into_iter()
            .map(|_| rng.sample(&range))
            .collect();
        let raw_vec = Arc::new(ScalarQuantization.quantize(&vector, StorageType::UnsignedByte));

        VectorEmbedding {
            raw_vec,
            hash_vec: VectorId::Int(rng.gen()),
        }
    }

    #[test]
    fn test_embedding_serialization() {
        let mut rng = thread_rng();
        let embedding = get_random_embedding(&mut rng);

        let mut writer = Cursor::new(Vec::new());
        let offset = write_embedding(&mut writer, &embedding).unwrap();

        let mut reader = Cursor::new(writer.into_inner());
        let (deserialized, _) = read_embedding(&mut reader, offset).unwrap();

        assert_eq!(embedding, deserialized);
    }

    #[test]
    fn test_embeddings_serialization() {
        let mut rng = thread_rng();
        let embeddings: Vec<VectorEmbedding> =
            (0..20).map(|_| get_random_embedding(&mut rng)).collect();

        let mut writer = Cursor::new(Vec::new());

        for embedding in &embeddings {
            write_embedding(&mut writer, embedding).unwrap();
        }

        let mut offset = 0;
        let mut reader = Cursor::new(writer.into_inner());

        for embedding in embeddings {
            let (deserialized, next) = read_embedding(&mut reader, offset).unwrap();
            offset = next;

            assert_eq!(embedding, deserialized);
        }
    }
}
