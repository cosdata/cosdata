use smallvec::SmallVec;

use crate::distance::DistanceFunction;
use crate::models::common::*;
use crate::models::custom_buffered_writer::CustomBufferedWriter;
use crate::models::file_persist::*;
use crate::models::lazy_load::*;
use crate::models::meta_persist::*;
use crate::models::types::*;
use crate::storage::Storage;
use std::collections::HashSet;
use std::fs::File;
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
                            mut offset,
                            ..
                        } => {
                            if let Some(xloc) = offset.get() {
                                match load_neighbor_from_db(*xloc, &vec_store) {
                                    Ok(Some(info)) => Some(info),
                                    Ok(None) => None,
                                    Err(e) => {
                                        eprintln!("Error loading neighbor: {}", e);
                                        None
                                    }
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                    .collect();
                Some((vector_id.clone(), nes))
            }
            LazyItem::Valid {
                data: None,
                mut offset,
                ..
            } => {
                if let Some(xloc) = offset.get() {
                    match load_node_from_persist(*xloc, &vec_store) {
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
    offset: FileOffset,
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
    offset: FileOffset,
    vec_store: &Arc<VectorStore>,
) -> Result<Option<(VectorId, f32)>, WaCustomError> {
    // Placeholder function to load neighbor from database
    // TODO: Implement actual database loading logic
    Err(WaCustomError::LazyLoadingError(
        "Not implemented".to_string(),
    ))
}
pub fn insert_embedding(
    vec_store: Arc<VectorStore>,
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
        insert_embedding(
            vec_store.clone(),
            vector_emb.clone(),
            z_clone[0].clone(),
            cur_level - 1,
            max_insert_level,
        )?;
        insert_node_create_edges(
            vec_store.clone(),
            fvec,
            vector_emb.hash_vec.clone(),
            z,
            cur_level,
        );
    } else {
        insert_embedding(
            vec_store.clone(),
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
) -> Result<(), WaCustomError> {
    let (mut node_arc, location) = match &lznode {
        LazyItem::Valid {
            data: Some(node),
            offset,
            ..
        } => (node.clone(), offset.clone().get().clone()),
        LazyItem::Valid {
            data: None, offset, ..
        } => {
            if let Some(offset) = offset.clone().get().clone() {
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
        persist_node_update_loc(buf_writer, node.clone())?;
    }

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
    fvec: Arc<Storage>,
    hs: VectorId,
    nbs: Vec<(LazyItem<MergedNode>, f32)>,
    cur_level: i8,
) -> Result<(), WaCustomError> {
    let node_prop = NodeProp {
        id: hs.clone(),
        value: fvec.clone(),
        location: None,
    };
    let mut nn = Item::new(MergedNode::new(0, cur_level as u8)); // Assuming MergedNode::new exists
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

            neighbor_list.push((LazyItem::from_item(nn.clone()), cs));

            neighbor_list
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            neighbor_list.truncate(20);

            nbr1_node.get().add_ready_neighbors(neighbor_list);
        }
    }

    queue_node_prop_exec(LazyItem::from_item(nn), vec_store.prop_file.clone())?;

    Ok(())
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
                PropState::Pending(_) => {
                    return Err(WaCustomError::NodeError(
                        "Neighbor prop is in pending state".to_string(),
                    ))
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
