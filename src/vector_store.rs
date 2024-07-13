use crate::models::common::*;
use crate::models::file_persist::*;
use crate::models::persist::Persist;
use crate::models::types::*;
use bincode;
use dashmap::DashMap;
use futures::stream::Collect;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use smallvec::SmallVec;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

pub fn vector_fetch(
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Vec<Option<(VectorId, Vec<(VectorId, f32)>)>> {
    let mut results: Vec<Option<(VectorId, Vec<(VectorId, f32)>)>> = Vec::new();

    let vector_id = vector_id.clone();
    // Loop through all cache levels
    for lev in 0..vec_store.max_cache_level {
        let vector_id = vector_id.clone();
        let maybe_res = load_vector_id_lsmdb(lev, vector_id.clone());

        let neighbors = if let Some(vth) = maybe_res {
            let nes: Vec<(VectorId, f32)> = vth
                .neighbors
                .clone()
                .read() // Locking the RwLock for reading
                .unwrap() // Handle the possibility of a poisoned lock
                .iter() // Iterate over the Vec
                .filter_map(|ne| match ne {
                    NeighbourRef::Ready {
                        node,
                        cosine_similarity,
                    } => Some((node.prop.id.clone(), *cosine_similarity)),
                    NeighbourRef::Pending(_) => None,
                })
                .collect();
            Some((vector_id, nes))
        } else {
            None
        };

        results.push(neighbors);
    }

    results
}

pub fn ann_search(
    vec_store: Arc<VectorStore>,
    vector_emb: VectorEmbedding,
    cur_entry: NodeRef,
    cur_level: i8,
) -> Option<Vec<(NodeRef, f32)>> {
    if cur_level == -1 {
        return Some(vec![]);
    }
    let fvec = vector_emb.raw_vec.clone();

    let mut skipm = HashSet::new();
    skipm.insert(vector_emb.hash_vec.clone());

    let z = traverse_find_nearest(
        vec_store.clone(),
        cur_entry.clone(),
        fvec.clone(),
        vector_emb.hash_vec.clone(),
        0,
        &mut skipm,
        cur_level,
        false,
    );

    let cs = cosine_coalesce(&fvec, &cur_entry.prop.value);
    let z = if z.is_empty() {
        vec![(cur_entry.clone(), cs)]
    } else {
        z
    };
    let result = ann_search(
        vec_store.clone(),
        vector_emb.clone(),
        z[0].0.clone(),
        cur_level - 1,
    );
    return add_option_vecs(&result, &Some(z));
}

pub fn insert_embedding(
    persist: Arc<Mutex<Persist>>,
    vec_store: Arc<VectorStore>,
    vector_emb: VectorEmbedding,
    cur_entry: NodeRef,
    cur_level: i8,
    max_insert_level: i8,
) {
    if cur_level == -1 {
        return;
    }

    let fvec = vector_emb.raw_vec.clone();
    let mut skipm = HashSet::new();
    skipm.insert(vector_emb.hash_vec.clone());
    let z = traverse_find_nearest(
        vec_store.clone(),
        cur_entry.clone(),
        fvec.clone(),
        vector_emb.hash_vec.clone(),
        0,
        &mut skipm,
        cur_level,
        true,
    );

    let cs = cosine_coalesce(&fvec, &cur_entry.prop.value);
    let z = if z.is_empty() {
        vec![(cur_entry.clone(), cs)]
    } else {
        z
    };
    let z_clone: Vec<_> = z.iter().map(|(first, _)| first.clone()).collect();

    if cur_level <= max_insert_level {
        insert_embedding(
            persist.clone(),
            vec_store.clone(),
            vector_emb.clone(),
            z_clone[0].clone(),
            cur_level - 1,
            max_insert_level,
        );
        insert_node_create_edges(
            vec_store.clone(),
            fvec,
            vector_emb.hash_vec.clone(),
            z,
            cur_level,
        );
    } else {
        let z_clone: Vec<_> = z.iter().map(|(first, _)| first.clone()).collect();

        insert_embedding(
            persist.clone(),
            vec_store.clone(),
            vector_emb.clone(),
            z_clone[0].clone(),
            cur_level - 1,
            max_insert_level,
        );
    }
}

pub fn queue_node_prop_exec(
    prop_file: Arc<File>,
    exec_queue_update_neighbors: ExecQueueUpdateNeighbors,
    exec_queue_nodes: &mut ExecQueueInsertNodes,
    node: NodeRef,
    hnsw_level: HNSWLevel,
) -> Result<(), WaCustomError> {
    let prop_location = write_prop_to_file(&node.prop, &prop_file);
    node.set_prop_location(prop_location);
    exec_queue_nodes.push((node.clone()));

    let _ = node.neighbors.read().unwrap().iter().map(|nbr| match nbr {
        NeighbourRef::Ready {
            node: nbrx,
            cosine_similarity: _,
        } => {
            // TODO# calculate with specialized serialization formula
            let result = serde_cbor::to_vec(&nbrx);
            let size = result.unwrap().len();

            exec_queue_update_neighbors.insert(
                (hnsw_level, nbrx.prop.id.clone()),
                (nbrx.clone(), size as u32),
            );
        }
        NeighbourRef::Pending(_) => todo!(),
    });
    Ok(())
}

pub fn finalize_transaction(
    vec_store: Arc<VectorStore>,
    exec_queue_update_neighbors: ExecQueueUpdateNeighbors,
    exec_queue_nodes: &mut ExecQueueInsertNodes,
    hnsw_level: HNSWLevel,
) {
    // Calculate the total sum of SizeBytes
    let total_size_bytes: SizeBytes = exec_queue_update_neighbors
        .iter()
        .map(|entry| entry.value().1) // Extract the SizeBytes
        .sum();
    let mut delta_offset: u32 = 0;

    for node in exec_queue_nodes.clone() {
        // TODO# calculate with specialized serialization formula
        let result = serde_cbor::to_vec(&node);
        let size = result.unwrap().len();
        delta_offset += size as u32;

        let new_offset = total_size_bytes + delta_offset;
        node.set_location((new_offset, size as u32));
    }

    for item in exec_queue_update_neighbors.iter() {
        let nbr = item.value().0.clone();
        match persist_node_update_loc(vec_store.wal_file.clone(), nbr, hnsw_level as u8) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Failed node persist(nbr1): {}", e);
                return;
            }
        };
    }

    for node in exec_queue_nodes {
        match persist_node_update_loc(vec_store.wal_file.clone(), node.clone(), hnsw_level as u8) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Failed node persist(nbr1): {}", e);
                return;
            }
        };
    }
}

fn insert_node_create_edges(
    vec_store: Arc<VectorStore>,
    fvec: Arc<VectorQt>,
    hs: VectorId,
    nbs: Vec<(NodeRef, f32)>,
    cur_level: i8,
) {
    println!("xxx id:{} nei-len:{}", hs, nbs.len());
    let nd_p = NodeProp::new(hs.clone(), fvec.clone());

    let nn = Node::new(nd_p.clone(), None, 0);

    nn.add_ready_neighbors(nbs.clone());

    for (nbr1, cs) in nbs.into_iter() {
        let mut neighbor_list: Vec<(NodeRef, f32)> = nbr1
            .neighbors
            .read()
            .unwrap()
            .iter()
            .filter_map(|nbr2| {
                if let NeighbourRef::Ready {
                    node: nodex,
                    cosine_similarity,
                } = nbr2
                {
                    Some((nodex.clone(), *cosine_similarity))
                } else {
                    None
                }
            })
            .collect();

        // Add the current (nn, cs) to the list
        neighbor_list.push((nn.clone(), cs));

        // Sort by cosine similarity in descending order
        neighbor_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Deduplicate and take the first 20 items
        let mut seen = HashSet::new();
        neighbor_list.retain(|(node, _)| seen.insert(Arc::as_ptr(node) as *const _));
        neighbor_list.truncate(20);
        // Update nbr1's neighbors
        println!("zzz id:{} nei-len:{:?}", nbr1.prop.id, neighbor_list.len());

        {
            let mut locked_neighbors = nbr1.neighbors.write().unwrap();

            *locked_neighbors = neighbor_list
                .into_iter()
                .map(|(node, cosine_similarity)| NeighbourRef::Ready {
                    node,
                    cosine_similarity,
                })
                .collect();
        } // Scope ends, write lock is released
    }

    match queue_node_prop_exec(
        vec_store.prop_file.clone(),
        vec_store.exec_queue_neighbors.clone(),
        &mut vec_store.exec_queue_nodes.clone(),
        nn,
        cur_level as u8,
    ) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed node persist(nbr1): {}", e);
            return;
        }
    };
}

fn traverse_find_nearest(
    vec_store: Arc<VectorStore>,
    vtm: NodeRef,
    fvec: Arc<VectorQt>,
    hs: VectorId,
    hops: u8,
    skipm: &mut HashSet<VectorId>,
    cur_level: i8,
    skip_hop: bool,
) -> Vec<(NodeRef, f32)> {
    let mut tasks: SmallVec<[Vec<(NodeRef, f32)>; 24]> = SmallVec::new();

    // Lock the neighbors Mutex to access the neighbors
    let neighbors_lock = vtm.neighbors.read().unwrap();

    for (index, nref) in neighbors_lock.iter().enumerate() {
        match nref {
            NeighbourRef::Ready {
                node: nbr,
                cosine_similarity: _,
            } => {
                let nb = nbr.prop.id.clone();
                if index % 2 != 0 && skip_hop && index > 4 {
                    //println!("skipping {} at hop {} ", index, hops);
                    continue; // Skip this iteration if the index is odd
                }
                //println!("traverse index:{}  nref:{} hop:{} ", index, nbr.prop.id, hops);

                let vec_store = vec_store.clone();
                let fvec = fvec.clone();
                let hs = hs.clone();

                if skipm.insert(nb.clone()) {
                    //println!("processing {} at hop {} ", index, hops);
                    let cs = cosine_coalesce(&fvec, &nbr.prop.value);

                    // ---------------------------
                    // -- TODO number of hops
                    // ---------------------------
                    let full_hops = 30;
                    if hops
                        <= tapered_total_hops(full_hops, cur_level as u8, vec_store.max_cache_level)
                    {
                        let mut z = traverse_find_nearest(
                            vec_store.clone(),
                            nbr.clone(),
                            fvec.clone(),
                            hs.clone(),
                            hops + 1,
                            skipm,
                            cur_level,
                            skip_hop,
                        );
                        z.push((nbr.clone(), cs));
                        tasks.push(z);
                    } else {
                        tasks.push(vec![(nbr.clone(), cs)]);
                    }
                }
            }
            NeighbourRef::Pending(_) => eprintln!(
                "Error case, should not happen: {} key {:?}",
                cur_level,
                (cur_level)
            ),
        }
    }

    let mut nn: Vec<_> = tasks.into_iter().flatten().collect();
    nn.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut seen = HashSet::new();
    nn.retain(|(vec_u8, _)| seen.insert(vec_u8.clone().prop.id.clone()));

    // ---------------------------
    // -- TODO number of closest to make edges
    // ---------------------------

    nn.into_iter().take(5).collect()
}

fn get_vector_from_db(db_name: &str, entry: VectorId) -> Option<Arc<VectorTreeNode>> {
    // Your implementation to get vector from the database
    unimplemented!()
}
