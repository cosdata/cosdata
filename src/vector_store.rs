use crate::models::common::*;
use crate::models::persist::Persist;
use crate::models::types::*;
use bincode;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
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
        let maybe_res = vec_store
            .cache
            .clone()
            .get(&(lev, vector_id.clone()))
            .map(|res| {
                let vthmm = res.value().clone();
                vthmm
            });

        results.push(if let Some(vthm) = maybe_res {
            if let Some(vth) = vthm {
                let ne = vth.neighbors.clone();
                Some((vector_id, ne))
            } else {
                None
            }
        } else {
            None
        });
    }

    return results;
}

pub fn ann_search(
    vec_store: Arc<VectorStore>,
    vector_emb: VectorEmbedding,
    cur_entry: VectorId,
    cur_level: i8,
) -> Option<Vec<(VectorId, f32)>> {
    if cur_level == -1 {
        return Some(vec![]);
    }
    let size = vec_store.cache.clone().len();
    println!("SIZE {}", size);

    let xx = vec_store.cache.clone();
    for i in 0..101 {
        let key = (0 as i8, VectorId::Int(i));
        let val = xx.get(&key);
        match val {
            Some(vv) => {
                println!("{:?} {:?}", i, vv.value());
            }
            None => {}
        }
    }
    let maybe_res = vec_store
        .cache
        .clone()
        .get(&(cur_level, cur_entry.clone()))
        .map(|res| {
            let fvec = vector_emb.raw_vec.clone();
            let vthm_mv = res.value().clone();
            (fvec, vthm_mv)
        });

    if let Some((fvec, vthm_mv)) = maybe_res {
        if let Some(vthm) = vthm_mv {
            let vtm = vthm.clone();
            let skipm = Arc::new(DashMap::new());
            let z = traverse_find_nearest(
                vec_store.clone(),
                vtm.clone(),
                fvec.clone(),
                vector_emb.hash_vec.clone(),
                0,
                skipm.clone(),
                cur_level,
                false,
            );

            let y = cosine_coalesce(&fvec, &vtm.vector_list, vec_store.quant_dim);
            let z = if z.is_empty() {
                vec![(cur_entry.clone(), y)]
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
        } else {
            if cur_level > vec_store.max_cache_level {
                let xvtm = get_vector_from_db(&vec_store.database_name, cur_entry.clone());
                if let Some(vtm) = xvtm {
                    let skipm = Arc::new(DashMap::new());
                    let z = traverse_find_nearest(
                        vec_store.clone(),
                        vtm.clone(),
                        fvec.clone(),
                        vector_emb.hash_vec.clone(),
                        0,
                        skipm.clone(),
                        cur_level,
                        false,
                    );
                    return Some(z);
                } else {
                    eprintln!(
                        "Error case, should have been found: {} {:?}",
                        cur_level, xvtm
                    );
                    return Some(vec![]);
                }
            } else {
                eprintln!("Error case, should not happen: {} ", cur_level);
                return Some(vec![]);
            }
        }
    } else {
        eprintln!("Error case, should not happen: {}", cur_level);
        return Some(vec![]);
    }
}

pub fn insert_embedding(
    persist: Arc<Mutex<Persist>>,
    vec_store: Arc<VectorStore>,
    vector_emb: VectorEmbedding,
    cur_entries: Vec<VectorId>,
    cur_level: i8,
    max_insert_level: i8,
) {
    if cur_level == -1 {
        return;
    }

    for cur_entry in cur_entries.iter() {
        let maybe_res = vec_store
            .cache
            .clone()
            .get(&(cur_level, cur_entry.clone()))
            .map(|res| {
                let fvec = vector_emb.raw_vec.clone();
                let vthm_mv = res.value().clone();
                (fvec, vthm_mv)
            });

        if let Some((fvec, vthm_mv)) = maybe_res {
            if let Some(vthm) = vthm_mv {
                let vtm = vthm.clone();
                let skipm = Arc::new(DashMap::new());
                let z = traverse_find_nearest(
                    vec_store.clone(),
                    vtm.clone(),
                    fvec.clone(),
                    vector_emb.hash_vec.clone(),
                    0,
                    skipm,
                    cur_level,
                    true,
                );

                let y = cosine_coalesce(&fvec, &vtm.vector_list, vec_store.quant_dim);
                let z = if z.is_empty() {
                    vec![(cur_entry.clone(), y)]
                } else {
                    z
                };
                let z_clone: Vec<_> = z.iter().take(1).map(|(first, _)| first.clone()).collect();

                if cur_level <= max_insert_level {
                    insert_embedding(
                        persist.clone(),
                        vec_store.clone(),
                        vector_emb.clone(),
                        z_clone.clone(),
                        cur_level - 1,
                        max_insert_level,
                    );
                    insert_node_create_edges(
                        persist.clone(),
                        vec_store.clone(),
                        fvec,
                        vector_emb.hash_vec.clone(),
                        z,
                        cur_level,
                    );
                } else {
                    let z_clone: Vec<_> =
                        z.iter().take(1).map(|(first, _)| first.clone()).collect();

                    insert_embedding(
                        persist.clone(),
                        vec_store.clone(),
                        vector_emb.clone(),
                        z_clone.clone(),
                        cur_level - 1,
                        max_insert_level,
                    );
                }
            } else {
                if cur_level > vec_store.max_cache_level {
                    let xvtm = get_vector_from_db(&vec_store.database_name, cur_entry.clone());
                    if let Some(vtm) = xvtm {
                        let skipm = Arc::new(DashMap::new());
                        let z = traverse_find_nearest(
                            vec_store.clone(),
                            vtm.clone(),
                            fvec.clone(),
                            vector_emb.hash_vec.clone(),
                            0,
                            skipm,
                            cur_level,
                            true,
                        );
                        insert_node_create_edges(
                            persist.clone(),
                            vec_store.clone(),
                            fvec,
                            vector_emb.hash_vec.clone(),
                            z,
                            cur_level,
                        );
                    } else {
                        eprintln!(
                            "Error case, should have been found: {} {:?}",
                            cur_level, xvtm
                        );
                    }
                } else {
                    eprintln!("Error case, should not happen: {} ", cur_level);
                }
            }
        } else {
            eprintln!("Error case, should not happen: {}", cur_level);
        }
    }
}

fn insert_node_create_edges(
    persist: Arc<Mutex<Persist>>,
    vec_store: Arc<VectorStore>,
    fvec: Arc<VectorW>,
    hs: VectorId,
    nbs: Vec<(VectorId, f32)>,
    cur_level: i8,
) {
    let nn = VectorTreeNode {
        vector_list: fvec.clone(),
        neighbors: nbs.clone(),
    };
    let nv = Arc::new(nn.clone());

    vec_store.cache.insert((cur_level, hs.clone()), Some(nv));

    // Serialize the vector_node
    let ser_vec = nn.serialize().unwrap();
    let ser_hs = bincode::serialize(&hs).unwrap();

    let _ = persist.lock().unwrap().put_cf("main", &ser_hs, &ser_vec);

    for (nb, cs) in nbs.into_iter() {
        vec_store.cache.alter(
            &(cur_level.clone(), nb.clone()).clone(),
            |_, existing_value| match existing_value {
                Some(res) => {
                    let vthm = res;
                    let mut ng = vthm.neighbors.clone();

                    // Extract and hash the original VectorId values
                    let original_hash = calculate_hash(&extract_ids(&ng));

                    ng.push((hs.clone(), cs));
                    ng.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    ng.dedup_by(|a, b| a.0 == b.0);

                    // ---------------------------
                    // -- TODO number of neighbors
                    // ---------------------------
                    let ng = ng.into_iter().take(20).collect::<Vec<_>>();

                    // Extract and hash the modified VectorId values
                    let new_hash = calculate_hash(&extract_ids(&ng));
                    let nn = VectorTreeNode {
                        vector_list: vthm.vector_list.clone(),
                        neighbors: ng,
                    };
                    let nv = Arc::new(nn.clone());

                    if original_hash != new_hash {
                        // Serialize the vector_node
                        let ser_vec = nn.serialize().unwrap();
                        let ser_nb = bincode::serialize(&nb).unwrap();
                        let _ = persist.lock().unwrap().put_cf("main", &ser_nb, &ser_vec);
                    }
                    return Some(nv);
                }
                None => {
                    return existing_value.clone();
                }
            },
        );
    }
}

fn traverse_find_nearest(
    vec_store: Arc<VectorStore>,
    vtm: Arc<VectorTreeNode>,
    fvec: Arc<VectorW>,
    hs: VectorId,
    hops: i8,
    skipm: Arc<DashMap<VectorId, ()>>,
    cur_level: i8,
    skip_hop: bool,
) -> Vec<(VectorId, f32)> {
    let mut tasks = vec![];

    for (index, (nb, _)) in vtm
        .neighbors
        .clone()
        .into_iter()
        .filter(|(nb, _)| *nb != hs)
        .enumerate()
    {
        //let skips = tapered_skips(1, index as i8, 20);

        if index % 2 != 0 && skip_hop && index > 4 {
            continue; // Skip this iteration if the index is odd
        }

        let skipm = skipm.clone();
        let vec_store = vec_store.clone();
        let fvec = fvec.clone();
        let hs = hs.clone();

        if !skipm.contains_key(&nb) {
            skipm.insert(nb.clone(), ());

            let maybe_res = vec_store.cache.get(&(cur_level, nb.clone())).map(|res| {
                let value = res.value().clone();
                value
            });

            if let Some(Some(vthm)) = maybe_res {
                let cs = cosine_coalesce(&fvec, &vthm.vector_list, vec_store.quant_dim);

                // ---------------------------
                // -- TODO number of hops
                // ---------------------------
                let full_hops = 30;
                if hops <= tapered_total_hops(full_hops, cur_level, vec_store.max_cache_level) {
                    let mut z = traverse_find_nearest(
                        vec_store.clone(),
                        vthm.clone(),
                        fvec.clone(),
                        hs.clone(),
                        hops + 1,
                        skipm.clone(),
                        cur_level,
                        skip_hop,
                    );
                    z.push((nb.clone(), cs));
                    tasks.push(z);
                } else {
                    tasks.push(vec![(nb.clone(), cs)]);
                }
            } else {
                eprintln!(
                    "Error case, should not happen: {} key {:?}",
                    cur_level,
                    (cur_level, nb)
                );
            }
        }
    }

    let mut nn: Vec<_> = tasks.into_iter().flatten().collect();
    nn.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut seen = HashSet::new();
    nn.retain(|(vec_u8, _)| seen.insert(vec_u8.clone()));

    // ---------------------------
    // -- TODO number of closest to make edges
    // ---------------------------

    nn.into_iter().take(5).collect()
}

fn get_vector_from_db(db_name: &str, entry: VectorId) -> Option<Arc<VectorTreeNode>> {
    // Your implementation to get vector from the database
    unimplemented!()
}
