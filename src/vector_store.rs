use crate::models::common::*;
use dashmap::DashMap;
use futures::future::{join_all, BoxFuture, FutureExt};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::task;

use crate::models::types::*;

pub async fn ann_search(
    vec_store: Arc<VectorStore>,
    vector_emb: VectorEmbedding,
    cur_entry: VectorId,
    cur_level: i8,
) -> Option<Vec<(VectorId, f32)>> {
    if cur_level == -1 {
        return Some(vec![]);
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
            )
            .await;

            let y = cosine_similarity(&fvec, &vtm.vector_list);
            let z = if z.is_empty() {
                vec![(cur_entry.clone(), y)]
            } else {
                z
            };
            let cloned_z = z.clone();
            let recursive_call = Box::pin(async move {
                let x = ann_search(
                    vec_store.clone(),
                    vector_emb.clone(),
                    z[0].0.clone(),
                    cur_level - 1,
                )
                .await;
                return x;
            });
            let result = recursive_call.await;
            return add_option_vecs(&result, &Some(cloned_z));
        } else {
            if cur_level > vec_store.max_cache_level {
                let xvtm = get_vector_from_db(&vec_store.database_name, cur_entry.clone()).await;
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
                    )
                    .await;
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

pub async fn insert_embedding(
    vec_store: Arc<VectorStore>,
    vector_emb: VectorEmbedding,
    cur_entry: VectorId,
    cur_level: i8,
    max_insert_level: i8,
) {
    if cur_level == -1 {
        return;
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
            )
            .await;

            let y = cosine_similarity(&fvec, &vtm.vector_list);
            let z = if z.is_empty() {
                vec![(cur_entry.clone(), y)]
            } else {
                z
            };

            let vec_store_clone = vec_store.clone();
            let vector_emb_clone = vector_emb.clone();
            let z_clone = z.clone();

            if cur_level <= max_insert_level {
                let recursive_call = Box::pin(async move {
                    insert_embedding(
                        vec_store.clone(),
                        vector_emb.clone(),
                        z[0].0.clone(),
                        cur_level - 1,
                        max_insert_level,
                    )
                    .await;
                });
                recursive_call.await;
                insert_node_create_edges(
                    vec_store_clone,
                    fvec,
                    vector_emb_clone.hash_vec.clone(),
                    z_clone,
                    cur_level,
                )
                .await;
            } else {
                let recursive_call = Box::pin(async move {
                    insert_embedding(
                        vec_store.clone(),
                        vector_emb.clone(),
                        z[0].0.clone(),
                        cur_level - 1,
                        max_insert_level,
                    )
                    .await;
                });
                recursive_call.await;
            }
        } else {
            if cur_level > vec_store.max_cache_level {
                let xvtm = get_vector_from_db(&vec_store.database_name, cur_entry.clone()).await;
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
                    )
                    .await;
                    insert_node_create_edges(
                        vec_store.clone(),
                        fvec,
                        vector_emb.hash_vec.clone(),
                        z,
                        cur_level,
                    )
                    .await;
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

async fn insert_node_create_edges(
    vec_store: Arc<VectorStore>,
    fvec: Arc<NumericVector>,
    hs: VectorId,
    nbs: Vec<(VectorId, f32)>,
    cur_level: i8,
) {
    let nv = Arc::new(VectorTreeNode {
        vector_list: fvec.clone(),
        neighbors: nbs.clone(),
    });

    vec_store.cache.insert((cur_level, hs.clone()), Some(nv));

    let tasks: Vec<_> = nbs
        .into_iter()
        .map(|(nb, cs)| {
            let vec_store = vec_store.clone();
            let hs = hs.clone();
            let cur_level = cur_level;
            task::spawn(async move {
                vec_store.cache.alter(
                    &(cur_level.clone(), nb.clone()).clone(),
                    |_, existing_value| match existing_value {
                        Some(res) => {
                            let vthm = res;
                            let mut ng = vthm.neighbors.clone();
                            ng.push((hs.clone(), cs));
                            ng.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            ng.dedup_by(|a, b| a.0 == b.0);
                            // ---------------------------
                            // -- TODO number of neighbors
                            // ---------------------------

                            let ng = ng.into_iter().take(6).collect::<Vec<_>>();

                            let nv = Arc::new(VectorTreeNode {
                                vector_list: vthm.vector_list.clone(),
                                neighbors: ng,
                            });
                            return Some(nv);
                        }
                        None => {
                            return existing_value.clone();
                        }
                    },
                );
            })
        })
        .collect();

    join_all(tasks).await;
}

async fn traverse_find_nearest(
    vec_store: Arc<VectorStore>,
    vtm: Arc<VectorTreeNode>,
    fvec: Arc<NumericVector>,
    hs: VectorId,
    hops: i8,
    skipm: Arc<DashMap<VectorId, ()>>,
    cur_level: i8,
) -> Vec<(VectorId, f32)> {
    traverse_find_nearest_inner(vec_store, vtm, fvec, hs, hops, skipm, cur_level).await
}

fn traverse_find_nearest_inner(
    vec_store: Arc<VectorStore>,
    vtm: Arc<VectorTreeNode>,
    fvec: Arc<NumericVector>,
    hs: VectorId,
    hops: i8,
    skipm: Arc<DashMap<VectorId, ()>>,
    cur_level: i8,
) -> BoxFuture<'static, Vec<(VectorId, f32)>> {
    async move {
        let tasks: Vec<_> = vtm
            .neighbors
            .clone()
            .into_iter()
            .filter(|(nb, _)| *nb != hs)
            .map(|(nb, _)| {
                let skipm = skipm.clone();
                let vec_store = vec_store.clone();
                let fvec = fvec.clone();
                let hs = hs.clone();
                task::spawn(async move {
                    if skipm.contains_key(&nb) {
                        vec![]
                    } else {
                        skipm.insert(nb.clone(), ());

                        let maybe_res = vec_store.cache.get(&(cur_level, nb.clone())).map(|res| {
                            let value = res.value().clone();
                            value
                        });

                        if let Some(Some(vthm)) = maybe_res {
                            let cs = cosine_similarity(&fvec, &vthm.vector_list);
                            // ---------------------------
                            // -- TODO number of hops
                            // ---------------------------

                            if hops < 16 {
                                let mut z = traverse_find_nearest_inner(
                                    vec_store.clone(),
                                    vthm.clone(),
                                    fvec.clone(),
                                    hs.clone(),
                                    hops + 1,
                                    skipm.clone(),
                                    cur_level,
                                )
                                .await;
                                z.push((nb.clone(), cs));
                                z
                            } else {
                                vec![(nb.clone(), cs)]
                            }
                        } else {
                            eprintln!(
                                "Error case, should not happen: {} key {:?}",
                                cur_level,
                                (cur_level, nb)
                            );
                            vec![]
                        }
                    }
                })
            })
            .collect();

        let results: Vec<Result<Vec<(VectorId, f32)>, task::JoinError>> = join_all(tasks).await;
        let mut nn: Vec<_> = results
            .into_iter()
            .filter_map(Result::ok) // Filter out the errors
            .flat_map(|inner_vec| inner_vec) // Flatten the inner vectors
            .collect();
        // ---------------------------
        // -- TODO number of closest to make edges
        // ---------------------------

        nn.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut seen = HashSet::new();
        nn.retain(|(vec_u8, _)| seen.insert(vec_u8.clone()));
        nn.into_iter().take(4).collect()
    }
    .boxed()
}

async fn get_vector_from_db(db_name: &str, entry: VectorId) -> Option<Arc<VectorTreeNode>> {
    // Your implementation to get vector from the database
    unimplemented!()
}
