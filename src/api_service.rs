use crate::models::common::*;
use crate::models::types::*;
use crate::models::user::{AuthResp, Statistics};
use crate::vector_store;
use crate::vector_store::*;
use chrono::prelude::*;
use dashmap::DashMap;
use log::info;
use serde::{Deserialize, Serialize};
use std::f64::consts::E;
use std::sync::{Arc, Mutex, RwLock};

pub async fn init_vector_store(
    name: String,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    max_cache_level: i8,
) -> Result<(), WaCustomError> {
    if name.is_empty() {
        return Err(WaCustomError::CreateDatabaseFailed);
    }
    let ain_env = get_app_env();
    let vec = (0..size)
        .map(|_| {
            rand::random::<f32>() * (upper_bound.unwrap_or(1.0) - lower_bound.unwrap_or(-1.0))
                + lower_bound.unwrap_or(-1.0)
        })
        .collect::<Vec<f32>>();

    let vec_hash = hash_float_vec(vec.clone());

    let root = (vec_hash.clone(), vec.clone());

    let cache = Arc::new(DashMap::new());

    for l in 0..=max_cache_level {
        cache.insert(
            (l, vec_hash.clone()),
            Some(Arc::new(VectorTreeNode {
                vector_list: Arc::new(vec.clone()),
                neighbors: vec![],
            })),
        );
    }

    let vec_store = VectorStore {
        cache,
        max_cache_level,
        database_name: name.clone(),
        root_vec: root,
    };

    ain_env
        .vector_store_map
        .insert(name.clone(), vec_store.clone());
    println!(
        "vector store map: {:?}",
        ain_env.vector_store_map.clone().len()
    );
    Ok(())
}

pub async fn run_upload(vec_store: Arc<VectorStore>, vecxx: Vec<Vec<f32>>) -> Vec<()> {
    use futures::stream::{self, StreamExt};

    stream::iter(vecxx)
        .map(|vec| {
            let vec_store = vec_store.clone();
            async move {
                let rhash = &vec_store.root_vec.0;
                let vec_hash = hash_float_vec(vec.clone());
                let vec_emb = VectorEmbedding {
                    raw_vec: Arc::new(vec),
                    hash_vec: vec_hash,
                };

                let iv = find_value(rand::random::<f32>().into());
                insert_embedding(
                    vec_store.clone(),
                    vec_emb,
                    rhash.clone(),
                    vec_store.max_cache_level,
                    iv.try_into().unwrap(),
                )
                .await;
            }
        })
        .buffer_unordered(10)
        .collect::<Vec<_>>()
        .await
}

pub async fn ann_vector_query(
    vec_store: Arc<VectorStore>,
    query: NumericVector,
) -> Option<Vec<(VectorHash, f32)>> {
    let vector_store = vec_store.clone();
    let vec_hash = hash_float_vec(query.clone());
    let rhash = &vector_store.root_vec.0;

    let vec_emb = VectorEmbedding {
        raw_vec: Arc::new(query),
        hash_vec: vec_hash,
    };
    let results = ann_search(
        vec_store.clone(),
        vec_emb,
        rhash.to_vec(),
        vec_store.max_cache_level,
    )
    .await;
    return results;
}

fn calculate_statistics(_: &[i32]) -> Option<Statistics> {
    // Placeholder for calculating statistics
    None
}

fn vector_knn(vs: &Vec<f32>, vecs: &Vec<f32>) -> Vec<(i8, i8, String, f64)> {
    // Placeholder for vector KNN
    vec![]
}
