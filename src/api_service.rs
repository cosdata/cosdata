use crate::models::persist::Persist;
use crate::models::rpc::VectorIdValue;
use crate::models::user::{AuthResp, Statistics};
use crate::models::{self, common::*};
use crate::models::{persist, types::*};
use crate::vector_store::*;
use dashmap::DashMap;
use log::info;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, RwLock};

pub async fn init_vector_store(
    name: String,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    max_cache_level: i8,
) -> Result<(), WaCustomError> {
    if name.is_empty() {
        return Err(WaCustomError::InvalidParams);
    }

    let vec = (0..size)
        .map(|_| {
            let min = lower_bound.unwrap_or(-1.0);
            let max = upper_bound.unwrap_or(1.0);
            let mut rng = rand::thread_rng();

            let random_number: f32 = rng.gen_range(min..max);
            random_number
        })
        .collect::<Vec<f32>>();

    let vec_hash = VectorId::Str("waco_default_hidden_root".to_string());

    let cache = Arc::new(DashMap::new());

    let quantized_values: Vec<Vec<u8>> = quantize_to_u8_bits(&vec.clone());
    let mpq: (f64, Vec<u32>) = get_magnitude_plus_quantized_vec(quantized_values.to_vec());
    let vector_list = VectorW::QuantizedVector {
        mag: mpq.0,
        quant_vec: mpq.1,
        resolution: 1,
    };

    let root = (vec_hash.clone(), vector_list.clone());
    let vl_arc = Arc::new(vector_list);

    for l in 0..=max_cache_level {
        cache.insert(
            (l, vec_hash.clone()),
            Some(Arc::new(VectorTreeNode {
                neighbors: vec![],
                vector_list: vl_arc.clone(),
            })),
        );
    }
    let factor_levels = 20.0;
    let lp = Arc::new(generate_tuples(factor_levels).into_iter().rev().collect());
    let vec_store = VectorStore {
        cache,
        max_cache_level,
        database_name: name.clone(),
        root_vec: root,
        levels_prob: lp,
    };
    let result = match get_app_env() {
        Ok(ain_env) => {
            ain_env
                .vector_store_map
                .insert(name.clone(), vec_store.clone());
            let rs = ain_env.persist.lock().unwrap().create_cf_family("main"); // create the default CF for main index
            match rs {
                Ok(__) => {
                    println!(
                        "vector store map: {:?}",
                        ain_env.vector_store_map.clone().len()
                    );
                    Ok(())
                }
                Err(e) => Err(WaCustomError::CreateCFFailed(e.to_string())),
            }
        }

        Err(e) => Err(WaCustomError::CFReadWriteFailed(e.to_string())),
    };
    return result;
}

pub async fn run_upload(
    persist: Arc<Mutex<Persist>>,
    vec_store: Arc<VectorStore>,
    vecxx: Vec<(VectorIdValue, Vec<f32>)>,
) -> Vec<()> {
    use futures::stream::{self, StreamExt};

    stream::iter(vecxx)
        .map(|(id, vec)| {
            let vec_store = vec_store.clone();
            let persist = persist.clone();
            async move {
                let rhash = &vec_store.root_vec.0;
                let vec_hash = convert_value(id);

                let quantized_values: Vec<Vec<u8>> = quantize_to_u8_bits(&vec.clone());
                let mpq: (f64, Vec<u32>) =
                    get_magnitude_plus_quantized_vec(quantized_values.to_vec());
                let vector_list = VectorW::QuantizedVector {
                    mag: mpq.0,
                    quant_vec: mpq.1,
                    resolution: 1,
                };

                let vec_emb = VectorEmbedding {
                    raw_vec: Arc::new(vector_list),
                    hash_vec: vec_hash,
                };
                let lp = &vec_store.levels_prob;
                let iv = get_max_insert_level(rand::random::<f32>().into(), lp.clone());
                insert_embedding(
                    persist,
                    vec_store.clone(),
                    vec_emb,
                    vec![rhash.clone()],
                    vec_store.max_cache_level,
                    iv.try_into().unwrap(),
                );
            }
        })
        .buffer_unordered(10)
        .collect::<Vec<_>>()
        .await
}

pub async fn ann_vector_query(
    vec_store: Arc<VectorStore>,
    query: Vec<f32>,
) -> Option<Vec<(VectorId, f32)>> {
    let vector_store = vec_store.clone();
    let vec_hash = VectorId::Str("query".to_string());
    let rhash = &vector_store.root_vec.0;

    let quantized_values: Vec<Vec<u8>> = quantize_to_u8_bits(&query.clone());
    let mpq: (f64, Vec<u32>) = get_magnitude_plus_quantized_vec(quantized_values.to_vec());
    let vector_list = VectorW::QuantizedVector {
        mag: mpq.0,
        quant_vec: mpq.1,
        resolution: 1,
    };

    let vec_emb = VectorEmbedding {
        raw_vec: Arc::new(vector_list),
        hash_vec: vec_hash,
    };
    let results = ann_search(
        vec_store.clone(),
        vec_emb,
        rhash.clone(),
        vec_store.max_cache_level,
    );
    let output = remove_duplicates_and_filter(results);
    return output;
}

fn calculate_statistics(_: &[i32]) -> Option<Statistics> {
    // Placeholder for calculating statistics
    None
}

fn vector_knn(vs: &Vec<f32>, vecs: &Vec<f32>) -> Vec<(i8, i8, String, f64)> {
    // Placeholder for vector KNN
    vec![]
}
