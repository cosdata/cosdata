use crate::models::file_persist::*;
use crate::models::persist::Persist;
use crate::models::rpc::VectorIdValue;
use crate::models::user::{AuthResp, Statistics};
use crate::models::{self, common::*};
use crate::models::{persist, types::*};
use crate::vector_store::{self, *};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use log::info;
use rand::Rng;
use std::fs::OpenOptions;
use std::sync::{Arc, Mutex, RwLock};

pub async fn init_vector_store(
    name: String,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    max_cache_level: u8,
) -> Result<(), WaCustomError> {
    if name.is_empty() {
        return Err(WaCustomError::InvalidParams);
    }

    let min = lower_bound.unwrap_or(-1.0);
    let max = upper_bound.unwrap_or(1.0);
    let vec = (0..size)
        .map(|_| {
            let mut rng = rand::thread_rng();

            let random_number: f32 = rng.gen_range(min..max);
            random_number
        })
        .collect::<Vec<f32>>();
    let vec_hash = VectorId::Int(-1);

    let exec_queue_neighbors = Arc::new(DashMap::new());

    let resolution = 1 as u8;
    let quant_dim = (size * resolution as usize / 32);
    let quantized_values: Vec<Vec<u32>> = quantize_to_u32_bits(&vec.clone(), 1);
    //let mpq: Vec<usize> = get_magnitude_plus_quantized_vec(&quantized_values, quant_dim);

    let vector_list = VectorQt {
        //mag: mpq,
        quant_vec: quantized_values,
        resolution: resolution,
    };

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    let prop_file = Arc::new(
        OpenOptions::new()
            .append(true)
            .open("prop.data")
            .expect("Failed to open file for writing"),
    );

    let wal_file = Arc::new(
        OpenOptions::new()
            .append(true)
            .open("index.0")
            .expect("Failed to open file for writing"),
    );

    let mut root: Option<NodeRef> = None;
    let mut prop_location: NodeFileRef = (0, 0);

    for l in 0..=max_cache_level {
        let prop = NodeProp::new(vec_hash.clone(), vector_list.clone().into());

        let nn = Node::new(prop.clone(), None, None, 0);

        if l == 0 {
            root = Some(nn.clone());
            prop_location = write_prop_to_file(&prop, &prop_file);
            nn.set_location(prop_location);
        }
        match persist_node_update_loc(wal_file.clone(), nn, l) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Failed node persist (init): {}", e);
            }
        };
    }

    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels).into_iter().rev().collect());
    let exec_queue_nodes = vec![];

    let vec_store = VectorStore {
        max_cache_level,
        database_name: name.clone(),
        root_vec: root.unwrap(),
        levels_prob: lp,
        quant_dim: (size / 32) as usize,
        prop_file,
        wal_file,
        exec_queue_neighbors,
        exec_queue_nodes,
    };

    let result = match get_app_env() {
        Ok(ain_env) => {
            ain_env
                .vector_store_map
                .insert(name.clone(), vec_store.clone());
            let rs = ain_env.persist.lock().unwrap().create_cf_family("main");
            // create the default CF for main index
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
    stream::iter(vecxx)
        .map(|(id, vec)| {
            let vec_store = vec_store.clone();
            let persist = persist.clone();
            async move {
                let root = &vec_store.root_vec;
                let vec_hash = convert_value(id);

                let quantized_values: Vec<Vec<u32>> = quantize_to_u32_bits(&vec.clone(), 1);
                //let mpq: Vec<usize> =
                // get_magnitude_plus_quantized_vec(&quantized_values, vec_store.quant_dim);

                let vector_list = VectorQt {
                    //mag: mpq,
                    quant_vec: quantized_values,
                    resolution: 1,
                };

                let vec_emb = VectorEmbedding {
                    raw_vec: Arc::new(vector_list.clone()),
                    hash_vec: vec_hash.clone(),
                };
                let lp = &vec_store.levels_prob;
                let iv = get_max_insert_level(rand::random::<f32>().into(), lp.clone());

                insert_embedding(
                    persist,
                    vec_store.clone(),
                    vec_emb,
                    root.clone(),
                    vec_store.max_cache_level.try_into().unwrap(),
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
    let root = &vector_store.root_vec;

    let quantized_values: Vec<Vec<u32>> = quantize_to_u32_bits(&query.clone(), 1);
    let mpq: Vec<usize> = get_magnitude_plus_quantized_vec(&quantized_values, vec_store.quant_dim);

    let vector_list = VectorQt {
        quant_vec: quantized_values,
        resolution: 1,
    };

    let vec_emb = VectorEmbedding {
        raw_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let results = ann_search(
        vec_store.clone(),
        vec_emb,
        root.clone(),
        vec_store.max_cache_level.try_into().unwrap(),
    );
    let output = remove_duplicates_and_filter(results);
    return output;
}

pub async fn fetch_vector_neighbors(
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Vec<Option<(VectorId, Vec<(VectorId, f32)>)>> {
    let results = vector_fetch(vec_store.clone(), vector_id);
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
