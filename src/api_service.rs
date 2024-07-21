use crate::models::cos_buffered_writer::CustomBufferedWriter;
use crate::models::file_persist::*;
use crate::models::meta_persist::*;
use crate::models::rpc::VectorIdValue;
use crate::models::types::*;
use crate::models::user::{AuthResp, Statistics};
use crate::models::{self, common::*};
use crate::vector_store::{self, *};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use lmdb::{Database, DatabaseFlags, Environment, Error as LmdbError, Transaction, WriteFlags};
use rand::Rng;
use std::cell::RefCell;
use std::fs::OpenOptions;
use std::rc::Rc;
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

    let exec_queue_nodes = Arc::new(DashMap::new());
    let vector_list = VectorQt::unsigned_byte(&vec);

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    let prop_file = Arc::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("prop.data")
            .expect("Failed to open file for writing"),
    );

    let ver_file = Rc::new(RefCell::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("0.index")
            .expect("Failed to open file for writing"),
    ));
    let mut root: Option<NodeRef> = None;
    let mut prev: Option<NodeRef> = None;

    for l in 0..=max_cache_level {
        let prop = NodeProp::new(vec_hash.clone(), vector_list.clone().into());

        let nn = Node::new(prop.clone(), None, None, 0);

        if let Some(ref mut p) = prev {
            nn.set_parent(p.clone());
            p.set_child(nn.clone());
        }

        prev = Some(nn.clone());
        if l == 0 {
            root = Some(nn.clone());
            nn.set_prop_location(write_prop_to_file(&prop, &prop_file));
        }
        let mut writer =
            CustomBufferedWriter::new(ver_file.clone()).expect("Failed opening custom buffer");

        match persist_node_update_loc(&mut writer, nn.clone(), l, true) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Failed node persist (init): {}", e);
            }
        };
        println!("sssss: {}", nn.clone());
    }

    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels).into_iter().rev().collect());

    let result = match get_app_env() {
        Ok(ain_env) => {
            let denv = ain_env.persist.clone();

            let db_result = denv.create_db(None, DatabaseFlags::empty());
            match db_result {
                Ok(db) => {
                    let vec_store = Arc::new(VectorStore {
                        max_cache_level,
                        database_name: name.clone(),
                        root_vec: root.unwrap(),
                        levels_prob: lp,
                        quant_dim: (size / 32) as usize,
                        prop_file,
                        exec_queue_nodes,
                        version_lmdb: MetaDb {
                            env: denv.clone(),
                            db: Arc::new(db.clone()),
                        },
                        current_version: Arc::new(RwLock::new(None)),
                    });
                    ain_env
                        .vector_store_map
                        .insert(name.clone(), vec_store.clone());

                    let result = store_current_version(vec_store.clone(), "main".to_string(), 0);
                    let version_hash = result.expect("Failed to get VersionHash");
                    vec_store
                        .set_current_version(Some(version_hash))
                        .expect("failed to store version");

                    Ok(())
                }
                Err(e) => {
                    eprintln!("Failed node persist(nbr1): {}", e);
                    Err(WaCustomError::LmdbError(e.to_string()))
                }
            }
        }

        Err(e) => Err(WaCustomError::LmdbError(e.to_string())),
    };
    return result;
}

pub async fn run_upload(vec_store: Arc<VectorStore>, vecxx: Vec<(VectorIdValue, Vec<f32>)>) -> () {
    stream::iter(vecxx)
        .map(|(id, vec)| {
            let vec_store = vec_store.clone();
            async move {
                let root = &vec_store.root_vec;
                let vec_hash = convert_value(id);
                let vector_list = VectorQt::unsigned_byte(&vec);
                let vec_emb = VectorEmbedding {
                    raw_vec: Arc::new(vector_list.clone()),
                    hash_vec: vec_hash.clone(),
                };
                let lp = &vec_store.levels_prob;
                let iv = get_max_insert_level(rand::random::<f32>().into(), lp.clone());

                // TODO: handle the error
                insert_embedding(
                    vec_store.clone(),
                    vec_emb,
                    root.clone(),
                    vec_store.max_cache_level.try_into().unwrap(),
                    iv.try_into().unwrap(),
                )
                .expect("Failed inserting embedding");
            }
        })
        .buffer_unordered(10)
        .collect::<Vec<_>>()
        .await;
    match auto_commit_transaction(vec_store.clone(), vec_store.exec_queue_nodes.clone()) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed node persist(nbr1): {}", e);
        }
    };
    ()
}

pub async fn ann_vector_query(
    vec_store: Arc<VectorStore>,
    query: Vec<f32>,
) -> Result<Option<Vec<(VectorId, f32)>>, WaCustomError> {
    let vector_store = vec_store.clone();
    let vec_hash = VectorId::Str("query".to_string());
    let root = &vector_store.root_vec;
    let vector_list = VectorQt::unsigned_byte(&query);

    let vec_emb = VectorEmbedding {
        raw_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let results = ann_search(
        vec_store.clone(),
        vec_emb,
        root.clone(),
        vec_store.max_cache_level.try_into().unwrap(),
    )?;
    let output = remove_duplicates_and_filter(results);
    Ok(output)
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
