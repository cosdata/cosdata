use crate::models::chunked_list::*;
use crate::models::common::*;
use crate::models::custom_buffered_writer::CustomBufferedWriter;
use crate::models::file_persist::*;
use crate::models::meta_persist::*;
use crate::models::rpc::VectorIdValue;
use crate::models::types::*;
use crate::models::user::Statistics;
use crate::quantization::Quantization;
use crate::quantization::StorageType;
use crate::vector_store::*;
use lmdb::DatabaseFlags;
use lmdb::Transaction;
use rand::Rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::array::TryFromSliceError;
use std::cell::RefCell;
use std::fs::OpenOptions;
use std::io::Write;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

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

    let quantization_metric = Arc::new(QuantizationMetric::Scalar);
    let storage_type = StorageType::UnsignedByte;

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

    let exec_queue_nodes: ExecQueueUpdate = Arc::new(RwLock::new(Vec::new()));
    let vector_list = Arc::new(quantization_metric.quantize(&vec, storage_type));

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

    let mut writer =
        CustomBufferedWriter::new(ver_file.clone()).expect("Failed opening custom buffer");

    let mut root: Option<LazyItemRef<MergedNode>> = None;
    let mut prev: Option<LazyItemRef<MergedNode>> = None;

    let mut nodes = Vec::new();
    for l in 0..=max_cache_level {
        let prop = Arc::new(NodeProp {
            id: vec_hash.clone(),
            value: vector_list.clone(),
            location: Some((0, 0)),
        });
        let current_node = Arc::new(RwLock::new(MergedNode {
            version_id: 0, // Initialize with appropriate version ID
            hnsw_level: l as u8,
            prop: Arc::new(RwLock::new(PropState::Ready(prop.clone()))),
            neighbors: LazyItems::new(),
            parent: None,
            child: None,
            versions: LazyItems::new(),
            persist_flag: Arc::new(RwLock::new(true)),
        }));

        let nn = LazyItemRef::new_with_lock(current_node.clone());

        if let Some(prev_node) = prev
            .as_ref()
            .and_then(|prev| prev.item.read().unwrap().data.clone())
        {
            let mut prev_guard = prev_node.write().unwrap();
            current_node.write().unwrap().set_parent(prev.clone());
            prev_guard.set_child(Some(nn.clone()));
        }
        prev = Some(nn.clone());

        if l == 0 {
            root = Some(nn.clone());
            let prop_location = write_prop_to_file(&prop, &prop_file);
            current_node.read().unwrap().set_prop_ready(prop);
        }
        nodes.push(nn.clone());
        println!("sssss: {:?}", nn);
    }

    for (l, nn) in nodes.iter_mut().enumerate() {
        match persist_node_update_loc(&mut writer, &mut *nn.item.write().unwrap()) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Failed node persist (init): {}", e);
            }
        };
    }

    writer
        .flush()
        .expect("Final Custom Buffered Writer flush failed ");
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels).into_iter().rev().collect());
    let ain_env = get_app_env().map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let denv = ain_env.persist.clone();

    let metadata_db = denv
        .create_db(Some("metadata"), DatabaseFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let embeddings_db = denv
        .create_db(Some("embeddings"), DatabaseFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vec_store = Arc::new(VectorStore::new(
        exec_queue_nodes,
        max_cache_level,
        name.clone(),
        root.unwrap(),
        lp,
        (size / 32) as usize,
        prop_file,
        MetaDb {
            env: denv.clone(),
            metadata_db: Arc::new(metadata_db.clone()),
            embeddings_db: Arc::new(embeddings_db),
        },
        Arc::new(RwLock::new(None)),
        Arc::new(QuantizationMetric::Scalar),
        Arc::new(DistanceMetric::Cosine),
        StorageType::UnsignedByte,
    ));
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

pub async fn run_upload(vec_store: Arc<VectorStore>, vecxx: Vec<(VectorIdValue, Vec<f32>)>) {
    vecxx.into_par_iter().for_each(|(id, vec)| {
        let hash_vec = convert_value(id);
        let storage = vec_store
            .quantization_metric
            .quantize(&vec, vec_store.storage_type);
        let vec_emb = VectorEmbedding {
            raw_vec: Arc::new(storage),
            hash_vec,
        };

        insert_embedding(vec_store.clone(), &vec_emb).expect("Failed to inert embedding to LMDB");
    });

    let env = vec_store.lmdb.env.clone();
    let metadata_db = vec_store.lmdb.metadata_db.clone();

    let txn = env.begin_rw_txn().expect("Failed to begin transaction");

    let count_unindexed = txn
        .get(*metadata_db, &"count_unindexed")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))
        .and_then(|bytes| {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            Ok(u32::from_le_bytes(bytes))
        })
        .expect("Failed to retrieve `count_unindexed`");

    txn.abort();

    if count_unindexed >= 100 {
        index_embeddings(vec_store.clone()).expect("Failed to index embeddings");
    }

    // Update version
    let ver = vec_store
        .get_current_version()
        .unwrap()
        .expect("No current version found");
    let new_ver = ver.version + 1;

    // Create new version file
    let ver_file = Rc::new(RefCell::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}.index", new_ver))
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to open new version file: {}", e))
            })
            .unwrap(),
    ));

    let mut writer =
        CustomBufferedWriter::new(ver_file.clone()).expect("Failed opening custom buffer");

    match auto_commit_transaction(vec_store.clone(), &mut writer) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed node persist(nbr1): {}", e);
        }
    };
}

pub async fn ann_vector_query(
    vec_store: Arc<VectorStore>,
    query: Vec<f32>,
) -> Result<Option<Vec<(VectorId, f32)>>, WaCustomError> {
    let vector_store = vec_store.clone();
    let vec_hash = VectorId::Str("query".to_string());
    let root = &vector_store.root_vec;
    let vector_list = vector_store
        .quantization_metric
        .quantize(&query, vector_store.storage_type);

    let vec_emb = VectorEmbedding {
        raw_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let results = ann_search(
        vec_store.clone(),
        vec_emb,
        root.item.read().unwrap().clone(),
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
    return results.expect("Failed fetching vector neighbors");
}

fn calculate_statistics(_: &[i32]) -> Option<Statistics> {
    // Placeholder for calculating statistics
    None
}

fn vector_knn(vs: &Vec<f32>, vecs: &Vec<f32>) -> Vec<(i8, i8, String, f64)> {
    // Placeholder for vector KNN
    vec![]
}
