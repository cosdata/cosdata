use crate::app_context::AppContext;
use crate::models::buffered_io::*;
use crate::models::cache_loader::NodeRegistry;
use crate::models::common::*;
use crate::models::file_persist::*;
use crate::models::lazy_load::*;
use crate::models::meta_persist::*;
use crate::models::rpc::VectorIdValue;
use crate::models::types::*;
use crate::models::user::Statistics;
use crate::models::versioning::Hash;
use crate::models::versioning::VersionControl;
use crate::quantization::{Quantization, StorageType};
use crate::vector_store::*;
use arcshift::ArcShift;
use lmdb::{DatabaseFlags, Transaction};
use rand::Rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::array::TryFromSliceError;
use std::fs::OpenOptions;
use std::sync::Arc;

pub async fn init_vector_store(
    ctx: Arc<AppContext>,
    name: String,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    max_cache_level: u8,
) -> Result<Arc<VectorStore>, WaCustomError> {
    if name.is_empty() {
        return Err(WaCustomError::InvalidParams);
    }

    let quantization_metric = Arc::new(QuantizationMetric::Scalar);
    let storage_type = StorageType::UnsignedByte;
    let ain_env = get_app_env().map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let denv = ain_env.persist.clone();

    let metadata_db = denv
        .create_db(Some("metadata"), DatabaseFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let embeddings_db = denv
        .create_db(Some("embeddings"), DatabaseFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vcs = Arc::new(
        VersionControl::new(denv.clone())
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
    );

    let lmdb = MetaDb {
        env: denv.clone(),
        metadata_db: Arc::new(metadata_db),
        embeddings_db: Arc::new(embeddings_db),
    };

    let hash = store_current_version(&lmdb, vcs.clone(), "main", 0)?;

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

    let exec_queue_nodes: ExecQueueUpdate = STM::new(Vec::new(), 1, true);
    let vector_list = Arc::new(quantization_metric.quantize(&vec, storage_type)?);

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    let prop_file = Arc::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("prop.data")
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    );

    let mut root: LazyItemRef<MergedNode> = LazyItemRef::new_invalid();
    let mut prev: LazyItemRef<MergedNode> = LazyItemRef::new_invalid();

    let mut nodes = Vec::new();
    for l in (0..=max_cache_level).rev() {
        let prop = Arc::new(NodeProp {
            id: vec_hash.clone(),
            value: vector_list.clone(),
            location: Some((FileOffset(0), BytesToRead(0))),
        });
        let current_node = Arc::new(MergedNode {
            hnsw_level: HNSWLevel(l as u8),
            prop: ArcShift::new(PropState::Ready(prop.clone())),
            neighbors: EagerLazyItemSet::new(),
            parent: LazyItemRef::new_invalid(),
            child: LazyItemRef::new_invalid(),
        });

        let lazy_node = LazyItem::from_arc(hash, 0, current_node.clone());
        let nn = LazyItemRef::from_arc(hash, 0, current_node.clone());

        if let Some(prev_node) = prev
            .item
            .get()
            .get_lazy_data()
            .and_then(|mut arc| arc.get().clone())
        {
            current_node.set_parent(prev.clone().item.get().clone());
            prev_node.set_child(lazy_node.clone());
        }
        prev = nn.clone();

        if l == 0 {
            root = nn.clone();
            let _prop_location = write_prop_to_file(&prop, &prop_file);
            current_node.set_prop_ready(prop);
        }
        nodes.push(nn.clone());
    }
    // TODO: include db name in the path
    let bufmans = &ctx.index_manager;
    for (l, nn) in nodes.iter_mut().enumerate() {
        match persist_node_update_loc(bufmans.clone(), &mut nn.item) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Failed node persist (init) for node {}: {}", l, e);
            }
        };
    }

    bufmans.flush_all()?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels, max_cache_level));

    let vec_store = Arc::new(VectorStore::new(
        exec_queue_nodes,
        max_cache_level,
        name.clone(),
        root,
        lp,
        (size / 32) as usize,
        prop_file,
        lmdb,
        ArcShift::new(hash),
        Arc::new(QuantizationMetric::Scalar),
        Arc::new(DistanceMetric::Cosine),
        StorageType::UnsignedByte,
        vcs,
    ));
    ain_env
        .vector_store_map
        .insert(name.clone(), vec_store.clone());

    Ok(vec_store)
}

pub fn run_upload_in_transaction(
    ctx: Arc<AppContext>,
    vec_store: Arc<VectorStore>,
    transaction_id: Hash,
    vecs: Vec<(VectorIdValue, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let current_version = transaction_id;

    let bufman = ctx
        .vec_raw_manager
        .get(&current_version)
        .map_err(|e| WaCustomError::BufIo(Arc::new(e)))?;

    vecs.into_par_iter()
        .map(|(id, vec)| {
            let hash_vec = convert_value(id);
            let vec_emb = RawVectorEmbedding {
                raw_vec: vec,
                hash_vec,
            };
            insert_embedding(bufman.clone(), vec_store.clone(), &vec_emb, current_version)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

pub fn run_upload(
    ctx: Arc<AppContext>,
    vec_store: Arc<VectorStore>,
    vecs: Vec<(VectorIdValue, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let current_version = vec_store
        .vcs
        .add_next_version("main")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    vec_store.set_current_version(current_version);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(format!("{}.vec_raw", *current_version))
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;
    let bufman = Arc::new(BufferManager::new(file).map_err(BufIoError::Io)?);
    vecs.into_par_iter()
        .map(|(id, vec)| {
            let hash_vec = convert_value(id);
            let vec_emb = RawVectorEmbedding {
                raw_vec: vec,
                hash_vec,
            };

            insert_embedding(bufman.clone(), vec_store.clone(), &vec_emb, current_version)
        })
        .collect::<Result<Vec<_>, _>>()?;
    bufman.flush()?;

    let env = vec_store.lmdb.env.clone();
    let metadata_db = vec_store.lmdb.metadata_db.clone();

    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let count_unindexed = txn
        .get(*metadata_db, &"count_unindexed")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))
        .and_then(|bytes| {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            Ok(u32::from_le_bytes(bytes))
        })?;

    txn.abort();

    if count_unindexed >= ctx.config.upload_threshold {
        index_embeddings(
            ctx.node_registry.clone(),
            &ctx.vec_raw_manager,
            vec_store.clone(),
            ctx.config.upload_process_batch_size,
        )?;
    }

    auto_commit_transaction(vec_store, ctx.index_manager.clone())?;
    ctx.index_manager.flush_all()?;

    Ok(())
}

pub async fn ann_vector_query(
    node_registry: Arc<NodeRegistry>,
    vec_store: Arc<VectorStore>,
    query: Vec<f32>,
) -> Result<Option<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    let vector_store = vec_store.clone();
    let vec_hash = VectorId::Str("query".to_string());
    let root = &vector_store.root_vec;
    let vector_list = vector_store
        .quantization_metric
        .quantize(&query, vector_store.storage_type)?;

    let vec_emb = QuantizedVectorEmbedding {
        quantized_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let results = ann_search(
        node_registry,
        vec_store.clone(),
        vec_emb,
        root.item.clone().get().clone(),
        HNSWLevel(vec_store.max_cache_level),
    )?;
    let output = remove_duplicates_and_filter(results);
    Ok(output)
}

pub async fn fetch_vector_neighbors(
    node_registry: Arc<NodeRegistry>,
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>> {
    let results = vector_fetch(node_registry, vec_store.clone(), vector_id);
    return results.expect("Failed fetching vector neighbors");
}

fn calculate_statistics(_: &[i32]) -> Option<Statistics> {
    // Placeholder for calculating statistics
    None
}

fn vector_knn(_vs: &Vec<f32>, _vecs: &Vec<f32>) -> Vec<(i8, i8, String, f64)> {
    // Placeholder for vector KNN
    vec![]
}
