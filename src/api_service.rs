use crate::app_context::AppContext;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::NodeRegistry;
use crate::models::common::*;
use crate::models::file_persist::*;
use crate::models::lazy_load::*;
use crate::models::meta_persist::store_current_version;
use crate::models::rpc::VectorIdValue;
use crate::models::types::*;
use crate::models::user::Statistics;
use crate::models::versioning::Hash;
use crate::models::versioning::VersionControl;
use crate::quantization::{Quantization, StorageType};
use crate::vector_store::*;
use arcshift::ArcShift;
use lmdb::Transaction;
use lmdb::WriteFlags;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::array::TryFromSliceError;
use std::fs;
use std::io::SeekFrom;
use std::path::Path;
use std::sync::Arc;

pub async fn init_vector_store(
    ctx: Arc<AppContext>,
    name: String,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    num_layers: u8,
    auto_config: bool,
) -> Result<Arc<VectorStore>, WaCustomError> {
    if name.is_empty() {
        return Err(WaCustomError::InvalidParams);
    }

    let collection_path: Arc<Path> = Path::new(&name).into();

    fs::create_dir_all(&collection_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;
    let quantization_metric = QuantizationMetric::Scalar;
    let storage_type = StorageType::UnsignedByte;

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), &name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vcs = Arc::new(vcs);

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
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(collection_path.join("prop.data"))
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    );

    let location = write_prop_to_file(&vec_hash, vector_list.clone(), &prop_file)?;

    let prop = ArcShift::new(PropState::Ready(Arc::new(NodeProp {
        id: vec_hash,
        value: vector_list.clone(),
        location: Some(location),
    })));

    let mut root: LazyItemRef<MergedNode> = LazyItemRef::new_invalid();
    let mut prev: LazyItemRef<MergedNode> = LazyItemRef::new_invalid();

    let mut nodes = Vec::new();
    for l in (0..=num_layers).rev() {
        let current_node = Arc::new(MergedNode {
            hnsw_level: HNSWLevel(l),
            prop: prop.clone(),
            neighbors: EagerLazyItemSet::new(),
            parent: prev.clone(),
            child: LazyItemRef::new_invalid(),
        });

        let lazy_node = LazyItem::from_arc(hash, 0, current_node.clone());
        let lazy_node_ref = LazyItemRef::from_arc(hash, 0, current_node.clone());

        if let Some(prev_node) = prev
            .item
            .get()
            .get_lazy_data()
            .and_then(|mut arc| arc.get().clone())
        {
            current_node.set_parent(prev.clone().item.get().clone());
            prev_node.set_child(lazy_node.clone());
        }
        prev = lazy_node_ref.clone();

        if l == 0 {
            root = lazy_node_ref.clone();
        }

        nodes.push(lazy_node_ref.clone());
    }

    let index_manager = Arc::new(BufferManagerFactory::new(
        collection_path.clone(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));
    let vec_raw_manager = Arc::new(BufferManagerFactory::new(
        collection_path.clone(),
        |root, ver| root.join(format!("{}.vec_raw", **ver)),
    ));
    // TODO: May be the value can be taken from config
    let cache = Arc::new(NodeRegistry::new(1000, index_manager.clone()));

    for item_ref in nodes.iter_mut() {
        persist_node_update_loc(index_manager.clone(), &mut item_ref.item)?;
    }

    index_manager.flush_all()?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels, num_layers));

    let vec_store = Arc::new(VectorStore::new(
        exec_queue_nodes,
        name.clone(),
        root,
        lp,
        size,
        prop_file,
        lmdb,
        ArcShift::new(hash),
        ArcShift::new(quantization_metric),
        ArcShift::new(DistanceMetric::Cosine),
        ArcShift::new(storage_type),
        vcs,
        num_layers,
        auto_config,
        cache,
        index_manager,
        vec_raw_manager,
    ));

    ctx.ain_env
        .vector_store_map
        .insert(&name, vec_store.clone())?;

    Ok(vec_store)
}

pub fn run_upload_in_transaction(
    vec_store: Arc<VectorStore>,
    transaction_id: Hash,
    vecs: Vec<(VectorIdValue, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let current_version = transaction_id;

    let bufman = vec_store
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
    let env = vec_store.lmdb.env.clone();
    let db = vec_store.lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    // Check if the previous version is unindexed, and continue from where we left.
    let prev_version = vec_store.get_current_version();
    let index_before_insertion = match txn.get(*db, &"next_embedding_offset") {
        Ok(bytes) => {
            let embedding_offset = EmbeddingOffset::deserialize(bytes)
                .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

            debug_assert_eq!(
                embedding_offset.version, prev_version,
                "Last unindexed embedding's version must be the previous version of the collection"
            );

            let prev_bufman = vec_store.vec_raw_manager.get(&prev_version)?;
            let cursor = prev_bufman.open_cursor()?;
            let prev_file_len = prev_bufman.seek_with_cursor(cursor, SeekFrom::End(0))? as u32;

            prev_file_len > embedding_offset.offset
        }
        Err(lmdb::Error::NotFound) => false,
        Err(e) => {
            return Err(WaCustomError::DatabaseError(e.to_string()));
        }
    };

    txn.abort();

    if index_before_insertion {
        index_embeddings(vec_store.clone(), ctx.config.upload_process_batch_size)?;
    }

    // Add next version
    let (current_version, _) = vec_store
        .vcs
        .add_next_version("main")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    vec_store.set_current_version(current_version);
    store_current_version(&vec_store.lmdb, current_version)?;

    // Update LMDB metadata
    let new_offset = EmbeddingOffset {
        version: current_version,
        offset: 0,
    };
    let new_offset_serialized = new_offset.serialize();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    txn.put(
        *db,
        &"next_embedding_offset",
        &new_offset_serialized,
        WriteFlags::empty(),
    )
    .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    txn.commit()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    // Insert vectors
    let bufman = vec_store.vec_raw_manager.get(&current_version)?;

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
    let db = vec_store.lmdb.db.clone();

    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let count_unindexed = txn
        .get(*db, &"count_unindexed")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))
        .and_then(|bytes| {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            Ok(u32::from_le_bytes(bytes))
        })?;

    txn.abort();

    if count_unindexed >= ctx.config.upload_threshold {
        index_embeddings(vec_store.clone(), ctx.config.upload_process_batch_size)?;
    }

    auto_commit_transaction(vec_store.clone())?;
    vec_store.vec_raw_manager.flush_all()?;
    vec_store.index_manager.flush_all()?;

    Ok(())
}

pub async fn ann_vector_query(
    vec_store: Arc<VectorStore>,
    query: Vec<f32>,
) -> Result<Option<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    let vector_store = vec_store.clone();
    let vec_hash = VectorId::Str("query".to_string());
    let root = &vector_store.root_vec;
    let vector_list = vector_store
        .quantization_metric
        .quantize(&query, *vector_store.storage_type.clone().get())?;

    let vec_emb = QuantizedVectorEmbedding {
        quantized_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let results = ann_search(
        vec_store.clone(),
        vec_emb,
        root.item.clone().get().clone(),
        HNSWLevel(vec_store.hnsw_params.clone().get().num_layers),
    )?;
    let output = remove_duplicates_and_filter(results);
    Ok(output)
}

pub async fn fetch_vector_neighbors(
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>> {
    let results = vector_fetch(vec_store.clone(), vector_id);
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
