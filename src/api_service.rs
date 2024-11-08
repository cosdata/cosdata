use crate::app_context::AppContext;
use crate::indexes::inverted_index::InvertedIndex;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::NodeRegistry;
use crate::models::collection::Collection;
use crate::models::common::*;
use crate::models::lazy_load::*;
use crate::models::meta_persist::update_current_version;
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
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::array::TryFromSliceError;
use std::fs;
use std::io::SeekFrom;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

/// creates a dense index for a collection
pub async fn init_dense_index_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    num_layers: u8,
    auto_config: bool,
) -> Result<Arc<DenseIndex>, WaCustomError> {
    let collection_name = &collection.name;
    let collection_path: Arc<Path> = collection.get_path();

    let quantization_metric = QuantizationMetric::Scalar;
    let storage_type = StorageType::UnsignedByte;

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), &collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vcs = Arc::new(vcs);

    let exec_queue_nodes: ExecQueueUpdate = STM::new(Vec::new(), 16, true);

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    let prop_file = Arc::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(collection_path.join("prop.data"))
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    );

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

    let root = create_root_node(
        num_layers,
        &quantization_metric,
        storage_type,
        size,
        prop_file.clone(),
        hash,
        index_manager.clone(),
    )?;

    index_manager.flush_all()?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels, num_layers));

    let dense_index = Arc::new(DenseIndex::new(
        exec_queue_nodes,
        collection_name.clone(),
        LazyItemRef::from_lazy(root),
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
        .collections_map
        .insert(&collection_name, dense_index.clone())?;

    Ok(dense_index)
}

/// creates an inverted index for a collection
pub async fn init_inverted_index_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
) -> Result<Arc<InvertedIndex>, WaCustomError> {
    let collection_name = &collection.name;
    let collection_path: Arc<Path> = collection.get_path();

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), &collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vcs = Arc::new(vcs);

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    let prop_file = Arc::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(collection_path.join("prop.data"))
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    );

    let index = InvertedIndex::new(
        collection_name.clone(),
        collection.description.clone(),
        collection.sparse_vector.auto_create_index,
        collection.metadata_schema.clone(),
        collection.config.max_vectors,
        collection.config.replication_factor,
        prop_file,
        lmdb,
        ArcShift::new(hash),
        Arc::new(QuantizationMetric::Scalar),
        Arc::new(DistanceMetric::DotProduct),
        StorageType::UnsignedByte,
        vcs,
    );
    Ok(Arc::new(index))
}

/// uploads a vector embedding within a transaction
pub fn run_upload_in_transaction(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    transaction_id: Hash,
    vecs: Vec<(VectorIdValue, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let current_version = transaction_id;

    let bufman = dense_index
        .vec_raw_manager
        .get(&current_version)
        .map_err(|e| WaCustomError::BufIo(Arc::new(e)))?;

    let (tx, rx) = mpsc::channel();

    ctx.threadpool.install(|| {
        vecs.into_par_iter()
            .map(|(id, vec)| {
                let bufman = bufman.clone();
                let dense_index = dense_index.clone();
                let hash_vec = convert_value(id);
                let vec_emb = RawVectorEmbedding {
                    raw_vec: vec,
                    hash_vec,
                };
                tx.send(vec_emb.clone()).unwrap();
                insert_embedding(
                    bufman.clone(),
                    dense_index.clone(),
                    &vec_emb,
                    current_version,
                )
            })
            .collect::<Result<Vec<()>, WaCustomError>>()
    })?;

    drop(tx);

    index_embeddings_in_transaction(ctx.clone(), dense_index, transaction_id, rx)?;

    bufman.flush()?;

    Ok(())
}

/// uploads a vector embedding
pub fn run_upload(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    vecs: Vec<(VectorIdValue, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let env = dense_index.lmdb.env.clone();
    let db = dense_index.lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    // Check if the previous version is unindexed, and continue from where we left.
    let prev_version = dense_index.get_current_version();
    let index_before_insertion = match txn.get(*db, &"next_embedding_offset") {
        Ok(bytes) => {
            let embedding_offset = EmbeddingOffset::deserialize(bytes)
                .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

            debug_assert_eq!(
                embedding_offset.version, prev_version,
                "Last unindexed embedding's version must be the previous version of the collection"
            );

            let prev_bufman = dense_index.vec_raw_manager.get(&prev_version)?;
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
        index_embeddings(dense_index.clone(), ctx.config.upload_process_batch_size)?;
    }

    // Add next version
    let (current_version, _) = dense_index
        .vcs
        .add_next_version("main")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    dense_index.set_current_version(current_version);
    update_current_version(&dense_index.lmdb, current_version)?;

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
    let bufman = dense_index.vec_raw_manager.get(&current_version)?;

    vecs.into_par_iter()
        .map(|(id, vec)| {
            let hash_vec = convert_value(id);
            let vec_emb = RawVectorEmbedding {
                raw_vec: vec,
                hash_vec,
            };

            insert_embedding(
                bufman.clone(),
                dense_index.clone(),
                &vec_emb,
                current_version,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    bufman.flush()?;

    let env = dense_index.lmdb.env.clone();
    let db = dense_index.lmdb.db.clone();

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
        index_embeddings(dense_index.clone(), ctx.config.upload_process_batch_size)?;
    }

    auto_commit_transaction(dense_index.clone())?;
    dense_index.vec_raw_manager.flush_all()?;
    dense_index.index_manager.flush_all()?;

    Ok(())
}

pub async fn ann_vector_query(
    dense_index: Arc<DenseIndex>,
    query: Vec<f32>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let dense_index = dense_index.clone();
    let vec_hash = VectorId::Str("query".to_string());
    let root = &dense_index.root_vec;
    let vector_list = dense_index
        .quantization_metric
        .quantize(&query, *dense_index.storage_type.clone().get())?;

    let vec_emb = QuantizedVectorEmbedding {
        quantized_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let results = ann_search(
        dense_index.clone(),
        vec_emb,
        root.item.clone().get().clone(),
        HNSWLevel(dense_index.hnsw_params.clone().get().num_layers),
    )?;
    let output = remove_duplicates_and_filter(results);
    Ok(output)
}

pub async fn fetch_vector_neighbors(
    dense_index: Arc<DenseIndex>,
    vector_id: VectorId,
) -> Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>> {
    let results = vector_fetch(dense_index.clone(), vector_id);
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
