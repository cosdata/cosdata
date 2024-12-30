use crate::app_context::AppContext;
use crate::indexes::inverted_index::InvertedIndex;
use crate::indexes::inverted_index_types::{RawSparseVectorEmbedding, SparsePair};
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::ProbCache;
use crate::models::collection::Collection;
use crate::models::common::*;
use crate::models::embedding_persist::EmbeddingOffset;
use crate::models::file_persist::write_node_to_file;
use crate::models::meta_persist::update_current_version;
use crate::models::types::*;
use crate::models::user::Statistics;
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
use std::sync::{Arc, RwLock};

/// creates a dense index for a collection
#[allow(unused_variables)]
pub async fn init_dense_index_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
    size: usize,
    _lower_bound: Option<f32>,
    _upper_bound: Option<f32>,
    num_layers: u8,
) -> Result<Arc<DenseIndex>, WaCustomError> {
    let collection_name = &collection.name;
    let collection_path: Arc<Path> = collection.get_path();
    let index_path = collection_path.join("dense_hnsw");
    // ensuring that the index has a separate directory created inside the collection directory
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let quantization_metric = QuantizationMetric::Scalar;
    let storage_type = StorageType::UnsignedByte;

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), &collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vcs = Arc::new(vcs);

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    //
    // what is the prop file exactly?
    // a file that stores the quantized version of raw vec
    let prop_file = Arc::new(RwLock::new(
        fs::OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(index_path.join("prop.data"))
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    ));

    let index_manager = Arc::new(BufferManagerFactory::new(
        index_path.clone().into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));
    let vec_raw_manager = Arc::new(BufferManagerFactory::new(index_path.into(), |root, ver| {
        root.join(format!("{}.vec_raw", **ver))
    }));

    // TODO: May be the value can be taken from config
    let cache = Arc::new(ProbCache::new(
        1000,
        index_manager.clone(),
        prop_file.clone(),
    ));

    let root = create_root_node(
        num_layers,
        &quantization_metric,
        storage_type,
        size,
        prop_file.clone(),
        hash,
        index_manager.clone(),
        ctx.config.hnsw.neighbors_count,
    )?;

    index_manager.flush_all()?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels, num_layers));

    let dense_index = Arc::new(DenseIndex::new(
        collection_name.clone(),
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
    let index_path = collection_path.join("sparse_inverted_index");
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), &collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vcs = Arc::new(vcs);
    //
    // what is the difference between vec_raw_manager and index_manager?
    // vec_raw_manager manages persisting raw embeddings/vectors on disk
    // index_manager manages persisting index data on disk
    let vec_raw_manager = Arc::new(BufferManagerFactory::new(
        index_path.clone().into(),
        |root, ver| root.join(format!("{}.vec_raw", **ver)),
    ));

    let index_manager = Arc::new(BufferManagerFactory::new(index_path.into(), |root, ver| {
        root.join(format!("{}.index", **ver))
    }));

    // TODO FIX THE FOLLOWING ISSUE
    // Has to call index.manager.get() here
    // to create the index file on disk upon index creation
    let _ = index_manager.get(&hash);
    index_manager.flush_all()?;

    let index = Arc::new(InvertedIndex::new(
        collection_name.clone(),
        collection.description.clone(),
        collection.sparse_vector.auto_create_index,
        collection.metadata_schema.clone(),
        collection.config.max_vectors,
        lmdb,
        ArcShift::new(hash),
        vcs,
        vec_raw_manager,
        index_manager,
    ));

    ctx.ain_env
        .collections_map
        .insert_inverted_index(&collection_name, index.clone())?;
    update_current_version(&index.lmdb, hash)?;
    Ok(index)
}

/// uploads a vector embedding within a transaction
pub fn run_upload_in_transaction(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    transaction: &DenseIndexTransaction,
    sample_points: Vec<(VectorId, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    transaction.increment_batch_count();
    let version = transaction.id;
    let version_number = transaction.version_number;

    index_embeddings_in_transaction(
        ctx.clone(),
        dense_index.clone(),
        version,
        version_number,
        transaction,
        sample_points,
    )?;

    transaction.start_serialization_round();

    Ok(())
}

/// uploads a sparse vector for inverted index
pub fn run_upload_sparse_vector(
    ctx: Arc<AppContext>,
    inverted_index: Arc<InvertedIndex>,
    vecs: Vec<(VectorId, Vec<SparsePair>)>,
) -> Result<(), WaCustomError> {
    // Adding next version
    //
    // does mean we are creating a new version with each vector?
    // no, but a new version with each transaction
    //
    // each version == a new file on the disk?
    // yes
    //
    // means we are storing each embedding in a new file (or list embeddings that come in one transaction)?
    // the list of embedding that come in one transaction are stored in a new file
    let (current_version, _) = inverted_index
        .vcs
        .add_next_version("main")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    inverted_index.set_current_version(current_version);
    update_current_version(&inverted_index.lmdb, current_version)?;

    // Insert vectors
    let bufman = inverted_index.vec_raw_manager.get(&current_version)?;

    vecs.into_par_iter()
        .map(|(id, vec)| {
            vec.iter().for_each(|vec| {
                // TODO (Question)
                // should InvertedIndexSparseAnnNodeBasic::insert params change
                // to accept vector_id as  u64 ??
                inverted_index.insert(vec.0, vec.1, id.0 as u32);
            });

            let vec_emb = RawSparseVectorEmbedding {
                raw_vec: Arc::new(vec),
                hash_vec: id,
            };

            // write embeddings to disk
            insert_sparse_embedding(
                bufman.clone(),
                inverted_index.clone(),
                &vec_emb,
                current_version,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    bufman.flush()?;

    inverted_index.vec_raw_manager.flush_all()?;
    inverted_index.index_manager.flush_all()?;

    Ok(())
}

/// uploads a vector embedding
pub fn run_upload(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    vecs: Vec<(VectorId, Vec<f32>)>,
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
            prev_bufman.close_cursor(cursor)?;

            prev_file_len > embedding_offset.offset
        }
        Err(lmdb::Error::NotFound) => false,
        Err(e) => {
            return Err(WaCustomError::DatabaseError(e.to_string()));
        }
    };

    txn.abort();
    let serialization_table = Arc::new(TSHashTable::new(16));
    let lazy_item_versions_table = Arc::new(TSHashTable::new(16));

    if index_before_insertion {
        index_embeddings(
            dense_index.clone(),
            ctx.config.upload_process_batch_size,
            serialization_table.clone(),
            lazy_item_versions_table.clone(),
            ctx.config.hnsw.neighbors_count,
        )?;
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
            let vec_emb = RawVectorEmbedding {
                raw_vec: Arc::new(vec),
                hash_vec: id,
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
        index_embeddings(
            dense_index.clone(),
            ctx.config.upload_process_batch_size,
            serialization_table.clone(),
            lazy_item_versions_table,
            ctx.config.hnsw.neighbors_count,
        )?;
    }

    let list = Arc::into_inner(serialization_table).unwrap().to_list();

    for (node, _) in list {
        write_node_to_file(&node, &dense_index.index_manager)?;
    }

    dense_index.vec_raw_manager.flush_all()?;
    dense_index.index_manager.flush_all()?;

    Ok(())
}

pub async fn ann_vector_query(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    query: Vec<f32>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let dense_index = dense_index.clone();
    let vec_hash = VectorId(u64::MAX - 1);
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
        dense_index.get_root_vec(),
        HNSWLevel(dense_index.hnsw_params.clone().get().num_layers),
        ctx.config.hnsw.neighbors_count,
    )?;
    let output = finalize_ann_results(dense_index, results, &query)?;
    Ok(output)
}

pub async fn fetch_vector_neighbors(
    dense_index: Arc<DenseIndex>,
    vector_id: VectorId,
) -> Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>> {
    let results = vector_fetch(dense_index.clone(), vector_id);
    return results.expect("Failed fetching vector neighbors");
}

#[allow(dead_code)]
fn calculate_statistics(_: &[i32]) -> Option<Statistics> {
    // Placeholder for calculating statistics
    None
}

#[allow(dead_code)]
fn vector_knn(_vs: &Vec<f32>, _vecs: &Vec<f32>) -> Vec<(i8, i8, String, f64)> {
    // Placeholder for vector KNN
    vec![]
}
