use crate::app_context::AppContext;
use crate::indexes::inverted_index::InvertedIndex;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::ProbCache;
use crate::models::collection::Collection;
use crate::models::common::*;
use crate::models::embedding_persist::EmbeddingOffset;
use crate::models::file_persist::write_node_to_file;
use crate::models::meta_persist::update_current_version;
use crate::models::types::*;
use crate::models::user::Statistics;
use crate::models::versioning::{Hash, VersionControl};
use crate::quantization::{Quantization, StorageType};
use crate::vector_store::*;
use arcshift::ArcShift;
use lmdb::Transaction;
use lmdb::WriteFlags;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::array::TryFromSliceError;
use std::fs;
use std::io::SeekFrom;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};

/// creates a dense index for a collection
#[allow(unused_variables)]
pub async fn init_dense_index_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
    values_range: Option<(f32, f32)>,
    hnsw_params: HNSWHyperParams,
    quantization_metric: QuantizationMetric,
    distance_metric: DistanceMetric,
    storage_type: StorageType,
    sample_threshold: usize,
    is_configured: bool,
) -> Result<Arc<DenseIndex>, WaCustomError> {
    let collection_name = &collection.name;
    let collection_path: Arc<Path> = collection.get_path();

    let values_range = values_range.unwrap_or((-1.0, 1.0));

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), &collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vcs = Arc::new(vcs);

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    let prop_file = Arc::new(RwLock::new(
        fs::OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(collection_path.join("prop.data"))
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    ));

    let index_manager = Arc::new(BufferManagerFactory::new(
        collection_path.clone(),
        |root, ver: &Hash| root.join(format!("{}.index", **ver)),
        ctx.config.flush_eagerness_factor,
    ));
    let vec_raw_manager = Arc::new(BufferManagerFactory::new(
        collection_path.clone(),
        |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
        ctx.config.flush_eagerness_factor,
    ));
    // TODO: May be the value can be taken from config
    let cache = Arc::new(ProbCache::new(
        1000,
        index_manager.clone(),
        prop_file.clone(),
    ));

    let root = create_root_node(
        &quantization_metric,
        storage_type,
        collection.dense_vector.dimension,
        prop_file.clone(),
        hash,
        index_manager.clone(),
        values_range,
        &hnsw_params,
    )?;

    index_manager.flush_all()?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels, hnsw_params.num_layers));

    let dense_index = Arc::new(DenseIndex::new(
        collection_name.clone(),
        root,
        lp,
        collection.dense_vector.dimension,
        prop_file,
        lmdb,
        ArcShift::new(hash),
        ArcShift::new(quantization_metric),
        ArcShift::new(distance_metric),
        ArcShift::new(storage_type),
        vcs,
        hnsw_params,
        cache,
        index_manager,
        vec_raw_manager,
        values_range,
        sample_threshold,
        is_configured,
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
            .read(true)
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
    update_current_version(&index.lmdb, hash)?;
    Ok(Arc::new(index))
}

/// uploads a vector embedding within a transaction
pub fn run_upload_in_transaction(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    transaction: &DenseIndexTransaction,
    mut sample_points: Vec<(u64, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let version = transaction.id;
    let version_number = transaction.version_number;

    let mut is_first_batch = false;

    if !dense_index.is_configured.load(Ordering::Acquire) {
        let collected_count = dense_index
            .vectors_collected
            .fetch_add(sample_points.len(), Ordering::SeqCst);

        if collected_count < dense_index.sample_threshold {
            for (_, values) in &sample_points {
                for value in values {
                    let value = *value;

                    if value > 0.1 {
                        dense_index
                            .sampling_data
                            .above_01
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 0.2 {
                        dense_index
                            .sampling_data
                            .above_02
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 0.3 {
                        dense_index
                            .sampling_data
                            .above_03
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 0.4 {
                        dense_index
                            .sampling_data
                            .above_04
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 0.5 {
                        dense_index
                            .sampling_data
                            .above_05
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.1 {
                        dense_index
                            .sampling_data
                            .below_01
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.2 {
                        dense_index
                            .sampling_data
                            .below_02
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.3 {
                        dense_index
                            .sampling_data
                            .below_03
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.4 {
                        dense_index
                            .sampling_data
                            .below_04
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.5 {
                        dense_index
                            .sampling_data
                            .below_05
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            let mut vectors = dense_index.vectors.write().unwrap();
            vectors.extend(sample_points);
            if vectors.len() < dense_index.sample_threshold {
                return Ok(());
            }

            let dimension = vectors[0].1.len();
            let values_count = (dimension * vectors.len()) as f32;

            let above_05_pecent =
                (dense_index.sampling_data.above_05.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_04_pecent =
                (dense_index.sampling_data.above_04.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_03_pecent =
                (dense_index.sampling_data.above_03.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_02_pecent =
                (dense_index.sampling_data.above_02.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_01_pecent =
                (dense_index.sampling_data.above_01.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;

            let below_05_pecent =
                (dense_index.sampling_data.below_05.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_04_pecent =
                (dense_index.sampling_data.below_04.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_03_pecent =
                (dense_index.sampling_data.below_03.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_02_pecent =
                (dense_index.sampling_data.below_02.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_01_pecent =
                (dense_index.sampling_data.below_01.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;

            println!("Above percentages:");
            println!("> 0.5: {:.2}%", above_05_pecent);
            println!("> 0.4: {:.2}%", above_04_pecent);
            println!("> 0.3: {:.2}%", above_03_pecent);
            println!("> 0.2: {:.2}%", above_02_pecent);
            println!("> 0.1: {:.2}%", above_01_pecent);

            println!("Below percentages:");
            println!("< -0.5: {:.2}%", below_05_pecent);
            println!("< -0.4: {:.2}%", below_04_pecent);
            println!("< -0.3: {:.2}%", below_03_pecent);
            println!("< -0.2: {:.2}%", below_02_pecent);
            println!("< -0.1: {:.2}%", below_01_pecent);

            let range_start = if below_01_pecent <= ctx.config.indexing.clamp_margin_percent {
                -0.1
            } else if below_02_pecent <= ctx.config.indexing.clamp_margin_percent {
                -0.2
            } else if below_03_pecent <= ctx.config.indexing.clamp_margin_percent {
                -0.3
            } else if below_04_pecent <= ctx.config.indexing.clamp_margin_percent {
                -0.4
            } else if below_05_pecent <= ctx.config.indexing.clamp_margin_percent {
                -0.5
            } else {
                -1.0
            };

            let range_end = if above_01_pecent <= ctx.config.indexing.clamp_margin_percent {
                0.1
            } else if above_02_pecent <= ctx.config.indexing.clamp_margin_percent {
                0.2
            } else if above_03_pecent <= ctx.config.indexing.clamp_margin_percent {
                0.3
            } else if above_04_pecent <= ctx.config.indexing.clamp_margin_percent {
                0.4
            } else if above_05_pecent <= ctx.config.indexing.clamp_margin_percent {
                0.5
            } else {
                1.0
            };

            let range = (range_start, range_end);
            println!("Range: {:?}", range);
            *dense_index.values_range.write().unwrap() = range;
            dense_index.is_configured.store(true, Ordering::Release);
            sample_points = std::mem::replace(&mut *vectors, Vec::new());
            is_first_batch = true;
        } else {
            while !dense_index.is_configured.load(Ordering::Relaxed) {
                drop(dense_index.vectors.read().unwrap());
            }
        }
    }
    transaction.increment_batch_count();

    if is_first_batch {
        sample_points
            .into_par_iter()
            .chunks(100)
            .map(|chunk| {
                index_embeddings_in_transaction(
                    ctx.clone(),
                    dense_index.clone(),
                    version,
                    version_number,
                    transaction,
                    chunk,
                )
            })
            .collect::<Result<(), WaCustomError>>()?;
    } else {
        index_embeddings_in_transaction(
            ctx.clone(),
            dense_index.clone(),
            version,
            version_number,
            transaction,
            sample_points,
        )?;
    }

    transaction.start_serialization_round();

    Ok(())
}

/// uploads a vector embedding
pub fn run_upload(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    vecs: Vec<(u64, Vec<f32>)>,
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

            let prev_bufman = dense_index.vec_raw_manager.get(prev_version)?;
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
            &ctx.config,
            dense_index.clone(),
            ctx.config.upload_process_batch_size,
            serialization_table.clone(),
            lazy_item_versions_table.clone(),
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
    let bufman = dense_index.vec_raw_manager.get(current_version)?;

    vecs.into_par_iter()
        .map(|(id, vec)| {
            let hash_vec = VectorId(id);
            let vec_emb = RawVectorEmbedding {
                raw_vec: Arc::new(vec),
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
        index_embeddings(
            &ctx.config,
            dense_index.clone(),
            ctx.config.upload_process_batch_size,
            serialization_table.clone(),
            lazy_item_versions_table,
        )?;
    }

    let list = Arc::into_inner(serialization_table).unwrap().to_list();

    for (node, _) in list {
        write_node_to_file(node, &dense_index.index_manager)?;
    }

    dense_index.vec_raw_manager.flush_all()?;
    dense_index.index_manager.flush_all()?;

    Ok(())
}

pub async fn ann_vector_query(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    query: Vec<f32>,
    k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let dense_index = dense_index.clone();
    let vec_hash = VectorId(u64::MAX - 1);
    let vector_list = dense_index.quantization_metric.quantize(
        &query,
        *dense_index.storage_type.clone().get(),
        *dense_index.values_range.read().unwrap(),
    )?;

    let vec_emb = QuantizedVectorEmbedding {
        quantized_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let hnsw_params = dense_index.hnsw_params.clone();
    let hnsw_params_guard = hnsw_params.read().unwrap();

    let results = ann_search(
        &ctx.config,
        dense_index.clone(),
        vec_emb,
        dense_index.get_root_vec(),
        HNSWLevel(hnsw_params_guard.num_layers),
        &*hnsw_params_guard,
    )?;
    let output = finalize_ann_results(dense_index, results, &query, k)?;
    Ok(output)
}

pub async fn batch_ann_vector_query(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
    queries: Vec<Vec<f32>>,
    k: Option<usize>,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    queries
        .into_par_iter()
        .map(|query| {
            let vec_hash = VectorId(u64::MAX - 1);
            let vector_list = dense_index.quantization_metric.quantize(
                &query,
                *dense_index.storage_type.clone().get(),
                *dense_index.values_range.read().unwrap(),
            )?;

            let vec_emb = QuantizedVectorEmbedding {
                quantized_vec: Arc::new(vector_list.clone()),
                hash_vec: vec_hash.clone(),
            };

            let hnsw_params = dense_index.hnsw_params.read().unwrap();
            let results = ann_search(
                &ctx.config,
                dense_index.clone(),
                vec_emb,
                dense_index.get_root_vec(),
                HNSWLevel(hnsw_params.num_layers),
                &hnsw_params,
            )?;
            let output = finalize_ann_results(dense_index.clone(), results, &query, k)?;
            Ok::<_, WaCustomError>(output)
        })
        .collect()
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
