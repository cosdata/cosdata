use crate::app_context::AppContext;
use crate::indexes::inverted_index::{InvertedIndex, InvertedIndexTransaction};
use crate::indexes::inverted_index_types::{RawSparseVectorEmbedding, SparsePair};
use crate::macros::key;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::DenseIndexCache;
use crate::models::collection::Collection;
use crate::models::common::*;
use crate::models::embedding_persist::EmbeddingOffset;
use crate::models::meta_persist::{store_values_range, update_current_version};
use crate::models::prob_node::ProbNode;
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
    let index_path = collection_path.join("dense_hnsw");
    // ensuring that the index has a separate directory created inside the collection directory
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

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
        |root, ver: &Hash| root.join(format!("{}.index", **ver)),
        ProbNode::get_serialized_size(hnsw_params.neighbors_count) * 1000,
    ));

    let level_0_index_manager = Arc::new(BufferManagerFactory::new(
        index_path.clone().into(),
        |root, ver: &Hash| root.join(format!("{}_0.index", **ver)),
        ProbNode::get_serialized_size(hnsw_params.level_0_neighbors_count) * 1000,
    ));
    let vec_raw_manager = Arc::new(BufferManagerFactory::new(
        index_path.into(),
        |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
        8192,
    ));

    // TODO: May be the value can be taken from config
    let cache = Arc::new(DenseIndexCache::new(
        index_manager.clone(),
        level_0_index_manager.clone(),
        prop_file.clone(),
    ));
    if let Some(values_range) = values_range {
        store_values_range(&lmdb, values_range).map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to store values range to LMDB: {}", e))
        })?;
    }
    let values_range = values_range.unwrap_or((-1.0, 1.0));

    let root = create_root_node(
        &quantization_metric,
        storage_type,
        collection.dense_vector.dimension,
        prop_file.clone(),
        hash,
        &index_manager,
        &level_0_index_manager,
        values_range,
        &hnsw_params,
    )?;

    index_manager.flush_all()?;
    update_current_version(&lmdb, hash)?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 4.0;
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
        level_0_index_manager,
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
    quantization_bits: u8,
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
        |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
        8192,
    ));

    let index = Arc::new(InvertedIndex::new(
        collection_name.clone(),
        collection.description.clone(),
        index_path.clone().into(),
        collection.sparse_vector.auto_create_index,
        // @TODO(vineet): Fix the following after confirming that
        // metadata schema is not required for inverted indexes
        None,
        collection.config.max_vectors,
        lmdb,
        hash,
        vcs,
        vec_raw_manager,
        quantization_bits,
    )?);

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
    mut sample_points: Vec<(VectorId, Vec<f32>)>,
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

            let above_05_percent =
                (dense_index.sampling_data.above_05.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_04_percent =
                (dense_index.sampling_data.above_04.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_03_percent =
                (dense_index.sampling_data.above_03.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_02_percent =
                (dense_index.sampling_data.above_02.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_01_percent =
                (dense_index.sampling_data.above_01.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;

            let below_05_percent =
                (dense_index.sampling_data.below_05.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_04_percent =
                (dense_index.sampling_data.below_04.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_03_percent =
                (dense_index.sampling_data.below_03.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_02_percent =
                (dense_index.sampling_data.below_02.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_01_percent =
                (dense_index.sampling_data.below_01.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;

            println!("Above percentages:");
            println!("> 0.5: {:.2}%", above_05_percent);
            println!("> 0.4: {:.2}%", above_04_percent);
            println!("> 0.3: {:.2}%", above_03_percent);
            println!("> 0.2: {:.2}%", above_02_percent);
            println!("> 0.1: {:.2}%", above_01_percent);

            println!("Below percentages:");
            println!("< -0.5: {:.2}%", below_05_percent);
            println!("< -0.4: {:.2}%", below_04_percent);
            println!("< -0.3: {:.2}%", below_03_percent);
            println!("< -0.2: {:.2}%", below_02_percent);
            println!("< -0.1: {:.2}%", below_01_percent);

            let range_start = if below_01_percent <= ctx.config.indexing.clamp_margin_percent {
                -0.1
            } else if below_02_percent <= ctx.config.indexing.clamp_margin_percent {
                -0.2
            } else if below_03_percent <= ctx.config.indexing.clamp_margin_percent {
                -0.3
            } else if below_04_percent <= ctx.config.indexing.clamp_margin_percent {
                -0.4
            } else if below_05_percent <= ctx.config.indexing.clamp_margin_percent {
                -0.5
            } else {
                -1.0
            };

            let range_end = if above_01_percent <= ctx.config.indexing.clamp_margin_percent {
                0.1
            } else if above_02_percent <= ctx.config.indexing.clamp_margin_percent {
                0.2
            } else if above_03_percent <= ctx.config.indexing.clamp_margin_percent {
                0.3
            } else if above_04_percent <= ctx.config.indexing.clamp_margin_percent {
                0.4
            } else if above_05_percent <= ctx.config.indexing.clamp_margin_percent {
                0.5
            } else {
                1.0
            };

            let range = (range_start, range_end);
            println!("Range: {:?}", range);
            *dense_index.values_range.write().unwrap() = range;
            dense_index.is_configured.store(true, Ordering::Release);
            store_values_range(&dense_index.lmdb, range).map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to store values range to LMDB: {}", e))
            })?;
            sample_points = std::mem::replace(&mut *vectors, Vec::new());
            is_first_batch = true;
        } else {
            while !dense_index.is_configured.load(Ordering::Relaxed) {
                drop(dense_index.vectors.read().unwrap());
            }
        }
    }

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

    Ok(())
}

/// uploads a sparse vector for inverted index
pub fn run_upload_sparse_vector(
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
    let bufman = inverted_index.vec_raw_manager.get(current_version)?;

    vecs.into_iter()
        .map(|(id, vec)| {
            vec.iter()
                .map(|vec| {
                    // TODO (Question)
                    // should InvertedIndexSparseAnnNodeBasic::insert params change
                    // to accept vector_id as  u64 ??
                    inverted_index.insert(vec.0, vec.1, id.0 as u32, current_version)
                })
                .collect::<Result<(), _>>()?;

            // let vec_emb = RawSparseVectorEmbedding {
            //     raw_vec: Arc::new(vec),
            //     hash_vec: id,
            // };

            // // write embeddings to disk
            // insert_sparse_embedding(
            //     bufman.clone(),
            //     inverted_index.clone(),
            //     &vec_emb,
            //     current_version,
            // )
            Ok::<_, WaCustomError>(())
        })
        .collect::<Result<Vec<_>, _>>()?;
    bufman.flush()?;

    inverted_index.vec_raw_manager.flush_all()?;
    inverted_index.root.cache.dim_bufman.flush()?;
    inverted_index.root.cache.data_bufmans.flush_all()?;

    Ok(())
}

/// uploads a vector embedding within a transaction
pub fn run_upload_sparse_vectors_in_transaction(
    inverted_index: Arc<InvertedIndex>,
    transaction: &InvertedIndexTransaction,
    sample_points: Vec<(VectorId, Vec<SparsePair>)>,
) -> Result<(), WaCustomError> {
    sample_points
        .into_iter()
        .map(|(id, vec)| {
            vec.iter()
                .map(|vec| inverted_index.insert(vec.0, vec.1, id.0 as u32, transaction.id))
                .collect::<Result<(), _>>()?;
            let vec_emb = RawSparseVectorEmbedding {
                raw_vec: Arc::new(vec),
                hash_vec: id,
            };
            transaction.post_raw_embedding(vec_emb);
            Ok::<_, WaCustomError>(())
        })
        .collect::<Result<(), _>>()?;

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

            let prev_bufman = dense_index.vec_raw_manager.get(prev_version)?;
            let cursor = prev_bufman.open_cursor()?;
            let prev_file_len = prev_bufman.file_size() as u32;
            prev_bufman.close_cursor(cursor)?;

            prev_file_len > embedding_offset.offset
        }
        Err(lmdb::Error::NotFound) => false,
        Err(e) => {
            return Err(WaCustomError::DatabaseError(e.to_string()));
        }
    };

    txn.abort();
    let lazy_item_versions_table = Arc::new(TSHashTable::new(16));

    let (node_size, level_0_node_size) = {
        let hnsw_params = dense_index.hnsw_params.read().unwrap();
        let node_size = ProbNode::get_serialized_size(hnsw_params.neighbors_count);
        let level_0_node_size = ProbNode::get_serialized_size(hnsw_params.level_0_neighbors_count);
        (node_size as u32, level_0_node_size as u32)
    };
    let mut offset = 0;
    let mut level_0_offset = 0;

    if index_before_insertion {
        index_embeddings(
            &ctx.config,
            dense_index.clone(),
            ctx.config.upload_process_batch_size,
            lazy_item_versions_table.clone(),
            || {
                let ret = offset;
                offset += node_size;
                ret
            },
            || {
                let ret = level_0_offset;
                level_0_offset += level_0_node_size;
                ret
            },
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
    let next_embedding_offset_key = key!(m:next_embedding_offset);
    txn.put(
        *db,
        &next_embedding_offset_key,
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
    let mut offset = 0;
    let mut level_0_offset = 0;

    if count_unindexed >= ctx.config.upload_threshold {
        index_embeddings(
            &ctx.config,
            dense_index.clone(),
            ctx.config.upload_process_batch_size,
            lazy_item_versions_table,
            || {
                let ret = offset;
                offset += node_size;
                ret
            },
            || {
                let ret = level_0_offset;
                level_0_offset += level_0_node_size;
                ret
            },
        )?;
    }

    // for list in nodes_lists {
    //     for node in list.into_inner().unwrap() {
    //         write_node_to_file(
    //             node as *const _ as *mut _,
    //             &dense_index.index_manager,
    //             &dense_index.level_0_index_manager,
    //             current_version,
    //         )?;
    //     }
    // }

    dense_index.vec_raw_manager.flush_all()?;
    dense_index.index_manager.flush_all()?;
    dense_index.level_0_index_manager.flush_all()?;

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
