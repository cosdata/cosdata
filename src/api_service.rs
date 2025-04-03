use crate::app_context::AppContext;
use crate::indexes::hnsw::transaction::HNSWIndexTransaction;
use crate::indexes::hnsw::types::{
    HNSWHyperParams, QuantizedDenseVectorEmbedding, RawDenseVectorEmbedding,
};
use crate::indexes::hnsw::HNSWIndex;
use crate::indexes::inverted::transaction::InvertedIndexTransaction;
use crate::indexes::inverted::types::{RawSparseVectorEmbedding, SparsePair};
use crate::indexes::inverted::InvertedIndex;
use crate::indexes::inverted_idf::transaction::InvertedIndexIDFTransaction;
use crate::indexes::inverted_idf::InvertedIndexIDF;
use crate::macros::key;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::HNSWIndexCache;
use crate::models::collection::Collection;
use crate::models::common::*;
use crate::models::embedding_persist::{
    write_dense_embedding, write_sparse_embedding, EmbeddingOffset,
};
use crate::models::meta_persist::{
    store_values_range, store_values_upper_bound, update_current_version,
};
use crate::models::prob_node::ProbNode;
use crate::models::types::*;
use crate::models::versioning::{Hash, VersionControl};
use crate::quantization::{Quantization, StorageType};
use crate::vector_store::*;
use lmdb::{Transaction, WriteFlags};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::array::TryFromSliceError;
use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};

/// creates a dense index for a collection
#[allow(clippy::too_many_arguments)]
pub async fn init_hnsw_index_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
    values_range: Option<(f32, f32)>,
    hnsw_params: HNSWHyperParams,
    quantization_metric: QuantizationMetric,
    distance_metric: DistanceMetric,
    storage_type: StorageType,
    sample_threshold: usize,
    is_configured: bool,
) -> Result<Arc<HNSWIndex>, WaCustomError> {
    let collection_name = &collection.name;
    let collection_path: Arc<Path> = collection.get_path();
    let index_path = collection_path.join("dense_hnsw");
    // ensuring that the index has a separate directory created inside the collection directory
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    //
    // what is the prop file exactly?
    // a file that stores the quantized version of raw vec
    let prop_file = RwLock::new(
        fs::OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(index_path.join("prop.data"))
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    );

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
    let vec_raw_manager = BufferManagerFactory::new(
        index_path.into(),
        |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
        8192,
    );
    let distance_metric = Arc::new(RwLock::new(distance_metric));

    // TODO: May be the value can be taken from config
    let cache = HNSWIndexCache::new(
        index_manager.clone(),
        level_0_index_manager.clone(),
        prop_file,
        distance_metric.clone(),
    );
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
        &cache.prop_file,
        hash,
        &index_manager,
        &level_0_index_manager,
        values_range,
        &hnsw_params,
        *distance_metric.read().unwrap(),
    )?;

    index_manager.flush_all()?;
    update_current_version(&lmdb, hash)?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 4.0;
    let lp = generate_level_probs(factor_levels, hnsw_params.num_layers);

    let hnsw_index = Arc::new(HNSWIndex::new(
        collection_name.clone(),
        root,
        lp,
        collection.dense_vector.dimension,
        lmdb,
        hash,
        quantization_metric,
        distance_metric,
        storage_type,
        vcs,
        hnsw_params,
        cache,
        vec_raw_manager,
        values_range,
        sample_threshold,
        is_configured,
    ));

    ctx.ain_env
        .collections_map
        .insert_hnsw_index(collection_name, hnsw_index.clone())?;

    Ok(hnsw_index)
}

/// creates an inverted index for a collection
pub async fn init_inverted_index_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
    quantization_bits: u8,
    sample_threshold: usize,
    early_terminate_threshold: f32,
) -> Result<Arc<InvertedIndex>, WaCustomError> {
    let collection_name = &collection.name;
    let collection_path: Arc<Path> = collection.get_path();
    let index_path = collection_path.join("sparse_inverted_index");
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    //
    // what is the difference between vec_raw_manager and index_manager?
    // vec_raw_manager manages persisting raw embeddings/vectors on disk
    // index_manager manages persisting index data on disk
    let vec_raw_manager = BufferManagerFactory::new(
        index_path.clone().into(),
        |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
        8192,
    );

    let index = Arc::new(InvertedIndex::new(
        collection_name.clone(),
        collection.description.clone(),
        index_path.clone(),
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
        sample_threshold,
        early_terminate_threshold,
        ctx.config.inverted_index_data_file_parts,
    )?);

    ctx.ain_env
        .collections_map
        .insert_inverted_index(collection_name, index.clone())?;
    update_current_version(&index.lmdb, hash)?;
    Ok(index)
}

/// creates an inverted index for a collection
pub async fn init_inverted_index_idf_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
    quantization_bits: u8,
    early_terminate_threshold: f32,
) -> Result<Arc<InvertedIndexIDF>, WaCustomError> {
    let collection_name = &collection.name;
    let collection_path: Arc<Path> = collection.get_path();
    let index_path = collection_path.join("sparse_inverted_index_idf");
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vec_raw_manager = BufferManagerFactory::new(
        index_path.clone().into(),
        |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
        8192,
    );

    let index = Arc::new(InvertedIndexIDF::new(
        collection_name.clone(),
        collection.description.clone(),
        index_path.clone(),
        collection.sparse_vector.auto_create_index,
        collection.config.max_vectors,
        lmdb,
        hash,
        vcs,
        vec_raw_manager,
        quantization_bits,
        ctx.config.inverted_index_data_file_parts,
        early_terminate_threshold,
    )?);

    ctx.ain_env
        .collections_map
        .insert_inverted_index_idf(collection_name, index.clone())?;
    update_current_version(&index.lmdb, hash)?;
    Ok(index)
}

/// uploads a vector embedding within a transaction
pub fn run_upload_dense_vectors_in_transaction(
    ctx: Arc<AppContext>,
    hnsw_index: Arc<HNSWIndex>,
    transaction: &HNSWIndexTransaction,
    mut sample_points: Vec<(VectorId, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let version = transaction.id;
    let version_number = transaction.version_number;

    let mut is_first_batch = false;

    if !hnsw_index.is_configured.load(Ordering::Acquire) {
        let collected_count = hnsw_index
            .vectors_collected
            .fetch_add(sample_points.len(), Ordering::SeqCst);

        if collected_count < hnsw_index.sample_threshold {
            for (_, values) in &sample_points {
                for value in values {
                    let value = *value;

                    if value > 0.1 {
                        hnsw_index
                            .sampling_data
                            .above_01
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 0.2 {
                        hnsw_index
                            .sampling_data
                            .above_02
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 0.3 {
                        hnsw_index
                            .sampling_data
                            .above_03
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 0.4 {
                        hnsw_index
                            .sampling_data
                            .above_04
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 0.5 {
                        hnsw_index
                            .sampling_data
                            .above_05
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.1 {
                        hnsw_index
                            .sampling_data
                            .below_01
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.2 {
                        hnsw_index
                            .sampling_data
                            .below_02
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.3 {
                        hnsw_index
                            .sampling_data
                            .below_03
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.4 {
                        hnsw_index
                            .sampling_data
                            .below_04
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value < -0.5 {
                        hnsw_index
                            .sampling_data
                            .below_05
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            let mut vectors = hnsw_index.vectors.write().unwrap();
            vectors.extend(sample_points);
            if vectors.len() < hnsw_index.sample_threshold {
                return Ok(());
            }

            let dimension = vectors[0].1.len();
            let values_count = (dimension * vectors.len()) as f32;

            let above_05_percent =
                (hnsw_index.sampling_data.above_05.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_04_percent =
                (hnsw_index.sampling_data.above_04.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_03_percent =
                (hnsw_index.sampling_data.above_03.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_02_percent =
                (hnsw_index.sampling_data.above_02.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let above_01_percent =
                (hnsw_index.sampling_data.above_01.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;

            let below_05_percent =
                (hnsw_index.sampling_data.below_05.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_04_percent =
                (hnsw_index.sampling_data.below_04.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_03_percent =
                (hnsw_index.sampling_data.below_03.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_02_percent =
                (hnsw_index.sampling_data.below_02.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;
            let below_01_percent =
                (hnsw_index.sampling_data.below_01.load(Ordering::Relaxed) as f32 / values_count)
                    * 100.0;

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
            *hnsw_index.values_range.write().unwrap() = range;
            hnsw_index.is_configured.store(true, Ordering::Release);
            store_values_range(&hnsw_index.lmdb, range).map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to store values range to LMDB: {}", e))
            })?;
            sample_points = std::mem::take(&mut *vectors);
            is_first_batch = true;
        } else {
            while !hnsw_index.is_configured.load(Ordering::Relaxed) {
                drop(hnsw_index.vectors.read().unwrap());
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
                    &hnsw_index,
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
            &hnsw_index,
            version,
            version_number,
            transaction,
            sample_points,
        )?;
    }

    Ok(())
}

/// uploads a sparse vector for inverted index
pub fn run_upload_sparse_vectors(
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

    let vec_embs = vecs
        .into_iter()
        .map(|(id, vec)| {
            vec.iter().try_for_each(|vec| {
                // TODO (Question)
                // should InvertedIndexRoot::insert params change
                // to accept vector_id as  u64 ??
                inverted_index.insert(vec.0, vec.1, id.0 as u32, current_version)
            })?;

            let vec_emb = RawSparseVectorEmbedding {
                raw_vec: Arc::new(vec),
                hash_vec: id,
            };

            let offset = write_sparse_embedding(&bufman, &vec_emb)?;
            Ok::<_, WaCustomError>((offset, vec_emb))
        })
        .collect::<Result<Vec<_>, _>>()?;
    bufman.flush()?;

    let env = inverted_index.lmdb.env.clone();
    let db = inverted_index.lmdb.db.clone();
    let mut txn = env.begin_rw_txn()?;

    for (offset, vec_emb) in vec_embs {
        let offset = EmbeddingOffset {
            offset,
            version: current_version,
        };
        let offset_serialized = offset.serialize();
        let key = key!(e:vec_emb.hash_vec);
        txn.put(*db, &key, &offset_serialized, WriteFlags::empty())?;
    }

    txn.commit()?;

    inverted_index.vec_raw_manager.flush_all()?;
    inverted_index.root.cache.dim_bufman.flush()?;
    inverted_index.root.cache.data_bufmans.flush_all()?;

    Ok(())
}

/// uploads a sparse vector for inverted index
pub fn run_upload_sparse_idf_vectors(
    idf_inverted_index: Arc<InvertedIndexIDF>,
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
    let (current_version, _) = idf_inverted_index
        .vcs
        .add_next_version("main")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    idf_inverted_index.set_current_version(current_version);
    update_current_version(&idf_inverted_index.lmdb, current_version)?;

    let bufman = idf_inverted_index.vec_raw_manager.get(current_version)?;

    let vec_embs = vecs
        .into_iter()
        .map(|(id, terms)| {
            idf_inverted_index
                .root
                .total_documents_count
                .fetch_add(1, Ordering::Relaxed);
            terms.iter().try_for_each(|SparsePair(dim, value)| {
                idf_inverted_index.insert(*dim, *value, id.0 as u32, current_version)
            })?;

            let vec_emb = RawSparseVectorEmbedding {
                raw_vec: Arc::new(terms),
                hash_vec: id,
            };

            let offset = write_sparse_embedding(&bufman, &vec_emb)?;
            Ok::<_, WaCustomError>((offset, vec_emb))
        })
        .collect::<Result<Vec<_>, _>>()?;

    bufman.flush()?;

    let env = idf_inverted_index.lmdb.env.clone();
    let db = idf_inverted_index.lmdb.db.clone();
    let mut txn = env.begin_rw_txn()?;

    for (offset, vec_emb) in vec_embs {
        let offset = EmbeddingOffset {
            offset,
            version: current_version,
        };

        let offset_serialized = offset.serialize();
        let key = key!(e:vec_emb.hash_vec);
        txn.put(*db, &key, &offset_serialized, WriteFlags::empty())?;
    }

    txn.commit()?;

    idf_inverted_index.root.cache.dim_bufman.flush()?;
    idf_inverted_index.root.cache.data_bufmans.flush_all()?;

    Ok(())
}

/// uploads a vector embedding within a transaction
pub fn run_upload_sparse_vectors_in_transaction(
    ctx: Arc<AppContext>,
    inverted_index: Arc<InvertedIndex>,
    transaction: &InvertedIndexTransaction,
    mut sample_points: Vec<(VectorId, Vec<SparsePair>)>,
) -> Result<(), WaCustomError> {
    if !inverted_index.is_configured.load(Ordering::Acquire) {
        let collected_count = inverted_index
            .vectors_collected
            .fetch_add(sample_points.len(), Ordering::SeqCst);

        if collected_count < inverted_index.sample_threshold {
            for (_, pairs) in &sample_points {
                for pair in pairs {
                    let value = pair.1;

                    if value > 1.0 {
                        inverted_index
                            .sampling_data
                            .above_1
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 2.0 {
                        inverted_index
                            .sampling_data
                            .above_2
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 3.0 {
                        inverted_index
                            .sampling_data
                            .above_3
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 4.0 {
                        inverted_index
                            .sampling_data
                            .above_4
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 5.0 {
                        inverted_index
                            .sampling_data
                            .above_5
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 6.0 {
                        inverted_index
                            .sampling_data
                            .above_6
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 7.0 {
                        inverted_index
                            .sampling_data
                            .above_7
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 8.0 {
                        inverted_index
                            .sampling_data
                            .above_8
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    if value > 9.0 {
                        inverted_index
                            .sampling_data
                            .above_9
                            .fetch_add(1, Ordering::Relaxed);
                    }

                    inverted_index
                        .sampling_data
                        .values_collected
                        .fetch_add(1, Ordering::Relaxed);
                }
            }

            let mut vectors = inverted_index.vectors.write().unwrap();
            vectors.extend(sample_points);
            if vectors.len() < inverted_index.sample_threshold {
                return Ok(());
            }

            let values_count = inverted_index
                .sampling_data
                .values_collected
                .load(Ordering::Relaxed) as f32;

            let above_1_percent = (inverted_index.sampling_data.above_1.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;
            let above_2_percent = (inverted_index.sampling_data.above_2.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;
            let above_3_percent = (inverted_index.sampling_data.above_3.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;
            let above_4_percent = (inverted_index.sampling_data.above_4.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;
            let above_5_percent = (inverted_index.sampling_data.above_5.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;
            let above_6_percent = (inverted_index.sampling_data.above_6.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;
            let above_7_percent = (inverted_index.sampling_data.above_7.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;
            let above_8_percent = (inverted_index.sampling_data.above_8.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;
            let above_9_percent = (inverted_index.sampling_data.above_9.load(Ordering::Relaxed)
                as f32
                / values_count)
                * 100.0;

            let values_upper_bound = if above_1_percent <= ctx.config.indexing.clamp_margin_percent
            {
                1.0
            } else if above_2_percent <= ctx.config.indexing.clamp_margin_percent {
                2.0
            } else if above_3_percent <= ctx.config.indexing.clamp_margin_percent {
                3.0
            } else if above_4_percent <= ctx.config.indexing.clamp_margin_percent {
                4.0
            } else if above_5_percent <= ctx.config.indexing.clamp_margin_percent {
                5.0
            } else if above_6_percent <= ctx.config.indexing.clamp_margin_percent {
                6.0
            } else if above_7_percent <= ctx.config.indexing.clamp_margin_percent {
                7.0
            } else if above_8_percent <= ctx.config.indexing.clamp_margin_percent {
                8.0
            } else if above_9_percent <= ctx.config.indexing.clamp_margin_percent {
                9.0
            } else {
                10.0
            };

            *inverted_index.values_upper_bound.write().unwrap() = values_upper_bound;
            inverted_index.is_configured.store(true, Ordering::Release);
            store_values_upper_bound(&inverted_index.lmdb, values_upper_bound).map_err(|e| {
                WaCustomError::DatabaseError(format!(
                    "Failed to store values upper bound to LMDB: {}",
                    e
                ))
            })?;
            sample_points = std::mem::take(&mut *vectors);
        } else {
            while !inverted_index.is_configured.load(Ordering::Relaxed) {
                drop(inverted_index.vectors.read().unwrap());
            }
        }
    }

    sample_points.into_iter().try_for_each(|(id, vec)| {
        vec.iter()
            .try_for_each(|vec| inverted_index.insert(vec.0, vec.1, id.0 as u32, transaction.id))?;
        let vec_emb = RawSparseVectorEmbedding {
            raw_vec: Arc::new(vec),
            hash_vec: id,
        };
        transaction.post_raw_embedding(vec_emb);
        Ok::<_, WaCustomError>(())
    })?;

    Ok(())
}

/// uploads a vector embedding within a transaction
pub fn run_upload_sparse_idf_vectors_in_transaction(
    idf_inverted_index: Arc<InvertedIndexIDF>,
    transaction: &InvertedIndexIDFTransaction,
    sample_points: Vec<(VectorId, Vec<SparsePair>)>,
) -> Result<(), WaCustomError> {
    sample_points.into_par_iter().try_for_each(|(id, terms)| {
        idf_inverted_index
            .root
            .total_documents_count
            .fetch_add(1, Ordering::Relaxed);
        terms.iter().try_for_each(|SparsePair(dim, value)| {
            idf_inverted_index.insert(*dim, *value, id.0 as u32, transaction.id)
        })?;
        let vec_emb = RawSparseVectorEmbedding {
            raw_vec: Arc::new(terms),
            hash_vec: id,
        };
        transaction.post_raw_embedding(vec_emb);
        Ok::<_, WaCustomError>(())
    })?;

    Ok(())
}

/// uploads a vector embedding
pub fn run_upload_dense_vectors(
    ctx: Arc<AppContext>,
    hnsw_index: Arc<HNSWIndex>,
    vecs: Vec<(VectorId, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let lazy_item_versions_table = Arc::new(TSHashTable::new(16));

    // Add next version
    let (current_version, _) = hnsw_index.vcs.add_next_version("main").map_err(|e| {
        log::error!("Error adding next version for main branch in lmdb");
        WaCustomError::DatabaseError(e.to_string())
    })?;
    hnsw_index.set_current_version(current_version);
    update_current_version(&hnsw_index.lmdb, current_version)?;

    let bufman = hnsw_index.vec_raw_manager.get(current_version)?;

    // Insert vectors
    let vec_embs = vecs
        .into_par_iter()
        .map(|(id, vec)| {
            let vec_emb = RawDenseVectorEmbedding {
                raw_vec: Arc::new(vec),
                hash_vec: id,
            };

            Ok::<_, WaCustomError>((write_dense_embedding(&bufman, &vec_emb)?, vec_emb))
        })
        .collect::<Result<Vec<_>, _>>()?;
    bufman.flush()?;

    let env = hnsw_index.lmdb.env.clone();
    let db = hnsw_index.lmdb.db.clone();

    let mut txn = env.begin_rw_txn()?;

    let count_unindexed_key = key!(m:count_unindexed);

    let count_unindexed = match txn.get(*db, &count_unindexed_key) {
        Ok(bytes) => {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            Ok(u32::from_le_bytes(bytes))
        }
        Err(lmdb::Error::NotFound) => Ok(0),
        Err(e) => {
            log::error!("Error reading 'count_unindexed' from metadata in lmdb");
            Err(WaCustomError::DatabaseError(e.to_string()))
        }
    }?;

    let mut new_count_unindexed = count_unindexed + vec_embs.len() as u32;

    for (offset, vec_emb) in vec_embs {
        let offset = EmbeddingOffset {
            offset,
            version: current_version,
        };
        let offset_serialized = offset.serialize();
        let embedding_key = key!(e:vec_emb.hash_vec);
        txn.put(*db, &embedding_key, &offset_serialized, WriteFlags::empty())?;
    }

    // Indexing happens while the LMDB lock is held, making sure the vectors are not double indexed
    if new_count_unindexed >= ctx.config.upload_threshold {
        let last_indexed_version_key = key!(m:last_indexed_version);
        let last_indexed_version = match txn.get(*db, &last_indexed_version_key) {
            Ok(bytes) => {
                let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                    WaCustomError::DeserializationError(e.to_string())
                })?;
                Ok(Some(Hash::from(u32::from_le_bytes(bytes))))
            }
            Err(lmdb::Error::NotFound) => Ok(None),
            Err(e) => {
                log::error!("Error reading 'count_unindexed' from metadata in lmdb");
                Err(WaCustomError::DatabaseError(e.to_string()))
            }
        }?;
        index_embeddings(
            &ctx.config,
            &hnsw_index,
            ctx.config.upload_process_batch_size,
            lazy_item_versions_table,
            last_indexed_version,
        )?;
        txn.put(
            *db,
            &last_indexed_version_key,
            &current_version.to_le_bytes(),
            WriteFlags::empty(),
        )?;
        let count_indexed_key = key!(m:count_indexed);
        let count_indexed = match txn.get(*db, &count_indexed_key) {
            Ok(bytes) => {
                let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                    WaCustomError::DeserializationError(e.to_string())
                })?;
                Ok(u32::from_le_bytes(bytes))
            }
            Err(lmdb::Error::NotFound) => Ok(0),
            Err(e) => Err(e),
        }?;
        let new_count_indexed = count_indexed + new_count_unindexed;
        txn.put(
            *db,
            &count_indexed_key,
            &new_count_indexed.to_le_bytes(),
            WriteFlags::empty(),
        )?;
        new_count_unindexed = 0;
    }

    txn.put(
        *db,
        &count_unindexed_key,
        &new_count_unindexed.to_le_bytes(),
        WriteFlags::empty(),
    )?;

    txn.commit()?;

    hnsw_index.vec_raw_manager.flush_all()?;
    hnsw_index.cache.flush_all()?;

    Ok(())
}

pub async fn ann_vector_query(
    ctx: Arc<AppContext>,
    hnsw_index: Arc<HNSWIndex>,
    query: Vec<f32>,
    k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let vec_hash = VectorId(u64::MAX - 1);
    let vector_list = hnsw_index.quantization_metric.read().unwrap().quantize(
        &query,
        *hnsw_index.storage_type.read().unwrap(),
        *hnsw_index.values_range.read().unwrap(),
    )?;

    let vec_emb = QuantizedDenseVectorEmbedding {
        quantized_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let hnsw_params_guard = hnsw_index.hnsw_params.read().unwrap();

    let results = ann_search(
        &ctx.config,
        hnsw_index.clone(),
        vec_emb,
        hnsw_index.get_root_vec(),
        HNSWLevel(hnsw_params_guard.num_layers),
        &hnsw_params_guard,
    )?;
    drop(hnsw_params_guard);
    let output = finalize_ann_results(hnsw_index, results, &query, k)?;
    Ok(output)
}

pub async fn batch_ann_vector_query(
    ctx: Arc<AppContext>,
    hnsw_index: Arc<HNSWIndex>,
    queries: Vec<Vec<f32>>,
    k: Option<usize>,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    queries
        .into_par_iter()
        .map(|query| {
            let vec_hash = VectorId(u64::MAX - 1);
            let vector_list = hnsw_index.quantization_metric.read().unwrap().quantize(
                &query,
                *hnsw_index.storage_type.read().unwrap(),
                *hnsw_index.values_range.read().unwrap(),
            )?;

            let vec_emb = QuantizedDenseVectorEmbedding {
                quantized_vec: Arc::new(vector_list.clone()),
                hash_vec: vec_hash.clone(),
            };

            let hnsw_params = hnsw_index.hnsw_params.read().unwrap();
            let results = ann_search(
                &ctx.config,
                hnsw_index.clone(),
                vec_emb,
                hnsw_index.get_root_vec(),
                HNSWLevel(hnsw_params.num_layers),
                &hnsw_params,
            )?;
            let output = finalize_ann_results(hnsw_index.clone(), results, &query, k)?;
            Ok::<_, WaCustomError>(output)
        })
        .collect()
}

pub async fn fetch_vector_neighbors(
    hnsw_index: Arc<HNSWIndex>,
    vector_id: VectorId,
) -> Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>> {
    let results = vector_fetch(hnsw_index.clone(), vector_id);
    results.expect("Failed fetching vector neighbors")
}
