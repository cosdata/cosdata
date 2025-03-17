use crate::app_context::AppContext;
use crate::config_loader::Config;
use crate::config_loader::VectorsIndexingMode;
use crate::distance::cosine::CosineSimilarity;
use crate::distance::DistanceFunction;
use crate::indexes::hnsw::transaction::HNSWIndexTransaction;
use crate::indexes::hnsw::types::HNSWHyperParams;
use crate::indexes::hnsw::types::QuantizedDenseVectorEmbedding;
use crate::indexes::hnsw::types::RawDenseVectorEmbedding;
use crate::indexes::hnsw::HNSWIndex;
use crate::indexes::inverted::types::RawSparseVectorEmbedding;
use crate::indexes::inverted::InvertedIndex;
use crate::macros::key;
use crate::metadata::schema::MetadataDimensions;
use crate::metadata::MetadataFields;
use crate::metadata::MetadataSchema;
use crate::models::buffered_io::*;
use crate::models::common::*;
use crate::models::dot_product::dot_product_f32;
use crate::models::embedding_persist::*;
use crate::models::file_persist::*;
use crate::models::fixedset::PerformantFixedSet;
use crate::models::prob_lazy_load::lazy_item::FileIndex;
use crate::models::prob_lazy_load::lazy_item::ProbLazyItem;
use crate::models::prob_lazy_load::lazy_item_array::ProbLazyItemArray;
use crate::models::prob_node::ProbNode;
use crate::models::prob_node::SharedNode;
use crate::models::types::*;
use crate::models::versioning::Hash;
use crate::quantization::{Quantization, StorageType};
use crate::storage::Storage;
use lmdb::{Transaction, WriteFlags};
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::array::TryFromSliceError;
use std::collections::BinaryHeap;
use std::fs::File;
use std::ptr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::RwLock;

#[allow(clippy::too_many_arguments)]
pub fn create_root_node(
    quantization_metric: &QuantizationMetric,
    storage_type: StorageType,
    dim: usize,
    prop_file: &RwLock<File>,
    hash: Hash,
    index_manager: &BufferManagerFactory<Hash>,
    level_0_index_manager: &BufferManagerFactory<Hash>,
    values_range: (f32, f32),
    hnsw_params: &HNSWHyperParams,
    distance_metric: DistanceMetric,
) -> Result<SharedNode, WaCustomError> {
    let vec = (0..dim)
        .map(|_| {
            let mut rng = rand::thread_rng();

            let random_number: f32 = rng.gen_range(values_range.0..values_range.1);
            random_number
        })
        .collect::<Vec<f32>>();
    let vec_hash = VectorId(u64::MAX);

    let vector_list = Arc::new(quantization_metric.quantize(&vec, storage_type, values_range)?);

    let mut prop_file_guard = prop_file.write().unwrap();
    let location = write_prop_value_to_file(&vec_hash, vector_list.clone(), &mut prop_file_guard)?;
    drop(prop_file_guard);

    let prop_value = Arc::new(NodePropValue {
        id: vec_hash,
        vec: vector_list.clone(),
        location,
    });

    let prop_metadata = None;

    let mut root = ProbLazyItem::new(
        ProbNode::new(
            HNSWLevel(0),
            prop_value.clone(),
            prop_metadata.clone(),
            ptr::null_mut(),
            ptr::null_mut(),
            hnsw_params.level_0_neighbors_count,
            distance_metric,
        ),
        hash,
        0,
        true,
        FileOffset(0),
    );

    let mut nodes = Vec::new();
    nodes.push(root);

    let mut offset = 0;
    let node_size = ProbNode::get_serialized_size(hnsw_params.neighbors_count) as u32;

    for l in 1..=hnsw_params.num_layers {
        let current_node = ProbNode::new(
            HNSWLevel(l),
            prop_value.clone(),
            prop_metadata.clone(),
            ptr::null_mut(),
            root,
            hnsw_params.neighbors_count,
            distance_metric,
        );

        let lazy_node = ProbLazyItem::new(current_node, hash, 0, false, FileOffset(offset));
        offset += node_size;

        if let Some(prev_node) = unsafe { &*root }.get_lazy_data() {
            prev_node.set_parent(lazy_node);
        }
        root = lazy_node;

        nodes.push(lazy_node);
    }

    for item in nodes {
        write_node_to_file(item, index_manager, level_0_index_manager, hash)?;
    }

    Ok(root)
}

pub fn ann_search(
    config: &Config,
    hnsw_index: Arc<HNSWIndex>,
    vector_emb: QuantizedDenseVectorEmbedding,
    cur_entry: SharedNode,
    cur_level: HNSWLevel,
    hnsw_params: &HNSWHyperParams,
) -> Result<Vec<(SharedNode, MetricResult)>, WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = PerformantFixedSet::new(if cur_level.0 == 0 {
        hnsw_params.level_0_neighbors_count
    } else {
        hnsw_params.neighbors_count
    });
    skipm.insert(vector_emb.hash_vec.0 as u32);

    let cur_node = unsafe { &*cur_entry }.try_get_data(&hnsw_index.cache)?;

    let z = traverse_find_nearest(
        config,
        &hnsw_index,
        cur_entry,
        &fvec,
        &mut 0,
        &mut skipm,
        &hnsw_index.distance_metric.read().unwrap(),
        false,
        hnsw_params.ef_search,
    )?;

    let mut z = if z.is_empty() {
        let dist = hnsw_index
            .distance_metric
            .read()
            .unwrap()
            .calculate(&fvec, &cur_node.prop_value.vec)?;

        vec![(cur_entry, dist)]
    } else {
        z
    };

    if cur_level.0 != 0 {
        let results = ann_search(
            config,
            hnsw_index.clone(),
            vector_emb,
            unsafe { &*z[0].0 }
                .try_get_data(&hnsw_index.cache)?
                .get_child(),
            HNSWLevel(cur_level.0 - 1),
            hnsw_params,
        )?;

        z.extend(results);
    };

    Ok(z)
}

#[allow(clippy::type_complexity)]
pub fn vector_fetch(
    _hnsw_index: Arc<HNSWIndex>,
    _vector_id: VectorId,
) -> Result<Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>>, WaCustomError> {
    Ok(Vec::new())
}

pub fn finalize_ann_results(
    hnsw_index: Arc<HNSWIndex>,
    results: Vec<(SharedNode, MetricResult)>,
    query: &[f32],
    k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let filtered = remove_duplicates_and_filter(results, k, &hnsw_index.cache);
    let mut results = Vec::with_capacity(k.unwrap_or(filtered.len()));
    let mag_query = query.iter().map(|x| x * x).sum::<f32>().sqrt();

    for (id, _) in filtered {
        let raw = get_dense_embedding_by_id(hnsw_index.clone(), &id)?;
        let dp = dot_product_f32(query, &raw.raw_vec);
        let mag_raw = raw.raw_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cs = dp / (mag_query * mag_raw);
        results.push((id, MetricResult::CosineSimilarity(CosineSimilarity(cs))));
    }
    results.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
    if let Some(k) = k {
        results.truncate(k);
    }
    Ok(results)
}

/// Retrieves a raw embedding vector from the vector store by its ID.
///
/// This function performs the following steps to retrieve the embedding:
/// 1. Begins a read-write transaction with the LMDB environment.
/// 2. Retrieves the serialized offset of the embedding from the database using the provided `vector_id`.
/// 3. Deserializes the offset to obtain the embedding offset and version.
/// 4. Uses a `BufferManagerFactory` to create a buffer manager for the appropriate version.
/// 5. Reads the embedding from the buffer using the offset.
///
/// # Arguments
///
/// * `dense_index` - An `Arc`-wrapped `DenseIndex` instance, which contains the LMDB environment and database for embeddings.
/// * `vector_id` - The ID of the vector whose embedding is to be retrieved.
///
/// # Returns
///
/// * `Ok(RawVectorEmbedding)` - On success, returns the embedding associated with the given `vector_id`.
/// * `Err(WaCustomError)` - On failure, returns a custom error indicating the reason for the failure.
///
/// # Errors
///
/// This function may return an `Err` variant of `WaCustomError` in cases where:
/// * There is an error beginning the LMDB transaction (e.g., database access issues).
/// * The `vector_id` does not exist in the database, leading to a failure when retrieving the serialized offset.
/// * Deserialization of the embedding offset fails.
/// * There are issues with accessing or reading from the buffer manager.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use std::path::Path;
/// use my_crate::{DenseIndex, get_embedding_by_id, RawVectorEmbedding, WaCustomError, VectorId};
///
/// let hnsw_index = Arc::new(HNSWIndex::new());
/// let vector_id = VectorId::Int(42); // Example vector ID
/// match get_embedding_by_id(hnsw_index.clone(), vector_id) {
///     Ok(embedding) => println!("Embedding: {:?}", embedding),
///     Err(err) => eprintln!("Error retrieving embedding: {:?}", err),
/// }
/// ```
///
/// # Panics
///
/// This function does not panic.
///
/// # Notes
///
/// Ensure that the buffer manager and the database are correctly initialized and configured before calling this function.
/// The function assumes the existence of methods and types like `EmbeddingOffset::deserialize`, `BufferManagerFactory::new`, and `read_embedding` which should be implemented correctly.
pub fn get_dense_embedding_by_id(
    hnsw_index: Arc<HNSWIndex>,
    vector_id: &VectorId,
) -> Result<RawDenseVectorEmbedding, WaCustomError> {
    let env = hnsw_index.lmdb.env.clone();
    let db = hnsw_index.lmdb.db.clone();

    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let embedding_key = key!(e:vector_id);

    let offset_serialized = txn.get(*db, &embedding_key).map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to get serialized embedding offset: {}", e))
    })?;

    let embedding_offset = EmbeddingOffset::deserialize(offset_serialized)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    txn.abort();

    let offset = embedding_offset.offset;
    let current_version = embedding_offset.version;
    let bufman = hnsw_index.vec_raw_manager.get(current_version)?;
    let (embedding, _next) = read_embedding(bufman.clone(), offset)?;

    Ok(embedding)
}

pub fn get_sparse_embedding_by_id(
    inverted_index: Arc<InvertedIndex>,
    vector_id: &VectorId,
) -> Result<RawSparseVectorEmbedding, WaCustomError> {
    let env = inverted_index.lmdb.env.clone();
    let db = inverted_index.lmdb.db.clone();

    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let embedding_key = key!(e:vector_id);

    let offset_serialized = txn.get(*db, &embedding_key).map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to get serialized embedding offset: {}", e))
    })?;

    let embedding_offset = EmbeddingOffset::deserialize(offset_serialized)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    txn.abort();

    let offset = embedding_offset.offset;
    let current_version = embedding_offset.version;
    let bufman = inverted_index.vec_raw_manager.get(current_version)?;
    let (embedding, _next) = read_sparse_embedding(bufman.clone(), offset)?;

    Ok(embedding)
}

// fn auto_config_storage_type(dense_index: Arc<DenseIndex>, vectors: &[&[f32]]) {
//     let threshold = 0.0;
//     let iterations = 32;

//     let vec = concat_vectors(vectors);

//     // First iteration with k = 16
//     let initial_centroids_16 = generate_initial_centroids(&vec, 16);
//     let (_, counts_16) = kmeans(&vec, &initial_centroids_16, iterations);
//     let storage_type = if should_continue(&counts_16, threshold, 8) {
//         // Second iteration with k = 8
//         let initial_centroids_8 = generate_initial_centroids(&vec, 8);
//         let (_, counts_8) = kmeans(&vec, &initial_centroids_8, iterations);
//         if should_continue(&counts_8, threshold, 4) {
//             // Third iteration with k = 4
//             let initial_centroids_4 = generate_initial_centroids(&vec, 4);
//             let (_, counts_4) = kmeans(&vec, &initial_centroids_4, iterations);

//             if should_continue(&counts_4, threshold, 2) {
//                 StorageType::SubByte(1)
//             } else {
//                 StorageType::SubByte(2)
//             }
//         } else {
//             // StorageType::SubByte(3)
//             StorageType::UnsignedByte
//         }
//     } else {
//         StorageType::UnsignedByte
//     };

//     dense_index.storage_type.update_shared(storage_type);
// }

/// Inserts a sparse embedding into a buffer and updates the inverted index.
///
/// This function inserts a given sparse vector embedding into a buffer managed by
/// the `BufferManager`, while also updating an associated inverted index to reflect
/// the new embedding. The operation is versioned with a `current_version` to ensure
/// consistency across data insertions.
///
/// # Arguments
///
/// * `bufman` - A reference-counted (`Arc`) `BufferManager` that manages the buffer
///   where the sparse embedding will be inserted. The `BufferManager` handles memory
///   management and access to the underlying buffer.
/// * `dense_index` - A reference-counted (`Arc`) `InvertedIndex` that is updated
///   to reflect the insertion of the new sparse embedding. The `InvertedIndex`
///   allows for fast lookups and indexing of the embeddings.
/// * `emb` - A reference to the `RawSparseVectorEmbedding` that is to be inserted.
///   The embedding is assumed to be in a raw, sparse vector format, and it will be
///   added to both the buffer and the index.
/// * `current_version` - A `Hash` representing the current version of the data. This
///   is used to ensure versioning consistency when inserting the embedding into the
///   buffer and the inverted index.
///
/// # Returns
///
/// This function returns a `Result`:
/// - `Ok(())`: If the insertion of the embedding into the buffer and update of the
///   inverted index is successful, it returns `Ok` with an empty tuple.
/// - `Err(WaCustomError)`: If the operation fails, it returns a `WaCustomError` detailing
///   the error encountered, which could be caused by issues with the buffer, the index,
///   or version mismatch.
///
/// # Errors
///
/// - Returns a `WaCustomError` if any of the following occur:
///   - An error with writing the embedding to the buffer.
///   - A failure in updating the inverted index.
///   - A version conflict or inconsistency with the provided `current_version`.
#[allow(unused)]
pub fn insert_sparse_embedding(
    bufman: Arc<BufferManager>,
    inverted_index: Arc<InvertedIndex>,
    emb: &RawSparseVectorEmbedding,
    current_version: Hash,
) -> Result<(), WaCustomError> {
    let env = inverted_index.lmdb.env.clone();
    let db = inverted_index.lmdb.db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    // TODO (mohamed.eliwa) make `write_sparse_embedding` function generic over embedding
    // writing the embedding on disk
    let offset = write_sparse_embedding(bufman, emb)?;

    // generating embedding key
    let offset = EmbeddingOffset {
        version: current_version,
        offset,
    };
    let offset_serialized = offset.serialize();
    let embedding_key = key!(e:emb.hash_vec);

    // What is the difference between the following insertion and
    // the insertion that happens in `write_sparse_embedding`
    // aren't both of them persisted on the disk at the end?
    // the `write_sparse_embedding` function writes the embedding itself on the disk and returns the offset of the embedding,
    // while here we store the embedding key and its offset in the lmdb
    // so we can read the actual embedding later from the disk easily with one disk seek using the stored offset
    //
    // storing (key_embedding, offset_serialized) pair in in-memory database
    txn.put(*db, &embedding_key, &offset_serialized, WriteFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to put data: {}", e)))?;

    txn.commit().map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
    })?;

    Ok(())
}

pub fn insert_embedding(
    bufman: Arc<BufferManager>,
    hnsw_index: Arc<HNSWIndex>,
    emb: &RawDenseVectorEmbedding,
    current_version: Hash,
) -> Result<(), WaCustomError> {
    let env = hnsw_index.lmdb.env.clone();
    let db = hnsw_index.lmdb.db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let count_unindexed = match txn.get(*db, &"count_unindexed") {
        Ok(bytes) => {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            u32::from_le_bytes(bytes)
        }
        Err(lmdb::Error::NotFound) => 0,
        Err(err) => return Err(WaCustomError::DatabaseError(err.to_string())),
    };

    let offset = write_dense_embedding(bufman, emb)?;

    let offset = EmbeddingOffset {
        version: current_version,
        offset,
    };
    let offset_serialized = offset.serialize();

    let embedding_key = key!(e:emb.hash_vec);

    txn.put(*db, &embedding_key, &offset_serialized, WriteFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to put data: {}", e)))?;
    let count_unindexed_key = key!(m:count_unindexed);

    txn.put(
        *db,
        &count_unindexed_key,
        &(count_unindexed + 1).to_le_bytes(),
        WriteFlags::empty(),
    )
    .map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to update `count_unindexed`: {}", e))
    })?;

    txn.commit().map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
    })?;

    Ok(())
}

/// Intermediate representation of the embedding in a form that's
/// ready for indexing.
///
/// i.e. with quantization performed and property values and metadata
/// fields converted into appropriate types.
struct IndexableEmbedding {
    prop_value: Arc<NodePropValue>,
    prop_metadata: Option<Arc<NodePropMetadata>>,
    embedding: QuantizedDenseVectorEmbedding,
}

/// Computes and returns all metadata dimensions for the provided
/// metadata `fields` based on the metadata `schema`
///
/// Note that the result includes base_dimensions as well as
/// weighted_dimensions.
fn metadata_dimensions(
    schema: &MetadataSchema,
    fields: &MetadataFields
) -> Vec<(u8, MetadataDimensions)> {
    let mut result = vec![];
    // First add base dimensions
    result.push((0_u8, schema.base_dimensions()));
    // @TODO(vineet): Replace with appropriate weight
    let wdims = schema.weighted_dimensions(fields, 64000).unwrap();
    for (i, wd) in wdims.into_iter().enumerate() {
        result.push(((i+1) as u8, wd));
    }
    result
}

/// Converts raw embeddings into IndexableEmbedding i.e. ready to be
/// indexed - with quantization performed and property values and
/// metadata fields converted into appropriate types.
fn preprocess_embedding(
    app_env: &AppEnv,
    hnsw_index: &HNSWIndex,
    quantization_metric: &RwLock<QuantizationMetric>,
    raw_emb: &RawDenseVectorEmbedding
) -> Vec<IndexableEmbedding> {
    let quantization = quantization_metric.read().unwrap();
    let quantized_vec = Arc::new(
        quantization
            .quantize(
                &raw_emb.raw_vec,
                *hnsw_index.storage_type.read().unwrap(),
                *hnsw_index.values_range.read().unwrap(),
            )
            .expect("Quantization failed"),
    );

    // Write props to the prop file
    let mut prop_file_guard = hnsw_index.cache.prop_file.write().unwrap();
    let location = write_prop_value_to_file(
        &raw_emb.hash_vec,
        quantized_vec.clone(),
        &mut *prop_file_guard,
    ).expect("failed to write prop");
    drop(prop_file_guard);

    let prop_value = Arc::new(NodePropValue {
        id: raw_emb.hash_vec.clone(),
        vec: quantized_vec.clone(),
        location,
    });

    let embedding = QuantizedDenseVectorEmbedding {
        quantized_vec,
        hash_vec: raw_emb.hash_vec.clone(),
    };

    let coll = app_env
        .collections_map
        .get_collection(&hnsw_index.name)
        .expect("Couldn't get collection from ain_env");
    // @TODO(vineet): Remove unwrap
    let metadata_schema = coll.metadata_schema.as_ref().unwrap();

    match &raw_emb.raw_metadata {
        Some(metadata_fields) => {
            let wdims = metadata_dimensions(metadata_schema, metadata_fields);
            let mut result = Vec::with_capacity(wdims.len());
            for (i, wdim) in wdims {
                let vec = Arc::new(Metadata::from(wdim));
                // @TODO(vineet): Need more clarity about how a
                // metadata ids that are unique within a replica set
                // (v/s unique across all vectors in a collections)
                // will result in deterministic composite identifiers
                // for the ProbNode. Ref:
                // https://discord.com/channels/1250672673856421938/1335809236537446452/1348638117984206848
                let id = MetadataId(i + 1);

                // Write metadata to the same prop file
                // @TODO(vineet): Remove unwrap
                let mut prop_file_guard = hnsw_index.cache.prop_file.write().unwrap();
                let location = write_prop_metadata_to_file(
                    &id,
                    vec.clone(),
                    &mut *prop_file_guard
                ).unwrap();
                drop(prop_file_guard);

                let prop_metadata = NodePropMetadata {
                    id,
                    vec,
                    location,
                };
                let emb = IndexableEmbedding {
                    prop_value: prop_value.clone(),
                    prop_metadata: Some(Arc::new(prop_metadata)),
                    embedding: embedding.clone()
                };
                result.push(emb);
            }
            result
        },
        None => {
            let emb = IndexableEmbedding {
                prop_value,
                prop_metadata: None,
                embedding: embedding.clone(),
            };
            vec![emb]
        },
    }
}

pub fn index_embeddings(
    config: &Config,
    app_env: &AppEnv,
    hnsw_index: &HNSWIndex,
    upload_process_batch_size: usize,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    mut offset_fn: impl FnMut() -> u32,
    mut level_0_offset_fn: impl FnMut() -> u32,
) -> Result<(), WaCustomError> {
    let env = hnsw_index.lmdb.env.clone();
    let db = hnsw_index.lmdb.db.clone();

    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let mut count_indexed = match txn.get(*db, &"count_indexed") {
        Ok(bytes) => {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            u32::from_le_bytes(bytes)
        }
        Err(lmdb::Error::NotFound) => 0,
        Err(err) => return Err(WaCustomError::DatabaseError(err.to_string())),
    };
    let mut count_unindexed = match txn.get(*db, &"count_unindexed") {
        Ok(bytes) => {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            u32::from_le_bytes(bytes)
        }
        Err(lmdb::Error::NotFound) => 0,
        Err(err) => return Err(WaCustomError::DatabaseError(err.to_string())),
    };

    let embedding_offset = match txn.get(*db, &"next_embedding_offset") {
        Ok(bytes) => EmbeddingOffset::deserialize(bytes)
            .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?,
        Err(err) => return Err(WaCustomError::DatabaseError(err.to_string())),
    };
    let version = embedding_offset.version;
    let version_hash = hnsw_index
        .vcs
        .get_version_hash(&version, &txn)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?
        .expect("Current version hash not found");
    let version_number = *version_hash.version as u16;

    txn.abort();

    let hnsw_params_guard = hnsw_index.hnsw_params.read().unwrap();

    let mut index = |embeddings: Vec<RawDenseVectorEmbedding>, next_offset: u32| -> Result<(), WaCustomError> {
        let results: Vec<()> = embeddings
            .into_iter()
            .flat_map(|raw_emb| preprocess_embedding(app_env, &hnsw_index, &hnsw_index.quantization_metric, &raw_emb))
            .map(|emb| {
                let iv = get_max_insert_level(rand::random::<f32>().into(), &hnsw_index.levels_prob);

                let current_level = HNSWLevel(iv.try_into().unwrap());

                let mut current_entry = hnsw_index.get_root_vec();

                loop {
                    let data = unsafe { &*current_entry }
                        .try_get_data(&hnsw_index.cache)
                        .expect("Unable to load data");
                    if data.hnsw_level.0 > current_level.0 {
                        current_entry = data.get_child();
                    } else if data.hnsw_level == current_level {
                        break;
                    } else {
                        panic!("missing node");
                    }
                }

                index_embedding(
                    config,
                    hnsw_index,
                    ptr::null_mut(),
                    emb.embedding,
                    emb.prop_value,
                    emb.prop_metadata,
                    current_entry,
                    current_level,
                    version,
                    version_number,
                    lazy_item_versions_table.clone(),
                    &hnsw_params_guard,
                    2,
                    &mut offset_fn,
                    &mut level_0_offset_fn,
                    *hnsw_index.distance_metric.read().unwrap(),
                )
                .expect("index_embedding failed");
            })
            .collect();

        let batch_size = results.len() as u32;
        count_indexed += batch_size;
        count_unindexed -= batch_size;

        let mut txn = env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e))
        })?;

        let next_embedding_offset = EmbeddingOffset {
            version,
            offset: next_offset,
        };
        let next_embedding_offset_serialized = next_embedding_offset.serialize();
        let next_embedding_offset_key = key!(m:next_embedding_offset);
        let count_indexed_key = key!(m:count_indexed);
        let count_unindexed_key = key!(m:count_unindexed);

        txn.put(
            *db,
            &next_embedding_offset_key,
            &next_embedding_offset_serialized,
            WriteFlags::empty(),
        )
        .map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to update `next_embedding_offset`: {}", e))
        })?;

        txn.put(
            *db,
            &count_indexed_key,
            &count_indexed.to_le_bytes(),
            WriteFlags::empty(),
        )
        .map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to update `count_indexed`: {}", e))
        })?;

        txn.put(
            *db,
            &count_unindexed_key,
            &count_unindexed.to_le_bytes(),
            WriteFlags::empty(),
        )
        .map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to update `count_unindexed`: {}", e))
        })?;

        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(())
    };

    let bufman = hnsw_index.vec_raw_manager.get(version)?;

    let mut i = embedding_offset.offset;
    let cursor = bufman.open_cursor()?;
    let file_len = bufman.file_size() as u32;

    let mut embeddings = Vec::new();

    loop {
        if i == file_len {
            index(embeddings, i)?;
            bufman.close_cursor(cursor)?;
            break;
        }

        let (embedding, next) = read_embedding(bufman.clone(), i)?;
        embeddings.push(embedding);
        i = next;

        if embeddings.len() == upload_process_batch_size {
            index(embeddings, i)?;
            embeddings = Vec::new();
        }
    }

    Ok(())
}

pub fn index_embeddings_in_transaction(
    ctx: Arc<AppContext>,
    hnsw_index: &HNSWIndex,
    version: Hash,
    version_number: u16,
    transaction: &HNSWIndexTransaction,
    vecs: Vec<(VectorId, Vec<f32>, Option<MetadataFields>)>,
) -> Result<(), WaCustomError> {
    let hnsw_params_guard = hnsw_index.hnsw_params.read().unwrap();
    let index = |vecs: Vec<(VectorId, Vec<f32>, Option<MetadataFields>)>| {
        let embeddings = vecs
            .into_iter()
            .map(|vec| {
                let (id, values, metadata) = vec;
                let raw_emb = RawDenseVectorEmbedding {
                    hash_vec: id,
                    raw_vec: Arc::new(values),
                    raw_metadata: metadata,
                };
                transaction.post_raw_embedding(raw_emb.clone());
                raw_emb
            })
            .flat_map(|emb| preprocess_embedding(&ctx.ain_env, &hnsw_index, &hnsw_index.quantization_metric, &emb))
            .collect::<Vec<IndexableEmbedding>>();
        for emb in embeddings {
            let max_level = get_max_insert_level(rand::random::<f32>().into(), &hnsw_index.levels_prob);
            // Start from root at highest level
            let root_entry = hnsw_index.get_root_vec();
            let highest_level = HNSWLevel(hnsw_params_guard.num_layers);

            index_embedding(
                &ctx.config,
                hnsw_index,
                ptr::null_mut(),
                emb.embedding,
                emb.prop_value,
                emb.prop_metadata,
                root_entry,
                highest_level,
                version,
                version_number,
                transaction.lazy_item_versions_table.clone(),
                &hnsw_params_guard,
                max_level as u8, // Pass max_level to let index_embedding control node creation
                &mut || transaction.get_new_node_offset(),
                &mut || transaction.get_new_level_0_node_offset(),
                *hnsw_index.distance_metric.read().unwrap(),
            )?;
        }
        Ok::<_, WaCustomError>(())
    };

    match ctx.config.indexing.mode {
        VectorsIndexingMode::Sequential => {
            index(vecs)?;
        }
        VectorsIndexingMode::Batch { batch_size } => {
            vecs.into_par_iter()
                .chunks(batch_size)
                .map(index)
                .collect::<Result<(), _>>()?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn index_embedding(
    config: &Config,
    hnsw_index: &HNSWIndex,
    parent: SharedNode,
    vector_emb: QuantizedDenseVectorEmbedding,
    prop_value: Arc<NodePropValue>,
    prop_metadata: Option<Arc<NodePropMetadata>>,
    cur_entry: SharedNode,
    cur_level: HNSWLevel,
    version: Hash,
    version_number: u16,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    hnsw_params: &HNSWHyperParams,
    max_level: u8,
    offset_fn: &mut impl FnMut() -> u32,
    level_0_offset_fn: &mut impl FnMut() -> u32,
    distance_metric: DistanceMetric,
) -> Result<(), WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = PerformantFixedSet::new(if cur_level.0 == 0 {
        hnsw_params.level_0_neighbors_count
    } else {
        hnsw_params.neighbors_count
    });
    skipm.insert(vector_emb.hash_vec.0 as u32);

    let cur_node = unsafe { &*ProbLazyItem::get_latest_version(cur_entry, &hnsw_index.cache)?.0 }
        .try_get_data(&hnsw_index.cache)?;

    let z = traverse_find_nearest(
        config,
        hnsw_index,
        cur_entry,
        &fvec,
        &mut 0,
        &mut skipm,
        &distance_metric,
        true,
        hnsw_params.ef_construction,
    )?;

    let z = if z.is_empty() {
        let dist = hnsw_index
            .distance_metric
            .read()
            .unwrap()
            .calculate(&fvec, &cur_node.prop_value.vec)?;

        vec![(cur_entry, dist)]
    } else {
        z
    };
    if cur_level.0 > max_level {
        // Just traverse down without creating nodes
        if cur_level.0 != 0 {
            index_embedding(
                config,
                hnsw_index,
                ptr::null_mut(),
                vector_emb.clone(),
                prop_value.clone(),
                prop_metadata.clone(),
                unsafe { &*z[0].0 }
                    .try_get_data(&hnsw_index.cache)?
                    .get_child(),
                HNSWLevel(cur_level.0 - 1),
                version,
                version_number,
                lazy_item_versions_table.clone(),
                hnsw_params,
                max_level,
                offset_fn,
                level_0_offset_fn,
                distance_metric,
            )?;
        }
    } else {
        let (neighbors_count, is_level_0, offset) = if cur_level.0 == 0 {
            (
                hnsw_params.level_0_neighbors_count,
                true,
                level_0_offset_fn(),
            )
        } else {
            (hnsw_params.neighbors_count, false, offset_fn())
        };

        // Create node and edges at max_level and below
        let lazy_node = create_node(
            version,
            version_number,
            cur_level,
            prop_value.clone(),
            prop_metadata.clone(),
            parent,
            ptr::null_mut(),
            neighbors_count,
            is_level_0,
            offset,
            distance_metric,
        );

        let node = unsafe { &*lazy_node }.get_lazy_data().unwrap();

        if let Some(parent) = unsafe { parent.as_ref() } {
            parent
                .try_get_data(&hnsw_index.cache)
                .unwrap()
                .set_child(lazy_node);
        }

        if cur_level.0 != 0 {
            index_embedding(
                config,
                hnsw_index,
                lazy_node,
                vector_emb.clone(),
                prop_value.clone(),
                prop_metadata.clone(),
                unsafe { &*z[0].0 }
                    .try_get_data(&hnsw_index.cache)?
                    .get_child(),
                HNSWLevel(cur_level.0 - 1),
                version,
                version_number,
                lazy_item_versions_table.clone(),
                hnsw_params,
                max_level,
                offset_fn,
                level_0_offset_fn,
                distance_metric,
            )?;
        }

        let (is_level_0, offset_fn): (bool, &mut dyn FnMut() -> u32) = if cur_level.0 == 0 {
            (true, level_0_offset_fn)
        } else {
            (false, offset_fn)
        };

        create_node_edges(
            hnsw_index,
            lazy_node,
            node,
            z,
            version,
            version_number,
            lazy_item_versions_table,
            if cur_level.0 == 0 {
                hnsw_params.level_0_neighbors_count
            } else {
                hnsw_params.neighbors_count
            },
            is_level_0,
            offset_fn,
            distance_metric,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn create_node(
    version_id: Hash,
    version_number: u16,
    hnsw_level: HNSWLevel,
    prop_value: Arc<NodePropValue>,
    prop_metadata: Option<Arc<NodePropMetadata>>,
    parent: SharedNode,
    child: SharedNode,
    neighbors_count: usize,
    is_level_0: bool,
    offset: u32,
    distance_metric: DistanceMetric,
) -> SharedNode {
    let node = ProbNode::new(
        hnsw_level,
        prop_value,
        prop_metadata,
        parent,
        child,
        neighbors_count,
        distance_metric,
    );
    ProbLazyItem::new(
        node,
        version_id,
        version_number,
        is_level_0,
        FileOffset(offset),
    )
}

#[allow(clippy::too_many_arguments)]
fn get_or_create_version(
    hnsw_index: &HNSWIndex,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    lazy_item: SharedNode,
    version_id: Hash,
    version_number: u16,
    is_level_0: bool,
    offset_fn: &mut dyn FnMut() -> u32,
    distance_metric: DistanceMetric,
) -> Result<(SharedNode, bool), WaCustomError> {
    let node = unsafe { &*lazy_item }.try_get_data(&hnsw_index.cache)?;

    let new_version = lazy_item_versions_table.get_or_create_with_flag(
        (node.get_id().clone(), version_number, node.hnsw_level.0),
        || {
            let root_version = ProbLazyItem::get_root_version(lazy_item, &hnsw_index.cache)
                .expect("Couldn't get root version");

            if let Some(version) =
                ProbLazyItem::get_version(root_version, version_number, &hnsw_index.cache)
                    .expect("Deserialization failed")
            {
                return version;
            }

            let new_node = ProbNode::new_with_neighbors_and_versions_and_root_version(
                node.hnsw_level,
                node.prop_value.clone(),
                node.prop_metadata.clone(),
                node.clone_neighbors(),
                node.get_parent(),
                node.get_child(),
                ProbLazyItemArray::new(),
                root_version,
                distance_metric,
            );

            let version = ProbLazyItem::new(
                new_node,
                version_id,
                version_number,
                is_level_0,
                FileOffset(offset_fn()),
            );

            let updated_node = ProbLazyItem::add_version(root_version, version, &hnsw_index.cache)
                .expect("Failed to add version")
                .map_err(|_| "Failed to add version")
                .unwrap();

            write_node_to_file(
                updated_node,
                &hnsw_index.cache.bufmans,
                &hnsw_index.cache.level_0_bufmans,
                unsafe { &*updated_node }.get_current_version_id(),
            )
            .unwrap();

            version
        },
    );

    Ok(new_version)
}

#[allow(clippy::too_many_arguments)]
fn create_node_edges(
    hnsw_index: &HNSWIndex,
    lazy_node: SharedNode,
    node: &ProbNode,
    neighbors: Vec<(SharedNode, MetricResult)>,
    version: Hash,
    version_number: u16,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    max_edges: usize,
    is_level_0: bool,
    offset_fn: &mut dyn FnMut() -> u32,
    distance_metric: DistanceMetric,
) -> Result<(), WaCustomError> {
    let mut successful_edges = 0;
    let mut neighbors_to_update = Vec::new();

    lazy_item_versions_table.insert(
        (node.get_id().clone(), version_number, node.hnsw_level.0),
        lazy_node,
    );

    // First loop: Handle neighbor connections and collect updates
    for (neighbor, dist) in neighbors {
        if successful_edges >= max_edges {
            break;
        }

        let (new_lazy_neighbor, found_in_map) = get_or_create_version(
            hnsw_index,
            lazy_item_versions_table.clone(),
            neighbor,
            version,
            version_number,
            is_level_0,
            offset_fn,
            distance_metric,
        )?;

        let new_neighbor = unsafe { &*new_lazy_neighbor }.try_get_data(&hnsw_index.cache)?;
        let neighbor_inserted_idx = node.add_neighbor(
            new_neighbor.get_id().0 as u32,
            new_lazy_neighbor,
            dist,
            &hnsw_index.cache,
            distance_metric,
        );

        let neighbour_update_info = if let Some(neighbor_inserted_idx) = neighbor_inserted_idx {
            let node_inserted_idx = new_neighbor.add_neighbor(
                node.get_id().0 as u32,
                lazy_node,
                dist,
                &hnsw_index.cache,
                distance_metric,
            );
            if let Some(idx) = node_inserted_idx {
                successful_edges += 1;
                Some((idx, dist))
            } else {
                node.remove_neighbor(neighbor_inserted_idx, new_neighbor.get_id().0 as u32);
                None
            }
        } else {
            None
        };

        if !found_in_map {
            write_node_to_file(
                new_lazy_neighbor,
                &hnsw_index.cache.bufmans,
                &hnsw_index.cache.level_0_bufmans,
                version,
            )?;
        } else if let Some((idx, dist)) = neighbour_update_info {
            neighbors_to_update.push((new_lazy_neighbor, idx, dist));
        }
    }

    // Second loop: Batch process file operations for updated neighbors
    if !neighbors_to_update.is_empty() {
        let bufman = if is_level_0 {
            hnsw_index.cache.level_0_bufmans.get(version)?
        } else {
            hnsw_index.cache.bufmans.get(version)?
        };
        let cursor = bufman.open_cursor()?;
        let mut current_node_link = Vec::with_capacity(14);
        current_node_link.extend((node.get_id().0 as u32).to_le_bytes());

        let node = unsafe { &*lazy_node };

        let FileIndex {
            offset: node_offset,
            version_number: node_version_number,
            version_id: node_version_id,
        } = node.get_file_index();
        current_node_link.extend(node_offset.0.to_le_bytes());
        current_node_link.extend(node_version_number.to_le_bytes());
        current_node_link.extend(node_version_id.to_le_bytes());

        for (neighbor, neighbor_idx, dist) in neighbors_to_update {
            let offset = unsafe { &*neighbor }.get_file_index().offset;
            let mut current_node_link_with_dist = Vec::with_capacity(19);
            current_node_link_with_dist.clone_from(&current_node_link);
            let (tag, value) = dist.get_tag_and_value();
            current_node_link_with_dist.push(tag);
            current_node_link_with_dist.extend(value.to_le_bytes());

            let neighbor_offset = (offset.0 + 41) + neighbor_idx as u32 * 19;
            bufman.seek_with_cursor(cursor, neighbor_offset as u64)?;
            bufman.update_with_cursor(cursor, &current_node_link_with_dist)?;
        }

        bufman.close_cursor(cursor)?;
    }

    write_node_to_file(
        lazy_node,
        &hnsw_index.cache.bufmans,
        &hnsw_index.cache.level_0_bufmans,
        version,
    )?;

    hnsw_index.cache.insert_lazy_object(version, lazy_node);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn traverse_find_nearest(
    config: &Config,
    hnsw_index: &HNSWIndex,
    start_node: SharedNode,
    fvec: &Storage,
    nodes_visited: &mut u32,
    skipm: &mut PerformantFixedSet,
    distance_metric: &DistanceMetric,
    is_indexing: bool,
    ef: u32,
) -> Result<Vec<(SharedNode, MetricResult)>, WaCustomError> {
    let mut candidate_queue = BinaryHeap::new();
    let mut results = BinaryHeap::new();

    let (start_version, _) = ProbLazyItem::get_latest_version(start_node, &hnsw_index.cache)?;
    let start_data = unsafe { &*start_version }.try_get_data(&hnsw_index.cache)?;
    let start_dist = distance_metric.calculate(&fvec, &start_data.prop_value.vec)?;

    let start_id = start_data.get_id().0 as u32;
    skipm.insert(start_id);
    candidate_queue.push((start_dist, start_node));

    while let Some((dist, current_node)) = candidate_queue.pop() {
        if *nodes_visited >= ef {
            break;
        }
        *nodes_visited += 1;
        results.push((dist, current_node));

        let (current_version, _) =
            ProbLazyItem::get_latest_version(current_node, &hnsw_index.cache)?;
        let node = unsafe { &*current_version }.try_get_data(&hnsw_index.cache)?;

        for neighbor in node
            .get_neighbors_raw()
            .iter()
            .take(config.search.shortlist_size)
        {
            let (neighbor_id, neighbor_node) = unsafe {
                if let Some((id, node, _)) = neighbor.load(Ordering::Relaxed).as_ref() {
                    (*id, *node)
                } else {
                    continue;
                }
            };

            if !skipm.is_member(neighbor_id) {
                let neighbor_data = unsafe { &*neighbor_node }.try_get_data(&hnsw_index.cache)?;
                let dist = distance_metric.calculate(&fvec, &neighbor_data.prop_value.vec)?;
                skipm.insert(neighbor_id);
                candidate_queue.push((dist, neighbor_node));
            }
        }
    }

    let results = results
        .into_sorted_vec() // Convert BinaryHeap to a sorted Vec
        .into_iter() // Iterate over the sorted Vec
        .rev() // Reverse the order (to get descending order)
        .map(|(dist, node)| (node, dist)) // Map to the desired tuple format
        .take(if is_indexing { 64 } else { 100 }) // Limit the number of results
        .collect::<Vec<_>>(); // Collect into a Vec

    Ok(results)
}
