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
use crate::metadata;
use crate::metadata::fields_to_dimensions;
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
    query_filter_dims: Option<&Vec<metadata::QueryFilterDimensions>>,
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

    let z = match query_filter_dims.clone() {
        Some(qf_dims) => {
            let mut z_candidates: Vec<(SharedNode, MetricResult)> = vec![];
            // @TODO: Can we compute the z_candidates in parallel?
            for qfd in qf_dims {
                let mdims = Metadata::from(qfd);
                let mut z_with_mdims = traverse_find_nearest(
                    config,
                    &hnsw_index,
                    cur_entry,
                    &fvec,
                    Some(&mdims),
                    &mut 0,
                    &mut skipm,
                    &hnsw_index.distance_metric.read().unwrap(),
                    false,
                    hnsw_params.ef_search,
                )?;
                // @NOTE: We're considering nearest neighbors computed
                // for all metadata dims. Here we're relying on
                // `traverse_find_nearest` to deduplicate the results
                // (thanks to the `skipm` argument)
                z_candidates.append(&mut z_with_mdims);
            }
            // Sort candidates by distance (asc)
            z_candidates.sort_by_key(|c| c.1);
            z_candidates.into_iter()
                .rev() // Reverse the order (to get descending order)
                .take(100) // Limit the number of results
                .collect::<Vec<_>>()
        },
        None => {
            traverse_find_nearest(
                config,
                &hnsw_index,
                cur_entry,
                &fvec,
                None,
                &mut 0,
                &mut skipm,
                &hnsw_index.distance_metric.read().unwrap(),
                false,
                hnsw_params.ef_search,
            )?
        },
    };

    let mut z = if z.is_empty() {
        let dist = match query_filter_dims.clone() {
            // In case of metadata filters in query, we calculate the
            // distances between the cur_node and all query filter
            // dimensions and take the minimum.
            //
            // @TODO: Not sure if this additional computation is
            // required because eventually the same node is being
            // returned. Also need to consider performing the
            // following in parallel.
            Some(qf_dims) => {
                let cur_node_metadata = cur_node.prop_metadata.clone().map(|pm| pm.vec.clone());
                let cur_node_data = VectorData {
                    quantized_vec: &cur_node.prop_value.vec,
                    metadata: cur_node_metadata.as_deref()
                };
                let mut dists = vec![];
                for qfd in qf_dims {
                    let fvec_metadata = Metadata::from(qfd);
                    let fvec_data = VectorData {
                        quantized_vec: &fvec,
                        metadata: Some(&fvec_metadata),
                    };
                    let d = hnsw_index
                        .distance_metric
                        .read()
                        .unwrap()
                        .calculate(&fvec_data, &cur_node_data)?;
                    dists.push(d)
                }
                dists.into_iter().min().unwrap()
            },
            None => {
                let fvec_data = VectorData::without_metadata(&fvec);
                let cur_node_data = VectorData::without_metadata(&cur_node.prop_value.vec);
                hnsw_index
                    .distance_metric
                    .read()
                    .unwrap()
                    .calculate(&fvec_data, &cur_node_data)?
            },
        };
        vec![(cur_entry, dist)]
    } else {
        z
    };

    if cur_level.0 != 0 {
        let results = ann_search(
            config,
            hnsw_index.clone(),
            vector_emb,
            query_filter_dims,
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

    for (orig_id, _, _) in filtered {
        let raw = get_dense_embedding_by_id(hnsw_index.clone(), &orig_id)?;
        let dp = dot_product_f32(query, &raw.raw_vec);
        let mag_raw = raw.raw_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cs = dp / (mag_query * mag_raw);
        results.push((orig_id, MetricResult::CosineSimilarity(CosineSimilarity(cs))));
    }
    results.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
    if let Some(k) = k {
        results.truncate(k);
    }
    Ok(results)
}

/// Retrieves a raw embedding vector from the vector store by its ID.
///
/// Note the id to be passed to this function is the user specified
/// identifier, which may not be the same as the id returned by
/// `ProbNode.get_id` function. This difference was introduced with
/// metadata filtering support. In case the collection supports
/// metadata filtering, then for an input vector, multiple replica
/// nodes may get created in the index, which have an internally
/// computed node id that's different from the user specified vector
/// id. The calling function is expected to always pass the user
/// specified id.
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

#[allow(clippy::too_many_arguments)]
pub fn index_embeddings_batch(
    config: &Config,
    app_env: &AppEnv,
    hnsw_index: &HNSWIndex,
    embeddings: Vec<RawDenseVectorEmbedding>,
    version: Hash,
    version_number: u16,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    hnsw_params: &HNSWHyperParams,
    offset_fn: &mut impl FnMut() -> u32,
    level_0_offset_fn: &mut impl FnMut() -> u32,
) -> Result<(), WaCustomError> {
    embeddings
        .into_iter()
        .flat_map(|raw_emb| preprocess_embedding(app_env, &hnsw_index, &hnsw_index.quantization_metric, &raw_emb))
        .map(|emb| {
            let max_level =
                get_max_insert_level(rand::random::<f32>().into(), &hnsw_index.levels_prob);
            let root_entry = hnsw_index.get_root_vec();
            let highest_level = HNSWLevel(hnsw_params.num_layers);

            index_embedding(
                config,
                hnsw_index,
                ptr::null_mut(),
                emb.embedding,
                emb.prop_value,
                emb.prop_metadata,
                root_entry,
                highest_level,
                version,
                version_number,
                lazy_item_versions_table.clone(),
                hnsw_params,
                max_level,
                offset_fn,
                level_0_offset_fn,
                *hnsw_index.distance_metric.read().unwrap(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
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

/// Computes "metadata replica sets" i.e. all metadata dimensions
/// along with an id for the provided metadata `fields` and based on
/// the metadata `schema`. If `fields` is None or an empty map, it
/// will return a vector with a single item i.e. the base dimensions.
fn metadata_replica_set(
    schema: &MetadataSchema,
    fields: Option<&MetadataFields>,
) -> Result<Vec<(MetadataId, Metadata)>, WaCustomError> {
    let dims = fields_to_dimensions(schema, fields)
        .map_err(WaCustomError::MetadataError)?;
    let replicas = dims.into_iter()
        .enumerate()
        .map(|(i, d)| {
            // @TODO(vineet): Need more clarity about how a metadata
            // ids that are unique within a replica set (v/s unique
            // across all vectors in a collections) will result in
            // deterministic composite identifiers for the
            // ProbNode. Ref:
            // https://discord.com/channels/1250672673856421938/1335809236537446452/1348638117984206848
            let mid = MetadataId(i as u8 + 1);
            let metadata = Metadata::from(d);
            (mid, metadata)
        })
        .collect();
    Ok(replicas)
}

/// Returns a vector of `NodePropMetadata` instances based on the
/// collection `schema` and `metadata_fields` as per the following
/// cases:
///
///   - If both `schema` and `metadata_fields` are not None, then it
///     computes the metadata dimensions and returns
///     `NodePropMetadata` instances based on those.
///
///   - If `metadata_fields` is None but `schema` is not None
///     (i.e. the collection supports metadata filtering but the
///     vector being inserted doesn't specify any fields), then a
///     single `NodePropMetadata` is returned corresponding to the
///     base dimensions.
///
///   - If schema is None, None is returned
///
/// Note that this function performs IO by writing metadata to the
/// prop_file
fn prop_metadata_replicas(
    schema: Option<&MetadataSchema>,
    metadata_fields: Option<&MetadataFields>,
    prop_file: &RwLock<File>,
) -> Result<Option<Vec<NodePropMetadata>>, WaCustomError> {
    if schema.is_none() {
        return Ok(None);
    }

    let replica_set = if metadata_fields.is_some() {
        Some(metadata_replica_set(schema.unwrap(), metadata_fields)?)
    } else {
        // If the collection supports metadata schema and
        // even if no metadata fields are specified with
        // the input vector, we create one replica with
        // base dimensions.
        match schema {
            Some(s) => {
                let mrset = metadata_replica_set(s, None)?;
                debug_assert_eq!(1, mrset.len());
                Some(mrset)
            },
            // Following is unreachable as the case of schema being
            // None has already been handled
            None => None,
        }
    };

    if let Some(replicas) = replica_set {
        let mut result = Vec::with_capacity(replicas.len());
        for (mid, m) in replicas {
            let mvalue = Arc::new(m);

            // Write metadata to the same prop file
            let mut prop_file_guard = prop_file.write()
                .map_err(|_| WaCustomError::LockError("Failed to acquire lock to write prop metadata".to_string()))?;
            let location = write_prop_metadata_to_file(
                &mid,
                mvalue.clone(),
                &mut *prop_file_guard
            )?;
            drop(prop_file_guard);

            let prop_metadata = NodePropMetadata {
                id: mid,
                vec: mvalue,
                location,
            };
            result.push(prop_metadata);
        }
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

/// Converts raw embeddings into `IndexableEmbedding` i.e. ready to be
/// indexed - with quantization performed and property values and
/// metadata fields converted into appropriate types.
///
/// If metadata filtering is supported for the collection, then one
/// input raw embedding may result in multiple `IndexableEmbedding`
/// instances.
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

    let metadata_replicas = prop_metadata_replicas(
        coll.metadata_schema.as_ref(),
        raw_emb.raw_metadata.as_ref(),
        &hnsw_index.cache.prop_file,
    ).unwrap();

    let mut embeddings: Vec<IndexableEmbedding> = vec![];

    match metadata_replicas {
        Some(replicas) => {
            for prop_metadata in replicas {
                let emb = IndexableEmbedding {
                    prop_value: prop_value.clone(),
                    prop_metadata: Some(Arc::new(prop_metadata)),
                    embedding: embedding.clone()
                };
                embeddings.push(emb);
            }
        },
        None => {
            let emb = IndexableEmbedding {
                prop_value,
                prop_metadata: None,
                embedding: embedding.clone(),
            };
            embeddings.push(emb);
        },
    }

    embeddings
}

pub fn index_embeddings(
    config: &Config,
    app_env: &AppEnv,
    hnsw_index: &HNSWIndex,
    upload_process_batch_size: usize,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    last_indexed_version: Option<Hash>,
) -> Result<(), WaCustomError> {
    let versions = if let Some(last_indexed_version) = last_indexed_version {
        hnsw_index
            .vcs
            .get_branch_versions_starting_from_exclusive("main", last_indexed_version)
    } else {
        hnsw_index.vcs.get_branch_versions("main")
    }?;

    let hnsw_params_guard = hnsw_index.hnsw_params.read().unwrap();

    let node_size = ProbNode::get_serialized_size(hnsw_params_guard.neighbors_count) as u32;
    let level_0_node_size =
        ProbNode::get_serialized_size(hnsw_params_guard.level_0_neighbors_count) as u32;

    for (version, version_info) in versions {
        let bufman = hnsw_index.vec_raw_manager.get(version)?;

        let mut i = 0;
        let cursor = bufman.open_cursor()?;
        let file_len = bufman.file_size() as u32;

        let mut embeddings = Vec::new();

        let mut offset = 0;
        let mut level_0_offset = 0;

        let mut offset_fn = || {
            let ret = offset;
            offset += node_size;
            ret
        };

        let mut level_0_offset_fn = || {
            let ret = level_0_offset;
            level_0_offset += level_0_node_size;
            ret
        };

        loop {
            if i == file_len {
                index_embeddings_batch(
                    config,
                    app_env,
                    hnsw_index,
                    embeddings,
                    version,
                    *version_info.version as u16,
                    lazy_item_versions_table.clone(),
                    &hnsw_params_guard,
                    &mut offset_fn,
                    &mut level_0_offset_fn,
                )?;
                bufman.close_cursor(cursor)?;
                break;
            }

            let (embedding, next) = read_embedding(bufman.clone(), i)?;
            embeddings.push(embedding);
            i = next;

            if embeddings.len() == upload_process_batch_size {
                index_embeddings_batch(
                    config,
                    app_env,
                    hnsw_index,
                    embeddings,
                    version,
                    *version_info.version as u16,
                    lazy_item_versions_table.clone(),
                    &hnsw_params_guard,
                    &mut offset_fn,
                    &mut level_0_offset_fn,
                )?;
                embeddings = Vec::new();
            }
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
                max_level, // Pass max_level to let index_embedding control node creation
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

    let mdims = prop_metadata.clone().map(|pm| pm.vec.clone());

    let z = traverse_find_nearest(
        config,
        hnsw_index,
        cur_entry,
        &fvec,
        mdims.as_deref(),
        &mut 0,
        &mut skipm,
        &distance_metric,
        true,
        hnsw_params.ef_construction,
    )?;

    let z = if z.is_empty() {
        let fvec_data = VectorData {
            quantized_vec: &fvec,
            metadata: mdims.as_deref(),
        };
        let cur_node_metadata = cur_node.prop_metadata.clone().map(|pm| pm.vec.clone());
        let cur_node_data = VectorData {
            quantized_vec: &cur_node.prop_value.vec,
            metadata: cur_node_metadata.as_deref()
        };
        let dist = hnsw_index
            .distance_metric
            .read()
            .unwrap()
            .calculate(&fvec_data, &cur_node_data)?;
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

            let neighbor_offset = (offset.0 + 49) + neighbor_idx as u32 * 19;
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
    mdims: Option<&Metadata>,
    nodes_visited: &mut u32,
    skipm: &mut PerformantFixedSet,
    distance_metric: &DistanceMetric,
    is_indexing: bool,
    ef: u32,
) -> Result<Vec<(SharedNode, MetricResult)>, WaCustomError> {
    let mut candidate_queue = BinaryHeap::new();
    let mut results = Vec::new();

    let (start_version, _) = ProbLazyItem::get_latest_version(start_node, &hnsw_index.cache)?;
    let start_data = unsafe { &*start_version }.try_get_data(&hnsw_index.cache)?;

    let fvec_data = VectorData {
        quantized_vec: fvec,
        metadata: mdims
    };

    let start_metadata = start_data.prop_metadata.clone().map(|pm| pm.vec.clone());
    let start_vec_data = VectorData {
        quantized_vec: &start_data.prop_value.vec,
        metadata: start_metadata.as_deref(),
    };
    let start_dist = distance_metric.calculate(&fvec_data, &start_vec_data)?;

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

        let _lock = node.lock_lowest_index();
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
                let neighbor_metadata = neighbor_data.prop_metadata.clone().map(|pm| pm.vec.clone());
                let neighbor_vec_data = VectorData {
                    quantized_vec: &neighbor_data.prop_value.vec,
                    metadata: neighbor_metadata.as_deref(),
                };
                let dist = distance_metric.calculate(&fvec_data, &neighbor_vec_data)?;
                skipm.insert(neighbor_id);
                candidate_queue.push((dist, neighbor_node));
            }
        }
    }

    let final_len = if is_indexing { 64 } else { 100 };
    if results.len() > final_len {
        results.select_nth_unstable_by(final_len, |(a, _), (b, _)| b.cmp(a));
        results.truncate(final_len);
    }

    results.sort_unstable_by(|(a, _), (b, _)| b.cmp(a));

    Ok(results.into_iter().map(|(sim, node)| (node, sim)).collect())
}
