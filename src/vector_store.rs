use crate::app_context::AppContext;
use crate::config_loader::Config;
use crate::config_loader::VectorsIndexingMode;
use crate::distance::cosine::CosineSimilarity;
use crate::distance::DistanceFunction;
use crate::indexes::inverted_index::InvertedIndex;
use crate::indexes::inverted_index_types::RawSparseVectorEmbedding;
use crate::macros::key;
use crate::models::buffered_io::*;
use crate::models::common::*;
use crate::models::dot_product::dot_product_f32;
use crate::models::embedding_persist::*;
use crate::models::file_persist::*;
use crate::models::fixedset::PerformantFixedSet;
use crate::models::lazy_load::FileIndex;
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

pub fn create_root_node(
    quantization_metric: &QuantizationMetric,
    storage_type: StorageType,
    dim: usize,
    prop_file: Arc<RwLock<File>>,
    hash: Hash,
    index_manager: &BufferManagerFactory<Hash>,
    level_0_index_manager: &BufferManagerFactory<Hash>,
    values_range: (f32, f32),
    hnsw_params: &HNSWHyperParams,
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
    let location = write_prop_to_file(&vec_hash, vector_list.clone(), &mut *prop_file_guard)?;
    drop(prop_file_guard);

    let prop = Arc::new(NodeProp {
        id: vec_hash,
        value: vector_list.clone(),
        location,
    });

    let mut root = ProbLazyItem::new(
        ProbNode::new(
            HNSWLevel(0),
            prop.clone(),
            ptr::null_mut(),
            ptr::null_mut(),
            hnsw_params.level_0_neighbors_count,
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
            prop.clone(),
            ptr::null_mut(),
            root,
            hnsw_params.neighbors_count,
        );

        let lazy_node = ProbLazyItem::new(current_node, hash, 0, false, FileOffset(offset));
        offset += node_size;

        if let Some(prev_node) = unsafe { &*root }.get_lazy_data() {
            prev_node.set_parent(lazy_node.clone());
        }
        root = lazy_node.clone();

        nodes.push(lazy_node);
    }

    for item in nodes {
        write_node_to_file(item, index_manager, level_0_index_manager, hash)?;
    }

    Ok(root)
}

pub fn ann_search(
    config: &Config,
    dense_index: Arc<DenseIndex>,
    vector_emb: QuantizedVectorEmbedding,
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

    let cur_node = unsafe { &*cur_entry }.try_get_data(&dense_index.cache)?;

    let z = traverse_find_nearest(
        config,
        &dense_index,
        cur_entry,
        &fvec,
        &mut 0,
        &mut skipm,
        false,
        hnsw_params.ef_search,
    )?;

    let mut z = if z.is_empty() {
        let dist = dense_index
            .distance_metric
            .calculate(&fvec, &cur_node.prop.value)?;

        vec![(cur_entry, dist)]
    } else {
        z
    };

    if cur_level.0 != 0 {
        let results = ann_search(
            config,
            dense_index.clone(),
            vector_emb,
            unsafe { &*z[0].0 }
                .try_get_data(&dense_index.cache)?
                .get_child(),
            HNSWLevel(cur_level.0 - 1),
            hnsw_params,
        )?;

        z.extend(results);
    };

    Ok(z)
}

pub fn vector_fetch(
    _dense_index: Arc<DenseIndex>,
    _vector_id: VectorId,
) -> Result<Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>>, WaCustomError> {
    Ok(Vec::new())
}

pub fn finalize_ann_results(
    dense_index: Arc<DenseIndex>,
    results: Vec<(SharedNode, MetricResult)>,
    query: &[f32],
    k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let filtered = remove_duplicates_and_filter(results, k, &dense_index.cache);
    let mut results = Vec::new();

    for (id, _) in filtered {
        let raw = get_embedding_by_id(dense_index.clone(), &id)?;
        let dp = dot_product_f32(query, &raw.raw_vec);
        let mag_query = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_raw = raw.raw_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cs = dp / (mag_query * mag_raw);
        results.push((id, MetricResult::CosineSimilarity(CosineSimilarity(cs))));
    }
    results.sort_unstable_by(|(_, a), (_, b)| {
        b.get_value()
            .partial_cmp(&a.get_value())
            .unwrap_or(std::cmp::Ordering::Greater)
    });
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
/// let dense_index = Arc::new(DenseIndex::new());
/// let vector_id = VectorId::Int(42); // Example vector ID
/// match get_embedding_by_id(dense_index.clone(), vector_id) {
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
pub fn get_embedding_by_id(
    dense_index: Arc<DenseIndex>,
    vector_id: &VectorId,
) -> Result<RawVectorEmbedding, WaCustomError> {
    let env = dense_index.lmdb.env.clone();
    let db = dense_index.lmdb.db.clone();

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
    let bufman = dense_index.vec_raw_manager.get(current_version)?;
    let (embedding, _next) = read_embedding(bufman.clone(), offset)?;

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
    dense_index: Arc<DenseIndex>,
    emb: &RawVectorEmbedding,
    current_version: Hash,
) -> Result<(), WaCustomError> {
    let env = dense_index.lmdb.env.clone();
    let db = dense_index.lmdb.db.clone();

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

    let offset = write_embedding(bufman, emb)?;

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

pub fn index_embeddings(
    config: &Config,
    dense_index: Arc<DenseIndex>,
    upload_process_batch_size: usize,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    mut offset_fn: impl FnMut() -> u32,
    mut level_0_offset_fn: impl FnMut() -> u32,
) -> Result<(), WaCustomError> {
    let env = dense_index.lmdb.env.clone();
    let db = dense_index.lmdb.db.clone();

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
    let version_hash = dense_index
        .vcs
        .get_version_hash(&version, &txn)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?
        .expect("Current version hash not found");
    let version_number = *version_hash.version as u16;

    txn.abort();

    let hnsw_params = dense_index.hnsw_params.clone();
    let hnsw_params_guard = hnsw_params.read().unwrap();

    let mut index = |embeddings: Vec<RawVectorEmbedding>,
                     next_offset: u32|
     -> Result<(), WaCustomError> {
        let mut quantization_arc = dense_index.quantization_metric.clone();
        let quantization = quantization_arc.get();

        let results: Vec<()> = embeddings
            .into_iter()
            .map(|raw_emb| {
                let lp = &dense_index.levels_prob;
                let iv = get_max_insert_level(rand::random::<f32>().into(), lp.clone());
                let quantized_vec = Arc::new(
                    quantization
                        .quantize(
                            &raw_emb.raw_vec,
                            dense_index.storage_type.clone().get().clone(),
                            *dense_index.values_range.read().unwrap(),
                        )
                        .expect("Quantization failed"),
                );
                let mut prop_file_guard = dense_index.prop_file.write().unwrap();
                let location = write_prop_to_file(
                    &raw_emb.hash_vec,
                    quantized_vec.clone(),
                    &mut *prop_file_guard,
                )
                .expect("failed to write prop");
                drop(prop_file_guard);
                let prop = Arc::new(NodeProp {
                    id: raw_emb.hash_vec.clone(),
                    value: quantized_vec.clone(),
                    location,
                });
                let embedding = QuantizedVectorEmbedding {
                    quantized_vec,
                    hash_vec: raw_emb.hash_vec,
                };

                let current_level = HNSWLevel(iv.try_into().unwrap());

                let mut current_entry = dense_index.get_root_vec();

                loop {
                    let data = unsafe { &*current_entry }
                        .try_get_data(&dense_index.cache)
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
                    dense_index.clone(),
                    ptr::null_mut(),
                    embedding,
                    prop,
                    current_entry,
                    current_level,
                    version,
                    version_number,
                    lazy_item_versions_table.clone(),
                    &hnsw_params_guard,
                    2,
                    &mut offset_fn,
                    &mut level_0_offset_fn,
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

    let bufman = dense_index.vec_raw_manager.get(version)?;

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
    dense_index: Arc<DenseIndex>,
    version: Hash,
    version_number: u16,
    transaction: &DenseIndexTransaction,
    vecs: Vec<(VectorId, Vec<f32>)>,
) -> Result<(), WaCustomError> {
    let quantization = &*dense_index.quantization_metric;
    let hnsw_params = dense_index.hnsw_params.clone();
    let hnsw_params_guard = hnsw_params.read().unwrap();
    let index = |vecs: Vec<(VectorId, Vec<f32>)>| {
        for (id, values) in vecs {
            let raw_emb = RawVectorEmbedding {
                hash_vec: id,
                raw_vec: Arc::new(values),
            };
            transaction.post_raw_embedding(raw_emb.clone());
            let lp = &dense_index.levels_prob;
            let max_level = get_max_insert_level(rand::random::<f32>().into(), lp.clone());
            let quantized_vec = Arc::new(quantization.quantize(
                &raw_emb.raw_vec,
                dense_index.storage_type.clone().get().clone(),
                *dense_index.values_range.read().unwrap(),
            )?);

            let mut prop_file_guard = dense_index.prop_file.write().unwrap();
            let location = write_prop_to_file(
                &raw_emb.hash_vec,
                quantized_vec.clone(),
                &mut *prop_file_guard,
            )?;
            drop(prop_file_guard);

            let prop = Arc::new(NodeProp {
                id: raw_emb.hash_vec.clone(),
                value: quantized_vec.clone(),
                location,
            });

            let embedding = QuantizedVectorEmbedding {
                quantized_vec,
                hash_vec: raw_emb.hash_vec,
            };

            // Start from root at highest level
            let root_entry = dense_index.get_root_vec();
            let highest_level = HNSWLevel(hnsw_params_guard.num_layers);

            index_embedding(
                &ctx.config,
                dense_index.clone(),
                ptr::null_mut(),
                embedding,
                prop,
                root_entry,
                highest_level,
                version,
                version_number,
                transaction.lazy_item_versions_table.clone(),
                &*hnsw_params_guard,
                max_level as u8, // Pass max_level to let index_embedding control node creation
                &mut || transaction.get_new_node_offset(),
                &mut || transaction.get_new_level_0_node_offset(),
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
                .collect::<Result<Vec<()>, _>>()?;
        }
    }

    Ok(())
}

#[allow(unused)]
pub fn index_sparse_embedding(
    inverted_index: Arc<InvertedIndex>,
    parent: Option<SharedNode>,
    vector_emb: QuantizedVectorEmbedding,
    version: Hash,
    version_number: u16,
    serialization_table: Arc<TSHashTable<SharedNode, ()>>,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16), SharedNode>>,
    neighbors_count: usize,
) -> Result<(), WaCustomError> {
    // let fvec = vector_emb.quantized_vec.clone();
    // let mut skipm = FxHashSet::default();
    // skipm.insert(vector_emb.hash_vec.clone());

    // TODO (mohamed.eliwa) implement this function
    Err(WaCustomError::DatabaseError(
        "index_sparse_embedding is not yet implemented".into(),
    ))
}

pub fn index_embedding(
    config: &Config,
    dense_index: Arc<DenseIndex>,
    parent: SharedNode,
    vector_emb: QuantizedVectorEmbedding,
    prop: Arc<NodeProp>,
    cur_entry: SharedNode,
    cur_level: HNSWLevel,
    version: Hash,
    version_number: u16,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    hnsw_params: &HNSWHyperParams,
    max_level: u8,
    offset_fn: &mut impl FnMut() -> u32,
    level_0_offset_fn: &mut impl FnMut() -> u32,
) -> Result<(), WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = PerformantFixedSet::new(if cur_level.0 == 0 {
        hnsw_params.level_0_neighbors_count
    } else {
        hnsw_params.neighbors_count
    });
    skipm.insert(vector_emb.hash_vec.0 as u32);

    let cur_node = unsafe { &*ProbLazyItem::get_latest_version(cur_entry, &dense_index.cache)?.0 }
        .try_get_data(&dense_index.cache)?;

    let z = traverse_find_nearest(
        config,
        &dense_index,
        cur_entry,
        &fvec,
        &mut 0,
        &mut skipm,
        true,
        hnsw_params.ef_construction,
    )?;

    let z = if z.is_empty() {
        let dist = dense_index
            .distance_metric
            .calculate(&fvec, &cur_node.prop.value)?;

        vec![(cur_entry, dist)]
    } else {
        z
    };
    if cur_level.0 > max_level {
        // Just traverse down without creating nodes
        if cur_level.0 != 0 {
            index_embedding(
                config,
                dense_index.clone(),
                ptr::null_mut(),
                vector_emb.clone(),
                prop.clone(),
                unsafe { &*z[0].0 }
                    .try_get_data(&dense_index.cache)?
                    .get_child(),
                HNSWLevel(cur_level.0 - 1),
                version,
                version_number,
                lazy_item_versions_table.clone(),
                hnsw_params,
                max_level,
                offset_fn,
                level_0_offset_fn,
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
            prop.clone(),
            parent,
            ptr::null_mut(),
            neighbors_count,
            is_level_0,
            offset,
        );

        let node = unsafe { &*lazy_node }.get_lazy_data().unwrap();

        if let Some(parent) = unsafe { parent.as_ref() } {
            parent
                .try_get_data(&dense_index.cache)
                .unwrap()
                .set_child(lazy_node.clone());
        }

        if cur_level.0 != 0 {
            index_embedding(
                config,
                dense_index.clone(),
                lazy_node,
                vector_emb.clone(),
                prop.clone(),
                unsafe { &*z[0].0 }
                    .try_get_data(&dense_index.cache)?
                    .get_child(),
                HNSWLevel(cur_level.0 - 1),
                version,
                version_number,
                lazy_item_versions_table.clone(),
                hnsw_params,
                max_level,
                offset_fn,
                level_0_offset_fn,
            )?;
        }

        let (is_level_0, offset_fn): (bool, &mut dyn FnMut() -> u32) = if cur_level.0 == 0 {
            (true, level_0_offset_fn)
        } else {
            (false, offset_fn)
        };

        create_node_edges(
            dense_index.clone(),
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
        )?;
    }

    Ok(())
}

fn create_node(
    version_id: Hash,
    version_number: u16,
    hnsw_level: HNSWLevel,
    prop: Arc<NodeProp>,
    parent: SharedNode,
    child: SharedNode,
    neighbors_count: usize,
    is_level_0: bool,
    offset: u32,
) -> SharedNode {
    let node = ProbNode::new(hnsw_level, prop, parent, child, neighbors_count);
    ProbLazyItem::new(
        node,
        version_id,
        version_number,
        is_level_0,
        FileOffset(offset),
    )
}

fn get_or_create_version(
    dense_index: Arc<DenseIndex>,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    lazy_item: SharedNode,
    version_id: Hash,
    version_number: u16,
    is_level_0: bool,
    offset_fn: &mut dyn FnMut() -> u32,
) -> Result<(SharedNode, bool), WaCustomError> {
    let node = unsafe { &*lazy_item }.try_get_data(&dense_index.cache)?;

    let new_version = lazy_item_versions_table.get_or_create_with_flag(
        (node.get_id().clone(), version_number, node.hnsw_level.0),
        || {
            let root_version = ProbLazyItem::get_root_version(lazy_item, &dense_index.cache)
                .expect("Couldn't get root version");

            if let Some(version) =
                ProbLazyItem::get_version(root_version, version_number, &dense_index.cache)
                    .expect("Deserialization failed")
            {
                return version;
            }

            let new_node = ProbNode::new_with_neighbors_and_versions_and_root_version(
                node.hnsw_level,
                node.prop.clone(),
                node.clone_neighbors(),
                node.get_parent(),
                node.get_child(),
                ProbLazyItemArray::new(),
                root_version,
            );

            let version = ProbLazyItem::new(
                new_node,
                version_id,
                version_number,
                is_level_0,
                FileOffset(offset_fn()),
            );

            let updated_node = ProbLazyItem::add_version(root_version, version, &dense_index.cache)
                .expect("Failed to add version")
                .map_err(|_| "Failed to add version")
                .unwrap();

            write_node_to_file(
                updated_node,
                &dense_index.index_manager,
                &dense_index.level_0_index_manager,
                unsafe { &*updated_node }.get_current_version_id(),
            )
            .unwrap();

            version
        },
    );

    Ok(new_version)
}

fn create_node_edges(
    dense_index: Arc<DenseIndex>,
    lazy_node: SharedNode,
    node: &ProbNode,
    neighbors: Vec<(SharedNode, MetricResult)>,
    version: Hash,
    version_number: u16,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    max_edges: usize,
    is_level_0: bool,
    offset_fn: &mut dyn FnMut() -> u32,
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
            dense_index.clone(),
            lazy_item_versions_table.clone(),
            neighbor,
            version,
            version_number,
            is_level_0,
            offset_fn,
        )?;

        let new_neighbor = unsafe { &*new_lazy_neighbor }.try_get_data(&dense_index.cache)?;
        let neighbor_inserted_idx = node.add_neighbor(
            new_neighbor.get_id().0 as u32,
            new_lazy_neighbor,
            dist,
            &dense_index.cache,
        );

        let neighbour_update_info = if let Some(neighbor_inserted_idx) = neighbor_inserted_idx {
            let node_inserted_idx = new_neighbor.add_neighbor(
                node.get_id().0 as u32,
                lazy_node,
                dist,
                &dense_index.cache,
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
                new_lazy_neighbor.clone(),
                &dense_index.index_manager,
                &dense_index.level_0_index_manager,
                version,
            )?;
        } else if let Some((idx, dist)) = neighbour_update_info {
            neighbors_to_update.push((new_lazy_neighbor, idx, dist));
        }
    }

    // Second loop: Batch process file operations for updated neighbors
    if !neighbors_to_update.is_empty() {
        let bufman = if is_level_0 {
            dense_index.level_0_index_manager.get(version)?
        } else {
            dense_index.index_manager.get(version)?
        };
        let cursor = bufman.open_cursor()?;
        let mut current_node_link = Vec::with_capacity(14);
        current_node_link.extend((node.get_id().0 as u32).to_le_bytes());

        let node = unsafe { &*lazy_node };

        let (node_offset, node_version_number, node_version_id) = match node.get_file_index() {
            FileIndex::Valid {
                offset,
                version_number,
                version_id,
            } => (offset.0, version_number, version_id),
            _ => unreachable!(),
        };
        current_node_link.extend(node_offset.to_le_bytes());
        current_node_link.extend(node_version_number.to_le_bytes());
        current_node_link.extend(node_version_id.to_le_bytes());

        for (neighbor, neighbor_idx, dist) in neighbors_to_update {
            let offset = unsafe { &*neighbor }.get_file_index().get_offset().unwrap();
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
        &dense_index.index_manager,
        &dense_index.level_0_index_manager,
        version,
    )?;

    Ok(())
}

fn traverse_find_nearest(
    config: &Config,
    dense_index: &DenseIndex,
    start_node: SharedNode,
    fvec: &Storage,
    nodes_visited: &mut u32,
    skipm: &mut PerformantFixedSet,
    is_indexing: bool,
    ef: u32,
) -> Result<Vec<(SharedNode, MetricResult)>, WaCustomError> {
    let mut candidate_queue = BinaryHeap::new();
    let mut results = BinaryHeap::new();

    let (start_version, _) = ProbLazyItem::get_latest_version(start_node, &dense_index.cache)?;
    let start_data = unsafe { &*start_version }.try_get_data(&dense_index.cache)?;
    let start_dist = dense_index
        .distance_metric
        .calculate(&fvec, &start_data.prop.value)?;
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
            ProbLazyItem::get_latest_version(current_node, &dense_index.cache)?;
        let node = unsafe { &*current_version }.try_get_data(&dense_index.cache)?;

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
                let neighbor_data = unsafe { &*neighbor_node }.try_get_data(&dense_index.cache)?;
                let dist = dense_index
                    .distance_metric
                    .calculate(&fvec, &neighbor_data.prop.value)?;
                skipm.insert(neighbor_id);
                candidate_queue.push((dist, neighbor_node));
            }
        }
    }

    let results = results
        .into_sorted_vec() // Convert BinaryHeap to a sorted Vec
        .into_iter() // Iterate over the sorted Vec
        .rev() // Reverse the order (to get descending order)
        .map(|(dist, node)| (node.clone(), dist)) // Map to the desired tuple format
        .take(if is_indexing { 64 } else { 100 }) // Limit the number of results
        .collect::<Vec<_>>(); // Collect into a Vec

    Ok(results)
}

// fn delete_node_update_neighbours(
//     dense_index: Arc<DenseIndex>,
//     item: LazyItem<MergedNode>,
//     skipm: HashSet<VectorId>,
//     version_id: Hash,
//     version_number: u16,
//     hnsw_level: HNSWLevel,
// ) -> Result<(), WaCustomError> {
//     let node = item.get_data(dense_index.cache.clone());
//     for nbr in node.neighbors.iter() {
//         let mut skipm = skipm.clone();
//         let nbr_node = nbr.1.get_data(dense_index.cache.clone());
//         let (nbr_id, nbr_vec) = match nbr_node.get_prop() {
//             PropState::Ready(prop) => (prop.id.clone(), prop.value.clone()),
//             PropState::Pending(_) => {
//                 // TODO: load prop
//                 return Err(WaCustomError::NodeError("PropState is Pending".to_string()));
//             }
//         };
//         skipm.insert(nbr_id);
//         let mut nbr_nbrs = traverse_find_nearest(
//             dense_index.clone(),
//             dense_index
//                 .root_vec
//                 .item
//                 .clone()
//                 .get()
//                 .get_latest_version(dense_index.cache.clone())
//                 .0,
//             nbr_vec,
//             0,
//             &mut skipm,
//             HNSWLevel(dense_index.hnsw_params.clone().get().num_layers),
//             false,
//         )?;

//         nbr_nbrs.truncate(20);
//         let nbr_nbrs_set = IdentitySet::from_iter(
//             nbr_nbrs
//                 .into_iter()
//                 .map(|(node, dist)| EagerLazyItem(dist, node)),
//         );

//         if let Some(version) = nbr.1.get_version(dense_index.cache.clone(), version_number) {
//             let nbr_current_version_data = version.get_data(dense_index.cache.clone());
//             nbr_current_version_data
//                 .neighbors
//                 .items
//                 .clone()
//                 .update(nbr_nbrs_set);
//         } else {
//             let (new_version, mut new_neighbours) = create_node_extract_neighbors(
//                 version_id,
//                 version_number,
//                 hnsw_level,
//                 nbr_node.prop.clone(),
//                 nbr_node.parent.clone(),
//                 nbr_node.child.clone(),
//             );
//             new_neighbours.items.update(nbr_nbrs_set);
//             nbr.1.add_version(dense_index.cache.clone(), new_version);
//             let mut exec_queue = dense_index.exec_queue_nodes.clone();
//             exec_queue
//                 .transactional_update(|queue| {
//                     let mut new_queue = queue.clone();
//                     new_queue.push(ArcShift::new(nbr.1.clone()));
//                     new_queue
//                 })
//                 .unwrap();
//         }
//     }
//     Ok(())
// }

// pub fn delete_vector_by_id(
//     dense_index: Arc<DenseIndex>,
//     vector_id: VectorId,
// ) -> Result<(), WaCustomError> {
//     let (current_version, current_version_number) = dense_index
//         .vcs
//         .add_next_version("main")
//         .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
//     dense_index.set_current_version(current_version);
//     update_current_version(&dense_index.lmdb, current_version)?;

//     let vec_raw = get_embedding_by_id(dense_index.clone(), vector_id.clone())?;
//     let quantized = Arc::new(dense_index.quantization_metric.clone().get().quantize(
//         &vec_raw.raw_vec,
//         dense_index.storage_type.clone().get().clone(),
//     )?);
//     let mut skipm = HashSet::new();
//     skipm.insert(vec_raw.hash_vec);
//     let items = traverse_find_nearest(
//         dense_index.clone(),
//         dense_index
//             .root_vec
//             .item
//             .clone()
//             .get()
//             .get_latest_version(dense_index.cache.clone())
//             .0,
//         quantized,
//         0,
//         &mut skipm,
//         HNSWLevel(dense_index.hnsw_params.clone().get().num_layers),
//         false,
//     )?;

//     let mut maybe_item = None;

//     for (item, _) in items {
//         let data = item.get_data(dense_index.cache.clone());
//         let prop = match data.get_prop() {
//             PropState::Ready(prop) => prop.id.clone(),
//             PropState::Pending(_) => {
//                 // TODO: load prop
//                 return Err(WaCustomError::NodeError("PropState is Pending".to_string()));
//             }
//         };

//         if prop == vector_id {
//             maybe_item = Some(item);
//         }
//     }

//     let Some(mut item) = maybe_item else {
//         return Err(WaCustomError::NodeError(
//             "Node not found in graph".to_string(),
//         ));
//     };

//     for level in 0..=dense_index.hnsw_params.clone().get().num_layers {
//         delete_node_update_neighbours(
//             dense_index.clone(),
//             item.clone(),
//             skipm.clone(),
//             current_version,
//             *current_version_number as u16,
//             HNSWLevel(level),
//         )?;

//         item = item
//             .get_data(dense_index.cache.clone())
//             .parent
//             .clone()
//             .item
//             .get()
//             .clone();
//     }

//     auto_commit_transaction(dense_index)?;

//     Ok(())
// }

// pub fn delete_vector_by_id_in_transaction(
//     dense_index: Arc<DenseIndex>,
//     vector_id: VectorId,
//     transaction_id: Hash,
// ) -> Result<(), WaCustomError> {
//     let txn = dense_index
//         .lmdb
//         .env
//         .begin_ro_txn()
//         .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;
//     let version_hash = dense_index
//         .vcs
//         .get_version_hash(&transaction_id, &txn)
//         .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?
//         .ok_or(WaCustomError::DatabaseError(
//             "VersionHash not found for transaction".to_string(),
//         ))?;
//     txn.abort();
//     let current_version_number = *version_hash.version as u16;

//     let vec_raw = get_embedding_by_id(dense_index.clone(), vector_id.clone())?;
//     let quantized = Arc::new(dense_index.quantization_metric.clone().get().quantize(
//         &vec_raw.raw_vec,
//         dense_index.storage_type.clone().get().clone(),
//     )?);
//     let mut skipm = HashSet::new();
//     skipm.insert(vec_raw.hash_vec);
//     let items = traverse_find_nearest(
//         dense_index.clone(),
//         dense_index
//             .root_vec
//             .item
//             .clone()
//             .get()
//             .get_latest_version(dense_index.cache.clone())
//             .0,
//         quantized,
//         0,
//         &mut skipm,
//         HNSWLevel(dense_index.hnsw_params.clone().get().num_layers),
//         false,
//     )?;

//     let mut maybe_item = None;

//     for (item, _) in items {
//         let data = item.get_data(dense_index.cache.clone());
//         let prop = match data.get_prop() {
//             PropState::Ready(prop) => prop.id.clone(),
//             PropState::Pending(_) => {
//                 // TODO: load prop
//                 return Err(WaCustomError::NodeError("PropState is Pending".to_string()));
//             }
//         };

//         if prop == vector_id {
//             maybe_item = Some(item);
//         }
//     }

//     let Some(mut item) = maybe_item else {
//         return Err(WaCustomError::NodeError(
//             "Node not found in graph".to_string(),
//         ));
//     };

//     for level in 0..=dense_index.hnsw_params.clone().get().num_layers {
//         delete_node_update_neighbours(
//             dense_index.clone(),
//             item.clone(),
//             skipm.clone(),
//             transaction_id,
//             current_version_number,
//             HNSWLevel(level),
//         )?;

//         item = item
//             .get_data(dense_index.cache.clone())
//             .parent
//             .clone()
//             .item
//             .get()
//             .clone();
//     }

//     auto_commit_transaction(dense_index)?;

//     Ok(())
// }
