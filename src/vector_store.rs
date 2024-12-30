use crate::app_context::AppContext;
use crate::config_loader::VectorsIndexingMode;
use crate::distance::cosine::CosineSimilarity;
use crate::distance::DistanceFunction;
use crate::indexes::inverted_index::InvertedIndex;
use crate::indexes::inverted_index_types::RawSparseVectorEmbedding;
use crate::macros::key;
use crate::models::buffered_io::*;
use crate::models::common::*;
use crate::models::embedding_persist::*;
use crate::models::file_persist::*;
use crate::models::fixedset::PerformantFixedSet;
use crate::models::lazy_load::*;
use crate::models::prob_lazy_load::lazy_item::ProbLazyItem;
use crate::models::prob_node::ProbNode;
use crate::models::prob_node::SharedNode;
use crate::models::types::*;
use crate::models::versioning::Hash;
use crate::quantization::{Quantization, StorageType};
use crate::storage::Storage;
use lmdb::{Transaction, WriteFlags};
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use smallvec::SmallVec;
use std::array::TryFromSliceError;
use std::fs::File;
use std::io::SeekFrom;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::RwLock;

pub fn create_root_node(
    num_layers: u8,
    quantization_metric: &QuantizationMetric,
    storage_type: StorageType,
    dim: usize,
    prop_file: Arc<RwLock<File>>,
    hash: Hash,
    index_manager: Arc<BufferManagerFactory>,
    neighbors_count: usize,
) -> Result<SharedNode, WaCustomError> {
    let min = -1.0;
    let max = 1.0;
    let vec = (0..dim)
        .map(|_| {
            let mut rng = rand::thread_rng();

            let random_number: f32 = rng.gen_range(min..max);
            random_number
        })
        .collect::<Vec<f32>>();
    let vec_hash = VectorId(u64::MAX);

    let vector_list = Arc::new(quantization_metric.quantize(&vec, storage_type)?);

    let mut prop_file_guard = prop_file.write().unwrap();
    let location = write_prop_to_file(&vec_hash, vector_list.clone(), &mut *prop_file_guard)?;
    drop(prop_file_guard);

    let prop = Arc::new(NodeProp {
        id: vec_hash,
        value: vector_list.clone(),
        location,
    });

    let mut root = ProbLazyItem::new(
        ProbNode::new(HNSWLevel(0), prop.clone(), None, None, neighbors_count),
        hash,
        0,
    );

    let mut nodes = Vec::new();

    for l in 1..=num_layers {
        let current_node = ProbNode::new(
            HNSWLevel(l),
            prop.clone(),
            None,
            Some(root.clone()),
            neighbors_count,
        );

        let lazy_node = ProbLazyItem::new(current_node, hash, 0);

        if let Some(prev_node) = root.get_lazy_data() {
            prev_node.set_parent(lazy_node.clone());
        }
        root = lazy_node.clone();

        nodes.push(lazy_node);
    }

    for item in nodes {
        write_node_to_file(&item, &index_manager)?;
    }

    Ok(root)
}

pub fn ann_search(
    dense_index: Arc<DenseIndex>,
    vector_emb: QuantizedVectorEmbedding,
    cur_entry: SharedNode,
    cur_level: HNSWLevel,
    neighbors_count: usize,
) -> Result<Vec<(SharedNode, MetricResult)>, WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = PerformantFixedSet::new();
    skipm.insert(vector_emb.hash_vec.0);

    let cur_node = cur_entry.try_get_data(&dense_index.cache)?;

    let z = traverse_find_nearest(
        &dense_index,
        &cur_entry,
        &fvec,
        0,
        &mut skipm,
        cur_level,
        false,
        true,
    )?;

    let mut z = if z.is_empty() {
        let dist = dense_index
            .distance_metric
            .calculate(&fvec, &cur_node.prop.value)?;

        vec![(cur_entry.clone(), dist)]
    } else {
        z
    };

    if cur_level.0 != 0 {
        let results = ann_search(
            dense_index.clone(),
            vector_emb,
            z[0].0
                .try_get_data(&dense_index.cache)?
                .get_child()
                .unwrap(),
            HNSWLevel(cur_level.0 - 1),
            neighbors_count,
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

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot_prod: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let mag_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    // Avoid division by zero
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot_prod / (mag_a * mag_b)
}

pub fn finalize_ann_results(
    dense_index: Arc<DenseIndex>,
    results: Vec<(SharedNode, MetricResult)>,
    query: &[f32],
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let filtered = remove_duplicates_and_filter(results);
    let mut results = Vec::new();

    for (id, _) in filtered {
        let raw = get_embedding_by_id(dense_index.clone(), &id)?;
        let cs = cosine_similarity(query, &raw.raw_vec);
        results.push((id, MetricResult::CosineSimilarity(CosineSimilarity(cs))));
    }
    results.sort_unstable_by(|(_, a), (_, b)| {
        b.get_value()
            .partial_cmp(&a.get_value())
            .unwrap_or(std::cmp::Ordering::Greater)
    });
    results.truncate(5);
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
    let bufman = dense_index.vec_raw_manager.get(&current_version)?;
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

    txn.put(
        *db,
        &"count_unindexed",
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
    dense_index: Arc<DenseIndex>,
    upload_process_batch_size: usize,
    serialization_table: Arc<TSHashTable<SharedNode, ()>>,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    neighbors_count: usize,
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
                    let data = current_entry
                        .try_get_data(&dense_index.cache)
                        .expect("Unable to load data");
                    if data.hnsw_level.0 > current_level.0 {
                        current_entry = data.get_child().unwrap();
                    } else if data.hnsw_level == current_level {
                        break;
                    } else {
                        panic!("missing node");
                    }
                }

                index_embedding(
                    dense_index.clone(),
                    None,
                    embedding,
                    prop,
                    current_entry,
                    current_level,
                    version,
                    version_number,
                    serialization_table.clone(),
                    lazy_item_versions_table.clone(),
                    neighbors_count,
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

        txn.put(
            *db,
            &"next_embedding_offset",
            &next_embedding_offset_serialized,
            WriteFlags::empty(),
        )
        .map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to update `next_embedding_offset`: {}", e))
        })?;

        txn.put(
            *db,
            &"count_indexed",
            &count_indexed.to_le_bytes(),
            WriteFlags::empty(),
        )
        .map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to update `count_indexed`: {}", e))
        })?;

        txn.put(
            *db,
            &"count_unindexed",
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

    let bufman = dense_index.vec_raw_manager.get(&version)?;

    let mut i = embedding_offset.offset;
    let cursor = bufman.open_cursor()?;
    let file_len = bufman.seek_with_cursor(cursor, SeekFrom::End(0))? as u32;
    bufman.seek_with_cursor(cursor, SeekFrom::Start(0))?;

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
    let index = |vecs: Vec<(VectorId, Vec<f32>)>| {
        for (id, values) in vecs {
            let raw_emb = RawVectorEmbedding {
                hash_vec: id,
                raw_vec: Arc::new(values),
            };
            transaction.post_raw_embedding(raw_emb.clone());
            let lp = &dense_index.levels_prob;
            let iv = get_max_insert_level(rand::random::<f32>().into(), lp.clone());

            let quantized_vec = Arc::new(quantization.quantize(
                &raw_emb.raw_vec,
                dense_index.storage_type.clone().get().clone(),
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

            let current_level = HNSWLevel(iv as u8);
            let mut current_entry = dense_index.get_root_vec();

            loop {
                let data = current_entry.try_get_data(&dense_index.cache)?;
                if data.hnsw_level.0 > current_level.0 {
                    current_entry = data.get_child().unwrap();
                } else if data.hnsw_level == current_level {
                    break;
                } else {
                    panic!("missing node");
                }
            }

            index_embedding(
                dense_index.clone(),
                None,
                embedding,
                prop,
                current_entry,
                current_level,
                version,
                version_number,
                transaction.serialization_table.clone(),
                transaction.lazy_item_versions_table.clone(),
                ctx.config.hnsw.neighbors_count,
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
                .collect::<Result<_, _>>()?;
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
    dense_index: Arc<DenseIndex>,
    parent: Option<SharedNode>,
    vector_emb: QuantizedVectorEmbedding,
    prop: Arc<NodeProp>,
    cur_entry: SharedNode,
    cur_level: HNSWLevel,
    version: Hash,
    version_number: u16,
    serialization_table: Arc<TSHashTable<SharedNode, ()>>,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    neighbors_count: usize,
) -> Result<(), WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = PerformantFixedSet::new();
    skipm.insert(vector_emb.hash_vec.0);

    let cur_node = cur_entry
        .get_latest_version(&dense_index.cache)?
        .0
        .try_get_data(&dense_index.cache)?;

    let z = traverse_find_nearest(
        &dense_index,
        &cur_entry,
        &fvec,
        0,
        &mut skipm,
        cur_level,
        true,
        false,
    )?;

    let z = if z.is_empty() {
        let dist = dense_index
            .distance_metric
            .calculate(&fvec, &cur_node.prop.value)?;

        vec![(cur_entry, dist)]
    } else {
        z
    };

    let lazy_node = create_node(
        version,
        version_number,
        cur_level,
        prop.clone(),
        parent.clone(),
        None,
        neighbors_count,
    );

    let node = lazy_node.get_lazy_data().unwrap();

    if let Some(parent) = parent {
        parent
            .try_get_data(&dense_index.cache)
            .unwrap()
            .set_child(lazy_node.clone());
    }

    if cur_level.0 != 0 {
        index_embedding(
            dense_index.clone(),
            Some(lazy_node.clone()),
            vector_emb.clone(),
            prop.clone(),
            z[0].0
                .try_get_data(&dense_index.cache)?
                .get_child()
                .unwrap(),
            HNSWLevel(cur_level.0 - 1),
            version,
            version_number,
            serialization_table.clone(),
            lazy_item_versions_table.clone(),
            neighbors_count,
        )?;
    }

    create_node_edges(
        dense_index.clone(),
        lazy_node,
        node,
        z,
        version,
        version_number,
        serialization_table,
        lazy_item_versions_table,
    )
    .expect("Failed insert_node_create_edges");

    Ok(())
}

fn create_node(
    version_id: Hash,
    version_number: u16,
    hnsw_level: HNSWLevel,
    prop: Arc<NodeProp>,
    parent: Option<SharedNode>,
    child: Option<SharedNode>,
    neighbors_count: usize,
) -> SharedNode {
    let node = ProbNode::new(hnsw_level, prop, parent, child, neighbors_count);
    ProbLazyItem::new(node, version_id, version_number)
}

fn get_or_create_version(
    dense_index: Arc<DenseIndex>,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    lazy_item: SharedNode,
    version: Hash,
    version_number: u16,
) -> Result<SharedNode, WaCustomError> {
    let node = lazy_item.try_get_data(&dense_index.cache)?;

    let new_version = lazy_item_versions_table.get_or_create(
        (node.get_id().clone(), version_number, node.hnsw_level.0),
        || {
            if let Some(version) = lazy_item
                .get_version(version_number, &dense_index.cache)
                .expect("Deserialization failed")
            {
                return version;
            }

            let new_node = ProbNode::new_with_neighbors(
                node.hnsw_level,
                node.prop.clone(),
                node.clone_neighbors(),
                node.get_parent(),
                node.get_child(),
            );

            let version = ProbLazyItem::new(new_node, version, version_number);

            lazy_item
                .add_version(version.clone(), &dense_index.cache)
                .expect("Failed to add version")
                .map_err(|_| "Failed to add version")
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
    serialization_table: Arc<TSHashTable<SharedNode, ()>>,
    lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
) -> Result<(), WaCustomError> {
    for (neighbor, dist) in neighbors.into_iter() {
        serialization_table.insert(neighbor.clone(), ());

        let new_lazy_neighbor = get_or_create_version(
            dense_index.clone(),
            lazy_item_versions_table.clone(),
            neighbor,
            version,
            version_number,
        )?;
        let new_neighbor = new_lazy_neighbor.try_get_data(&dense_index.cache)?;

        node.add_neighbor(new_lazy_neighbor, new_neighbor.get_id(), dist);
        new_neighbor.add_neighbor(lazy_node.clone(), &node.prop.id, dist);
    }

    serialization_table.insert(lazy_node, ());

    Ok(())
}

fn traverse_find_nearest(
    dense_index: &DenseIndex,
    vtm: &SharedNode,
    fvec: &Storage,
    hops: u8,
    skipm: &mut PerformantFixedSet,
    cur_level: HNSWLevel,
    truncate_results: bool,
    shortlist: bool,
) -> Result<Vec<(SharedNode, MetricResult)>, WaCustomError> {
    let mut tasks: SmallVec<[Vec<(SharedNode, MetricResult)>; 24]> = SmallVec::new();

    let (latest_version_lazy_node, _latest_version) = vtm.get_latest_version(&dense_index.cache)?;

    let node = latest_version_lazy_node.try_get_data(&dense_index.cache)?;
    if shortlist {
        let mut neighbors = Vec::new();

        for neighbor in node.get_neighbors_raw() {
            let neighbor_node = unsafe {
                if let Some(neighbor) = neighbor.load(Ordering::Acquire).as_ref() {
                    neighbor.0.clone()
                } else {
                    continue;
                }
            };
            let neighbor = neighbor_node.try_get_data(&dense_index.cache)?;

            if skipm.is_member(neighbor.prop.id.0) {
                continue;
            }

            skipm.insert(neighbor.prop.id.0);

            let dist = dense_index
                .distance_metric
                .calculate(&fvec, &neighbor.prop.value)?;

            neighbors.push((neighbor_node, dist));
        }

        neighbors.sort_unstable_by(|(_, a), (_, b)| {
            b.get_value()
                .partial_cmp(&a.get_value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (neighbor_idx, (neighbor_node, dist)) in neighbors.into_iter().enumerate() {
            if hops <= 10 && neighbor_idx < 10 {
                let mut z = traverse_find_nearest(
                    dense_index,
                    &neighbor_node,
                    fvec,
                    hops + 1,
                    skipm,
                    cur_level,
                    truncate_results,
                    true,
                )?;
                z.push((neighbor_node, dist));
                tasks.push(z);
            } else {
                tasks.push(vec![(neighbor_node, dist)]);
            }
        }
    } else {
        for neighbor in node.get_neighbors_raw() {
            let neighbor_node = unsafe {
                if let Some(neighbor) = neighbor.load(Ordering::Acquire).as_ref() {
                    neighbor.0.clone()
                } else {
                    continue;
                }
            };
            let neighbor = neighbor_node.try_get_data(&dense_index.cache)?;

            if skipm.is_member(neighbor.prop.id.0) {
                continue;
            }
            skipm.insert(neighbor.prop.id.0);

            let dist = dense_index
                .distance_metric
                .calculate(&fvec, &neighbor.prop.value)?;

            if hops <= 10 {
                let mut z = traverse_find_nearest(
                    dense_index,
                    &neighbor_node,
                    fvec,
                    hops + 1,
                    skipm,
                    cur_level,
                    truncate_results,
                    false,
                )?;
                z.push((neighbor_node, dist));
                tasks.push(z);
            } else {
                tasks.push(vec![(neighbor_node, dist)]);
            }
        }
    }

    let mut nn: Vec<_> = tasks.into_iter().flatten().collect();
    nn.sort_unstable_by(|a, b| b.1.get_value().partial_cmp(&a.1.get_value()).unwrap());
    if truncate_results {
        nn.truncate(5);
    }
    Ok(nn)
}

pub fn create_index_in_collection(
    ctx: Arc<AppContext>,
    dense_index: Arc<DenseIndex>,
) -> Result<(), WaCustomError> {
    let mut quantization_metric_arc = dense_index.quantization_metric.clone();
    let quantization_metric = quantization_metric_arc.get();

    let hash = dense_index.get_root_vec().get_current_version();

    let root = create_root_node(
        dense_index.hnsw_params.num_layers,
        quantization_metric,
        *dense_index.storage_type.clone().get(),
        dense_index.dim,
        dense_index.prop_file.clone(),
        hash,
        dense_index.index_manager.clone(),
        ctx.config.hnsw.neighbors_count,
    )?;

    // The whole index is empty now
    dense_index.set_root_vec(root);

    Ok(())
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
