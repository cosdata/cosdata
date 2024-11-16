use crate::app_context::AppContext;
use crate::distance::DistanceFunction;
use crate::macros::key;
use crate::models::buffered_io::BufferManager;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::Allocator;
use crate::models::common::*;
use crate::models::file_persist::*;
use crate::models::kmeans::{concat_vectors, generate_initial_centroids, kmeans, should_continue};
use crate::models::lazy_load::*;
use crate::models::prob_lazy_load::lazy_item::ProbLazyItem;
use crate::models::types::*;
use crate::models::versioning::Hash;
use crate::quantization::{Quantization, StorageType};
use crate::storage::Storage;
use arcshift::ArcShift;
use lmdb::{Transaction, WriteFlags};
use rand::Rng;
use smallvec::SmallVec;
use std::array::TryFromSliceError;
use std::collections::HashSet;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::SeekFrom;
use std::path::Path;
use std::ptr;
use std::sync::mpsc;
use std::sync::Arc;

pub fn create_root_node(
    allocator: &Allocator,
    num_layers: u8,
    quantization_metric: &QuantizationMetric,
    storage_type: StorageType,
    dim: usize,
    prop_file: Arc<File>,
    hash: Hash,
    index_manager: Arc<BufferManagerFactory>,
) -> Result<*mut ProbLazyItem<ProbNode>, WaCustomError> {
    let min = -1.0;
    let max = 1.0;
    let vec = (0..dim)
        .map(|_| {
            let mut rng = rand::thread_rng();

            let random_number: f32 = rng.gen_range(min..max);
            random_number
        })
        .collect::<Vec<f32>>();
    let vec_hash = VectorId::Int(-1);

    let vector_list = Arc::new(quantization_metric.quantize(&vec, storage_type)?);

    let location = write_prop_to_file(&vec_hash, vector_list.clone(), &prop_file)?;

    let prop = ArcShift::new(PropState::Ready(Arc::new(NodeProp {
        id: vec_hash,
        value: vector_list.clone(),
        location,
    })));

    let mut root: *mut ProbLazyItem<ProbNode> = ptr::null_mut();

    let mut nodes = Vec::new();

    for l in 0..=num_layers {
        let current_node = ProbNode::new(HNSWLevel(l), prop.clone(), ptr::null_mut(), root.clone());

        let lazy_node = ProbLazyItem::new(allocator, current_node, hash, 0);

        if let Some(prev_node) = unsafe { root.as_ref() }.and_then(|root| root.get_lazy_data()) {
            prev_node.set_parent(lazy_node.clone());
        }
        root = lazy_node;

        nodes.push(lazy_node);
    }

    for item in nodes {
        write_node_to_file(item, index_manager.clone())?;
    }

    Ok(root)
}

pub fn ann_search(
    dense_index: Arc<DenseIndex>,
    vector_emb: QuantizedVectorEmbedding,
    cur_entry: *mut ProbLazyItem<ProbNode>,
    cur_level: HNSWLevel,
) -> Result<Vec<(*mut ProbLazyItem<ProbNode>, MetricResult)>, WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = HashSet::new();
    skipm.insert(vector_emb.hash_vec.clone());

    let cur_node = unsafe { &*cur_entry }.try_get_data(dense_index.cache.clone())?;

    let mut prop_arc = cur_node.prop.clone();
    let prop_state = prop_arc.get();

    let node_prop = match &*prop_state {
        PropState::Ready(prop) => prop,
        PropState::Pending(_) => {
            return Err(WaCustomError::NodeError(
                "Node prop is in pending state".to_string(),
            ))
        }
    };

    let z = traverse_find_nearest(
        dense_index.clone(),
        cur_entry.clone(),
        fvec.clone(),
        0,
        &mut skipm,
        cur_level,
        false,
    )?;

    let dist = dense_index
        .distance_metric
        .calculate(&fvec, &node_prop.value)?;

    let mut z = if z.is_empty() {
        vec![(cur_entry.clone(), dist)]
    } else {
        z
    };

    if cur_level.0 != 0 {
        let results = ann_search(
            dense_index.clone(),
            vector_emb,
            unsafe { &*z[0].0 }
                .try_get_data(dense_index.cache.clone())?
                .get_child(),
            HNSWLevel(cur_level.0 - 1),
        )?;

        z.extend(results);
    };

    Ok(z)
}

pub fn vector_fetch(
    dense_index: Arc<DenseIndex>,
    vector_id: VectorId,
) -> Result<Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>>, WaCustomError> {
    let mut results = Vec::new();

    for lev in 0..=dense_index.hnsw_params.clone().get().num_layers {
        let neighbors = None;

        results.push(neighbors);
    }
    Ok(results)
}

fn get_vector_id_from_node(node: Arc<MergedNode>) -> Option<VectorId> {
    let mut prop_arc = node.prop.clone();
    match prop_arc.get() {
        PropState::Ready(node_prop) => Some(node_prop.id.clone()),
        PropState::Pending(_) => None,
    }
}

pub fn write_embedding(
    bufman: Arc<BufferManager>,
    emb: &RawVectorEmbedding,
) -> Result<u32, WaCustomError> {
    // TODO: select a better value for `N` (number of bytes to pre-allocate)
    let serialized = rkyv::to_bytes::<_, 256>(emb)
        .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

    let len = serialized.len() as u32;
    let cursor = bufman.open_cursor()?;

    let start = bufman.seek_with_cursor(cursor, SeekFrom::End(0))? as u32;
    bufman.write_u32_with_cursor(cursor, len)?;
    bufman.write_with_cursor(cursor, &serialized)?;

    bufman.close_cursor(cursor)?;

    Ok(start)
}

pub struct EmbeddingOffset {
    pub version: Hash,
    pub offset: u32,
}

impl EmbeddingOffset {
    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(8);

        result.extend_from_slice(&self.version.to_le_bytes());
        result.extend_from_slice(&self.offset.to_le_bytes());

        result
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != 8 {
            return Err("Input must be exactly 8 bytes");
        }

        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let offset = u32::from_le_bytes(bytes[4..8].try_into().unwrap());

        Ok(Self {
            version: Hash::from(version),
            offset,
        })
    }
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
    vector_id: VectorId,
) -> Result<RawVectorEmbedding, WaCustomError> {
    let env = dense_index.lmdb.env.clone();
    let db = dense_index.lmdb.db.clone();

    let txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let embedding_key = key!(e:vector_id);

    let offset_serialized = txn.get(*db, &embedding_key).map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to get serialized embedding offset: {}", e))
    })?;

    let embedding_offset = EmbeddingOffset::deserialize(offset_serialized)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let offset = embedding_offset.offset;
    let current_version = embedding_offset.version;
    let bufman = dense_index.vec_raw_manager.get(&current_version)?;
    let (embedding, _next) = read_embedding(bufman.clone(), offset)?;

    Ok(embedding)
}

fn read_embedding(
    bufman: Arc<BufferManager>,
    offset: u32,
) -> Result<(RawVectorEmbedding, u32), WaCustomError> {
    let cursor = bufman.open_cursor()?;

    bufman
        .seek_with_cursor(cursor, SeekFrom::Start(offset as u64))
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let len = bufman
        .read_u32_with_cursor(cursor)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let mut buf = vec![0; len as usize];

    bufman
        .read_with_cursor(cursor, &mut buf)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let emb = unsafe { rkyv::from_bytes_unchecked(&buf) }.map_err(|e| {
        WaCustomError::DeserializationError(format!("Failed to deserialize VectorEmbedding: {}", e))
    })?;

    let next = bufman
        .cursor_position(cursor)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))? as u32;

    bufman.close_cursor(cursor)?;

    Ok((emb, next))
}

fn auto_config_storage_type(dense_index: Arc<DenseIndex>, vectors: &[&[f32]]) {
    let threshold = 0.0;
    let iterations = 32;

    let vec = concat_vectors(vectors);

    // First iteration with k = 16
    let initial_centroids_16 = generate_initial_centroids(&vec, 16);
    let (_, counts_16) = kmeans(&vec, &initial_centroids_16, iterations);
    let storage_type = if should_continue(&counts_16, threshold, 8) {
        // Second iteration with k = 8
        let initial_centroids_8 = generate_initial_centroids(&vec, 8);
        let (_, counts_8) = kmeans(&vec, &initial_centroids_8, iterations);
        if should_continue(&counts_8, threshold, 4) {
            // Third iteration with k = 4
            let initial_centroids_4 = generate_initial_centroids(&vec, 4);
            let (_, counts_4) = kmeans(&vec, &initial_centroids_4, iterations);

            if should_continue(&counts_4, threshold, 2) {
                StorageType::SubByte(1)
            } else {
                StorageType::SubByte(2)
            }
        } else {
            // StorageType::SubByte(3)
            StorageType::UnsignedByte
        }
    } else {
        StorageType::UnsignedByte
    };

    dense_index.storage_type.update_shared(storage_type);
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
    thread_id: u32,
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
        // Set to auto config but is not configured
        if dense_index.get_auto_config_flag() && !dense_index.get_configured_flag() {
            let quantization = quantization_arc.get();
            let mut new_quantization = quantization.clone();
            let vectors: Vec<&[f32]> = embeddings
                .iter()
                .map(|embedding| &embedding.raw_vec as &[f32])
                .collect();
            new_quantization.train(&vectors)?;
            quantization_arc.update(new_quantization);
            auto_config_storage_type(dense_index.clone(), &vectors);
            dense_index.set_configured_flag(true);
        }
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
                let location = write_prop_to_file(
                    &raw_emb.hash_vec,
                    quantized_vec.clone(),
                    &dense_index.prop_file,
                )
                .expect("failed to write prop");
                let prop = ArcShift::new(PropState::Ready(Arc::new(NodeProp {
                    id: raw_emb.hash_vec.clone(),
                    value: quantized_vec.clone(),
                    location,
                })));
                let embedding = QuantizedVectorEmbedding {
                    quantized_vec,
                    hash_vec: raw_emb.hash_vec,
                };

                let current_level = HNSWLevel(iv.try_into().unwrap());

                let mut current_entry = dense_index.get_root_vec();

                loop {
                    let data = unsafe { &*current_entry }
                        .try_get_data(dense_index.cache.clone())
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
                    dense_index.clone(),
                    ptr::null_mut(),
                    embedding,
                    prop,
                    current_entry,
                    current_level,
                    version,
                    version_number,
                    thread_id,
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
    transaction_id: Hash,
    embeddings: mpsc::Receiver<RawVectorEmbedding>,
    thread_id: u32,
) -> Result<(), WaCustomError> {
    let version = transaction_id;

    let txn = dense_index
        .lmdb
        .env
        .begin_ro_txn()
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;
    let version_hash = dense_index
        .vcs
        .get_version_hash(&version, &txn)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?
        .expect("Current version hash not found");
    let version_number = *version_hash.version as u16;
    txn.abort();

    println!("debug 0");
    let mut quantization_arc = dense_index.quantization_metric.clone();
    println!("debug 1");
    // Set to auto config but is not configured
    if dense_index.get_auto_config_flag() && !dense_index.get_configured_flag() {
        let quantization = quantization_arc.get();
        let mut new_quantization = quantization.clone();
        let embeddings: Vec<_> = embeddings.into_iter().collect();
        let vectors: Vec<&[f32]> = embeddings
            .iter()
            .map(|embedding| &embedding.raw_vec as &[f32])
            .collect();
        new_quantization.train(&vectors)?;
        quantization_arc.update(new_quantization);
        auto_config_storage_type(dense_index.clone(), &vectors);
        dense_index.set_configured_flag(true);

        let quantization = quantization_arc.get();

        let root = create_root_node(
            &dense_index.cache.get_allocator(),
            dense_index.hnsw_params.clone().get().num_layers,
            quantization,
            dense_index.storage_type.clone().get().clone(),
            dense_index.dim,
            dense_index.prop_file.clone(),
            unsafe { &*dense_index.get_root_vec() }.get_current_version(),
            dense_index.index_manager.clone(),
        )?;

        dense_index.set_root_vec(root);

        index_embeddings_in_transaction_inner(
            ctx.clone(),
            dense_index,
            embeddings.into_iter(),
            quantization,
            version,
            version_number,
            thread_id,
        );
    } else {
        let quantization = quantization_arc.get();
        index_embeddings_in_transaction_inner(
            ctx.clone(),
            dense_index,
            embeddings.into_iter(),
            quantization,
            version,
            version_number,
            thread_id,
        );
    }

    fn index_embeddings_in_transaction_inner(
        ctx: Arc<AppContext>,
        dense_index: Arc<DenseIndex>,
        embeddings: impl Iterator<Item = RawVectorEmbedding> + Send,
        quantization: &QuantizationMetric,
        version: Hash,
        version_number: u16,
        thread_id: u32,
    ) {
        ctx.threadpool.scope(|s| {
            let workers: [_; 16] = std::array::from_fn(|_| {
                let (tx, rx) = mpsc::channel::<RawVectorEmbedding>();
                let dense_index = dense_index.clone();
                let quantization = quantization.clone();
                s.spawn(move |_| {
                    for raw_emb in rx {
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
                        let location = write_prop_to_file(
                            &raw_emb.hash_vec,
                            quantized_vec.clone(),
                            &dense_index.prop_file,
                        )
                        .expect("failed to write prop");

                        let prop = ArcShift::new(PropState::Ready(Arc::new(NodeProp {
                            id: raw_emb.hash_vec.clone(),
                            value: quantized_vec.clone(),
                            location,
                        })));
                        let embedding = QuantizedVectorEmbedding {
                            quantized_vec,
                            hash_vec: raw_emb.hash_vec,
                        };

                        let current_level = HNSWLevel(iv.try_into().unwrap());

                        let mut current_entry = dense_index.get_root_vec();

                        loop {
                            let data = unsafe { &*current_entry }
                                .try_get_data(dense_index.cache.clone())
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
                            dense_index.clone(),
                            ptr::null_mut(),
                            embedding,
                            prop,
                            current_entry,
                            current_level,
                            version,
                            version_number,
                            thread_id,
                        )
                        .expect("index_embedding failed");
                    }
                });
                tx
            });

            let mut worker_idx = 0;

            for raw_emb in embeddings {
                workers[worker_idx].send(raw_emb).unwrap();
                worker_idx = (worker_idx + 1) % 16;
            }

            drop(workers);
        });
    }

    Ok(())
}

pub fn index_embedding(
    dense_index: Arc<DenseIndex>,
    parent: *mut ProbLazyItem<ProbNode>,
    vector_emb: QuantizedVectorEmbedding,
    prop: ArcShift<PropState>,
    cur_entry: *mut ProbLazyItem<ProbNode>,
    cur_level: HNSWLevel,
    version: Hash,
    version_number: u16,
    thread_id: u32,
) -> Result<(), WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = HashSet::new();
    skipm.insert(vector_emb.hash_vec.clone());

    let cur_node =
        unsafe { &*ProbLazyItem::get_latest_version(cur_entry, dense_index.cache.clone())?.0 }
            .try_get_data(dense_index.cache.clone())?;

    let mut prop_arc = cur_node.prop.clone();
    let prop_state = prop_arc.get();

    let node_prop = match &*prop_state {
        PropState::Ready(prop) => prop,
        PropState::Pending(_) => {
            return Err(WaCustomError::NodeError(
                "Node prop is in pending state".to_string(),
            ))
        }
    };

    let z = traverse_find_nearest(
        dense_index.clone(),
        cur_entry.clone(),
        fvec.clone(),
        0,
        &mut skipm,
        cur_level,
        true,
    )?;

    // @DOUBT: May be this distance can be calculated only if z is
    // empty
    let dist = dense_index
        .distance_metric
        .calculate(&fvec, &node_prop.value)?;

    // @DOUBT: Perhaps, traverse_find_nearest can itself handle this
    // case. Because, logically not finding nearest nodes doesn't make
    // sense given that cur_entry is not optional (root node always
    // exists). In that case cur_entry will be the nearest which is
    // how that case is being handled below.
    let z = if z.is_empty() {
        vec![(cur_entry.clone(), dist)]
    } else {
        z
    };

    let z_clone: Vec<_> = z.iter().map(|(first, _)| first.clone()).collect();

    let (lazy_node, node) = create_node(
        &dense_index.cache.get_allocator(),
        version,
        version_number,
        cur_level,
        prop.clone(),
        parent,
        ptr::null_mut(),
    );

    unsafe {
        if let Some(parent) = parent.as_ref() {
            parent.get_lazy_data().unwrap().set_child(lazy_node.clone());
        }
    }

    if cur_level.0 != 0 {
        index_embedding(
            dense_index.clone(),
            lazy_node,
            vector_emb.clone(),
            prop.clone(),
            unsafe { &*z_clone[0] }
                .try_get_data(dense_index.cache.clone())?
                .get_child(),
            HNSWLevel(cur_level.0 - 1),
            version,
            version_number,
            thread_id,
        )?;
    }

    create_node_edges(
        dense_index.clone(),
        lazy_node,
        node,
        z,
        cur_level,
        version,
        version_number,
        thread_id,
    )
    .expect("Failed insert_node_create_edges");

    Ok(())
}

pub fn queue_node_prop_exec(
    dense_index: Arc<DenseIndex>,
    lazy_node: *mut ProbLazyItem<ProbNode>,
    node: Arc<ProbNode>,
    thread_id: u32,
) -> Result<(), WaCustomError> {
    for neighbor in node.get_neighbors() {
        unsafe {
            let neighbor_node = &*neighbor.0;
            // neighbor_node.lock(thread_id);
            neighbor_node.set_persistence(true);
        }
    }

    // Add the node to exec_queue_nodes
    let mut exec_queue = dense_index.exec_queue_nodes.clone();
    exec_queue
        .transactional_update(|queue| {
            let mut new_queue = queue.clone();
            new_queue.push(lazy_node.clone());
            new_queue
        })
        .unwrap();

    Ok(())
}

pub fn auto_commit_transaction(dense_index: Arc<DenseIndex>) -> Result<(), WaCustomError> {
    // Retrieve exec_queue_nodes from dense_index
    let mut exec_queue_nodes_arc = dense_index.exec_queue_nodes.clone();
    let exec_queue_nodes = exec_queue_nodes_arc.get().clone();

    for node in exec_queue_nodes.iter() {
        write_node_to_file(*node, dense_index.index_manager.clone())?;
    }

    exec_queue_nodes_arc.update(Vec::new());

    dense_index.index_manager.flush_all()?;

    Ok(())
}

fn create_node(
    allocator: &Allocator,
    version_id: Hash,
    version_number: u16,
    hnsw_level: HNSWLevel,
    prop: ArcShift<PropState>,
    parent: *mut ProbLazyItem<ProbNode>,
    child: *mut ProbLazyItem<ProbNode>,
) -> (*mut ProbLazyItem<ProbNode>, Arc<ProbNode>) {
    let node = Arc::new(ProbNode::new(hnsw_level, prop, parent, child));
    let lazy_node = ProbLazyItem::from_arc(allocator, node.clone(), version_id, version_number);
    (lazy_node, node)
}

fn create_node_edges(
    dense_index: Arc<DenseIndex>,
    lazy_node: *mut ProbLazyItem<ProbNode>,
    node: Arc<ProbNode>,
    nbs: Vec<(*mut ProbLazyItem<ProbNode>, MetricResult)>,
    cur_level: HNSWLevel,
    version: Hash,
    version_number: u16,
    thread_id: u32,
) -> Result<(), WaCustomError> {
    for (nbr1_ptr, dist) in nbs.into_iter() {
        let nbr1 = unsafe { &*nbr1_ptr };
        let old_neighbor = nbr1.try_get_data(dense_index.cache.clone())?;

        let (new_lazy_neighbor, new_neighbor) = if let Some(version) =
            ProbLazyItem::get_version(nbr1_ptr, version_number, dense_index.cache.clone())?
        {
            let node = unsafe { &*version }.try_get_data(dense_index.cache.clone())?;
            (version, node)
        } else {
            let prop_arc = old_neighbor.prop.clone();
            let parent = old_neighbor.get_parent();
            let child = old_neighbor.get_child();
            let new_neighbor = Arc::new(ProbNode::new_with_neighbors(
                cur_level,
                prop_arc,
                old_neighbor.clone_neighbors(),
                parent,
                child,
            ));
            let new_lazy_neighbor = ProbLazyItem::from_arc(
                &dense_index.cache.get_allocator(),
                new_neighbor.clone(),
                version,
                version_number,
            );
            ProbLazyItem::add_version(nbr1_ptr, new_lazy_neighbor, dense_index.cache.clone())?;
            let mut exec_queue = dense_index.exec_queue_nodes.clone();
            exec_queue
                .transactional_update(|queue| {
                    let mut new_queue = queue.clone();
                    new_queue.push(nbr1_ptr.clone());
                    new_queue
                })
                .unwrap();

            (new_lazy_neighbor, new_neighbor)
        };

        new_neighbor.add_neighbor(lazy_node.clone(), node.get_id().unwrap(), dist);
        node.add_neighbor(new_lazy_neighbor, new_neighbor.get_id().unwrap(), dist);
    }

    queue_node_prop_exec(dense_index, lazy_node, node, thread_id)?;

    Ok(())
}

fn traverse_find_nearest(
    dense_index: Arc<DenseIndex>,
    vtm: *mut ProbLazyItem<ProbNode>,
    fvec: Arc<Storage>,
    hops: u8,
    skipm: &mut HashSet<VectorId>,
    cur_level: HNSWLevel,
    skip_hop: bool,
) -> Result<Vec<(*mut ProbLazyItem<ProbNode>, MetricResult)>, WaCustomError> {
    let mut tasks: SmallVec<[Vec<(*mut ProbLazyItem<ProbNode>, MetricResult)>; 24]> =
        SmallVec::new();

    let node = unsafe { &*ProbLazyItem::get_latest_version(vtm, dense_index.cache.clone())?.0 }
        .try_get_data(dense_index.cache.clone())?;

    for (index, (nref, _)) in node.get_neighbors().into_iter().enumerate() {
        if let Some(neighbor) = unsafe { &*nref }.get_lazy_data() {
            let mut prop_arc = neighbor.prop.clone();
            let prop_state = prop_arc.get();

            let node_prop = match prop_state {
                PropState::Ready(prop) => prop.clone(),
                PropState::Pending(loc) => {
                    return Err(WaCustomError::NodeError(format!(
                        "Neighbor prop is in pending state at loc: {:?}",
                        loc
                    )))
                }
            };

            let nb = node_prop.id.clone();

            if index % 2 != 0 && skip_hop && index > 4 {
                continue;
            }

            let dense_index = dense_index.clone();
            let fvec = fvec.clone();

            if skipm.insert(nb.clone()) {
                let dist = dense_index
                    .distance_metric
                    .calculate(&fvec, &node_prop.value)?;

                let full_hops = 30;
                if hops
                    <= tapered_total_hops(
                        full_hops,
                        cur_level.0,
                        dense_index.hnsw_params.clone().get().num_layers,
                    )
                {
                    let mut z = traverse_find_nearest(
                        dense_index.clone(),
                        nref,
                        fvec.clone(),
                        hops + 1,
                        skipm,
                        cur_level,
                        skip_hop,
                    )?;
                    z.push((nref, dist));
                    tasks.push(z);
                } else {
                    tasks.push(vec![(nref, dist)]);
                }
            }
        }
    }

    let mut nn: Vec<_> = tasks.into_iter().flatten().collect();
    nn.sort_by(|a, b| b.1.get_value().partial_cmp(&a.1.get_value()).unwrap());
    let mut seen = HashSet::new();
    nn.retain(|(lazy_node, _)| {
        unsafe { &**lazy_node }
            .get_lazy_data()
            .and_then(|node| node.get_id())
            .map_or(false, |id| seen.insert(id))
    });

    Ok(nn.into_iter().take(5).collect())
}

pub fn create_index_in_collection(dense_index: Arc<DenseIndex>) -> Result<(), WaCustomError> {
    let collection_path: Arc<Path> = Path::new(&dense_index.database_name).into();

    let mut quantization_metric_arc = dense_index.quantization_metric.clone();
    let quantization_metric = quantization_metric_arc.get();

    let prop_file = Arc::new(
        OpenOptions::new()
            .create(true)
            .truncate(true) // removes all the previous data from the file
            .append(true)
            .open(collection_path.join("prop.data"))
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
    );

    let hash = unsafe { &*dense_index.get_root_vec() }.get_current_version();

    let root = create_root_node(
        &dense_index.cache.get_allocator(),
        dense_index.hnsw_params.num_layers,
        quantization_metric,
        *dense_index.storage_type.clone().get(),
        dense_index.dim,
        prop_file,
        hash,
        dense_index.index_manager.clone(),
    )?;

    // The whole index is empty now
    dense_index.set_root_vec(root);
    dense_index.set_auto_config_flag(false);
    dense_index.set_configured_flag(true);

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

#[cfg(test)]
mod tests {
    use super::{read_embedding, write_embedding, RawVectorEmbedding};
    use crate::models::{buffered_io::BufferManager, types::VectorId};
    use rand::{distributions::Uniform, rngs::ThreadRng, thread_rng, Rng};
    use std::sync::Arc;
    use tempfile::tempfile;

    fn get_random_embedding(rng: &mut ThreadRng) -> RawVectorEmbedding {
        let range = Uniform::new(-1.0, 1.0);

        let raw_vec: Vec<f32> = (0..rng.gen_range(100..200))
            .into_iter()
            .map(|_| rng.sample(&range))
            .collect();

        RawVectorEmbedding {
            raw_vec,
            hash_vec: VectorId::Int(rng.gen()),
        }
    }

    #[test]
    fn test_embedding_serialization() {
        let mut rng = thread_rng();
        let embedding = get_random_embedding(&mut rng);
        let tempfile = tempfile().unwrap();

        let bufman = Arc::new(BufferManager::new(tempfile).unwrap());
        let offset = write_embedding(bufman.clone(), &embedding).unwrap();

        let (deserialized, _) = read_embedding(bufman.clone(), offset).unwrap();

        assert_eq!(embedding, deserialized);
    }

    #[test]
    fn test_embeddings_serialization() {
        let mut rng = thread_rng();
        let embeddings: Vec<_> = (0..20).map(|_| get_random_embedding(&mut rng)).collect();
        let tempfile = tempfile().unwrap();

        let bufman = Arc::new(BufferManager::new(tempfile).unwrap());

        for embedding in &embeddings {
            write_embedding(bufman.clone(), embedding).unwrap();
        }

        let mut offset = 0;

        for embedding in embeddings {
            let (deserialized, next) = read_embedding(bufman.clone(), offset).unwrap();
            offset = next;

            assert_eq!(embedding, deserialized);
        }
    }
}
