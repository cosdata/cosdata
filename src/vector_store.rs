use crate::distance::DistanceFunction;
use crate::macros::key;
use crate::models::buffered_io::BufferManager;
use crate::models::common::*;
use crate::models::file_persist::*;
use crate::models::identity_collections::IdentitySet;
use crate::models::kmeans::{concat_vectors, generate_initial_centroids, kmeans, should_continue};
use crate::models::lazy_load::*;
use crate::models::meta_persist::store_current_version;
use crate::models::types::*;
use crate::models::versioning::Hash;
use crate::quantization::{Quantization, StorageType};
use crate::storage::Storage;
use arcshift::ArcShift;
use lmdb::{Transaction, WriteFlags};
use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use smallvec::SmallVec;
use std::array::TryFromSliceError;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::SeekFrom;
use std::path::Path;
use std::sync::Arc;

pub fn ann_search(
    vec_store: Arc<VectorStore>,
    vector_emb: QuantizedVectorEmbedding,
    cur_entry: LazyItem<MergedNode>,
    cur_level: HNSWLevel,
) -> Result<Option<Vec<(LazyItem<MergedNode>, MetricResult)>>, WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = HashSet::new();
    skipm.insert(vector_emb.hash_vec.clone());

    let cur_node = cur_entry.get_data(vec_store.cache.clone());

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
        vec_store.clone(),
        cur_entry.clone(),
        fvec.clone(),
        0,
        &mut skipm,
        cur_level,
        false,
    )?;

    let dist = vec_store
        .distance_metric
        .calculate(&fvec, &node_prop.value)?;

    let z = if z.is_empty() {
        vec![(cur_entry.clone(), dist)]
    } else {
        z
    };

    let result = if cur_level.0 == 0 {
        Some(vec![])
    } else {
        ann_search(
            vec_store,
            vector_emb,
            z[0].0.clone(),
            HNSWLevel(cur_level.0 - 1),
        )?
    };

    Ok(add_option_vecs(&result, &Some(z)))
}

pub fn vector_fetch(
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Result<Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>>, WaCustomError> {
    let mut results = Vec::new();

    for lev in 0..=vec_store.hnsw_params.clone().get().num_layers {
        let maybe_res = load_vector_id_lsmdb(HNSWLevel(lev), vector_id.clone());
        let neighbors = maybe_res
            .try_get_data(vec_store.cache.clone())
            .ok()
            .and_then(|data| {
                let id = get_vector_id_from_node(data.clone())?;

                Some((
                    id,
                    data.neighbors
                        .iter()
                        .filter_map(|ne| {
                            let data = ne.1.try_get_data(vec_store.cache.clone()).ok()?;
                            Some((get_vector_id_from_node(data)?, ne.0))
                        })
                        .collect(),
                ))
            });

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
/// * `vec_store` - An `Arc`-wrapped `VectorStore` instance, which contains the LMDB environment and database for embeddings.
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
/// use my_crate::{VectorStore, get_embedding_by_id, RawVectorEmbedding, WaCustomError, VectorId};
///
/// let vec_store = Arc::new(VectorStore::new());
/// let vector_id = VectorId::Int(42); // Example vector ID
/// match get_embedding_by_id(vec_store.clone(), vector_id) {
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
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Result<RawVectorEmbedding, WaCustomError> {
    let env = vec_store.lmdb.env.clone();
    let db = vec_store.lmdb.db.clone();

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
    let bufman = vec_store.vec_raw_manager.get(&current_version)?;
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

fn auto_config_storage_type(vec_store: Arc<VectorStore>, vectors: &[&[f32]]) {
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
            StorageType::SubByte(3)
        }
    } else {
        StorageType::UnsignedByte
    };

    vec_store.storage_type.update_shared(storage_type);
}

pub fn insert_embedding(
    bufman: Arc<BufferManager>,
    vec_store: Arc<VectorStore>,
    emb: &RawVectorEmbedding,
    current_version: Hash,
) -> Result<(), WaCustomError> {
    let env = vec_store.lmdb.env.clone();
    let db = vec_store.lmdb.db.clone();

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
    vec_store: Arc<VectorStore>,
    upload_process_batch_size: usize,
) -> Result<(), WaCustomError> {
    let env = vec_store.lmdb.env.clone();
    let db = vec_store.lmdb.db.clone();

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
    let version_hash = vec_store
        .vcs
        .get_version_hash(&version)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?
        .expect("Current version hash not found");
    let version_number = *version_hash.version as u16;

    txn.abort();

    let mut index = |embeddings: Vec<RawVectorEmbedding>,
                     next_offset: u32|
     -> Result<(), WaCustomError> {
        let mut quantization_arc = vec_store.quantization_metric.clone();
        // Set to auto config but is not configured
        if vec_store.get_auto_config_flag() && !vec_store.get_configured_flag() {
            let quantization = quantization_arc.get();
            let mut new_quantization = quantization.clone();
            let vectors: Vec<&[f32]> = embeddings
                .iter()
                .map(|embedding| &embedding.raw_vec as &[f32])
                .collect();
            new_quantization.train(&vectors)?;
            quantization_arc.update(new_quantization);
            auto_config_storage_type(vec_store.clone(), &vectors);
            vec_store.set_configured_flag(true);
        }
        let quantization = quantization_arc.get();
        let results: Vec<()> = embeddings
            .into_par_iter()
            .map(|raw_emb| {
                let lp = &vec_store.levels_prob;
                let iv = get_max_insert_level(rand::random::<f32>().into(), lp.clone());
                let quantized_vec = Arc::new(
                    quantization
                        .quantize(
                            &raw_emb.raw_vec,
                            vec_store.storage_type.clone().get().clone(),
                        )
                        .expect("Quantization failed"),
                );
                let prop = ArcShift::new(PropState::Ready(Arc::new(NodeProp {
                    id: raw_emb.hash_vec.clone(),
                    value: quantized_vec.clone(),
                    location: None,
                })));
                let embedding = QuantizedVectorEmbedding {
                    quantized_vec,
                    hash_vec: raw_emb.hash_vec,
                };

                index_embedding(
                    vec_store.clone(),
                    None,
                    embedding,
                    prop,
                    vec_store.root_vec.item.clone().get().clone(),
                    HNSWLevel(vec_store.hnsw_params.clone().get().num_layers),
                    HNSWLevel(iv.try_into().unwrap()),
                    version,
                    version_number,
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

    let bufman = vec_store.vec_raw_manager.get(&version)?;

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

pub fn index_embedding(
    vec_store: Arc<VectorStore>,
    parent: Option<LazyItem<MergedNode>>,
    vector_emb: QuantizedVectorEmbedding,
    prop: ArcShift<PropState>,
    cur_entry: LazyItem<MergedNode>,
    cur_level: HNSWLevel,
    max_insert_level: HNSWLevel,
    version: Hash,
    version_number: u16,
) -> Result<(), WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = HashSet::new();
    skipm.insert(vector_emb.hash_vec.clone());

    let cur_node = cur_entry
        .get_latest_version(vec_store.cache.clone())
        .0
        .get_data(vec_store.cache.clone());

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
        vec_store.clone(),
        cur_entry.clone(),
        fvec.clone(),
        0,
        &mut skipm,
        cur_level,
        true,
    )?;

    // @DOUBT: May be this distance can be calculated only if z is
    // empty
    let dist = vec_store
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

    let parent = if cur_level.0 <= max_insert_level.0 {
        let parent = insert_node_create_edges(
            vec_store.clone(),
            parent,
            prop.clone(),
            z,
            cur_level,
            version,
            version_number,
        )
        .expect("Failed insert_node_create_edges");

        Some(parent)
    } else {
        None
    };

    if cur_level.0 != 0 {
        index_embedding(
            vec_store.clone(),
            parent,
            vector_emb.clone(),
            prop,
            z_clone[0].clone(),
            HNSWLevel(cur_level.0 - 1),
            max_insert_level,
            version,
            version_number,
        )?;
    }

    Ok(())
}

pub fn queue_node_prop_exec(
    vec_store: Arc<VectorStore>,
    lznode: LazyItem<MergedNode>,
) -> Result<(), WaCustomError> {
    let (node, _location) = (
        lznode.try_get_data(vec_store.cache.clone())?,
        lznode
            .get_file_index()
            .ok_or(WaCustomError::LazyLoadingError(
                "Missing FileIndex".to_string(),
            ))?,
    );

    if let PropState::Ready(prop) = node.get_prop() {
        if prop.location.is_none() {
            let location = write_prop_to_file(&prop.id, prop.value.clone(), &vec_store.prop_file)?;
            node.set_prop_location(location);
        }
    };

    for neighbor in node.neighbors.iter() {
        neighbor.1.set_persistence(true);
    }

    // Add the node to exec_queue_nodes
    let mut exec_queue = vec_store.exec_queue_nodes.clone();
    exec_queue
        .transactional_update(|queue| {
            let mut new_queue = queue.clone();
            new_queue.push(ArcShift::new(lznode.clone()));
            new_queue
        })
        .unwrap();

    Ok(())
}

pub fn auto_commit_transaction(vec_store: Arc<VectorStore>) -> Result<(), WaCustomError> {
    // Retrieve exec_queue_nodes from vec_store
    let mut exec_queue_nodes_arc = vec_store.exec_queue_nodes.clone();
    let mut exec_queue_nodes = exec_queue_nodes_arc.get().clone();

    for node in exec_queue_nodes.iter_mut() {
        persist_node_update_loc(vec_store.index_manager.clone(), node)?;
    }

    exec_queue_nodes_arc.update(Vec::new());

    Ok(())
}

fn create_node_extract_neighbours(
    version_id: Hash,
    version_number: u16,
    hnsw_level: HNSWLevel,
    prop: ArcShift<PropState>,
    parent: LazyItemRef<MergedNode>,
    child: LazyItemRef<MergedNode>,
) -> (
    LazyItem<MergedNode>,
    EagerLazyItemSet<MergedNode, MetricResult>,
) {
    let neighbours = EagerLazyItemSet::new();
    let node = LazyItem::from_data(
        version_id,
        version_number,
        MergedNode {
            hnsw_level,
            prop,
            neighbors: neighbours.clone(),
            parent,
            child,
        },
    );
    (node, neighbours)
}

fn insert_node_create_edges(
    vec_store: Arc<VectorStore>,
    parent: Option<LazyItem<MergedNode>>,
    prop: ArcShift<PropState>,
    nbs: Vec<(LazyItem<MergedNode>, MetricResult)>,
    cur_level: HNSWLevel,
    version: Hash,
    version_number: u16,
) -> Result<LazyItem<MergedNode>, WaCustomError> {
    let (node, neighbours) = create_node_extract_neighbours(
        version,
        version_number,
        cur_level,
        prop,
        parent.clone().map_or_else(
            || LazyItemRef::new_invalid(),
            |parent| LazyItemRef::from_lazy(parent),
        ),
        LazyItemRef::new_invalid(),
    );
    if let Some(parent) = parent {
        parent
            .get_lazy_data()
            .unwrap()
            .get()
            .clone()
            .unwrap()
            .set_child(node.clone());
    }

    for (nbr1, dist) in nbs.into_iter() {
        if let Ok(old_neighbour) = nbr1.try_get_data(vec_store.cache.clone()) {
            let (new_neighbor, mut new_neighbor_neighbors, mut neighbor_list) =
                if let Some(version) = nbr1.get_version(vec_store.cache.clone(), version_number) {
                    let node = version.get_data(vec_store.cache.clone());
                    let neighbor_list: Vec<(LazyItem<MergedNode>, MetricResult)> =
                        node.neighbors.iter().map(|nbr2| (nbr2.1, nbr2.0)).collect();
                    (version, node.neighbors.clone(), neighbor_list)
                } else {
                    let prop_arc = old_neighbour.prop.clone();
                    let parent = old_neighbour.parent.clone();
                    let child = old_neighbour.child.clone();
                    let (new_neighbour, new_neighbour_neighbours) = create_node_extract_neighbours(
                        version,
                        version_number,
                        cur_level,
                        prop_arc,
                        parent,
                        child,
                    );
                    nbr1.add_version(vec_store.cache.clone(), new_neighbour.clone());
                    let neighbor_list: Vec<(LazyItem<MergedNode>, MetricResult)> = old_neighbour
                        .neighbors
                        .iter()
                        .map(|nbr2| (nbr2.1, nbr2.0))
                        .collect();
                    let mut exec_queue = vec_store.exec_queue_nodes.clone();
                    exec_queue
                        .transactional_update(|queue| {
                            let mut new_queue = queue.clone();
                            new_queue.push(ArcShift::new(nbr1.clone()));
                            new_queue
                        })
                        .unwrap();

                    (new_neighbour, new_neighbour_neighbours, neighbor_list)
                };

            neighbor_list.push((node.clone(), dist));

            neighbor_list.sort_by(|a, b| {
                b.1.get_value()
                    .partial_cmp(&a.1.get_value())
                    .unwrap_or(Ordering::Equal)
            });

            neighbor_list.truncate(20);
            let new_neighbour_neighbours_set = IdentitySet::from_iter(
                neighbor_list
                    .into_iter()
                    .map(|(node, dist)| EagerLazyItem(dist, node)),
            );
            new_neighbor_neighbors
                .items
                .update(new_neighbour_neighbours_set);
            neighbours.insert(EagerLazyItem(dist, new_neighbor));
        }
    }

    queue_node_prop_exec(vec_store, node.clone())?;

    Ok(node)
}

fn traverse_find_nearest(
    vec_store: Arc<VectorStore>,
    vtm: LazyItem<MergedNode>,
    fvec: Arc<Storage>,
    hops: u8,
    skipm: &mut HashSet<VectorId>,
    cur_level: HNSWLevel,
    skip_hop: bool,
) -> Result<Vec<(LazyItem<MergedNode>, MetricResult)>, WaCustomError> {
    let mut tasks: SmallVec<[Vec<(LazyItem<MergedNode>, MetricResult)>; 24]> = SmallVec::new();

    let node = vtm
        .get_latest_version(vec_store.cache.clone())
        .0
        .get_data(vec_store.cache.clone());

    for (index, nref) in node.neighbors.iter().enumerate() {
        if let Some(mut neighbor_arc) = nref.1.get_lazy_data() {
            if let Some(neighbor) = neighbor_arc.get() {
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

                let vec_store = vec_store.clone();
                let fvec = fvec.clone();

                if skipm.insert(nb.clone()) {
                    let dist = vec_store
                        .distance_metric
                        .calculate(&fvec, &node_prop.value)?;

                    let full_hops = 30;
                    if hops
                        <= tapered_total_hops(
                            full_hops,
                            cur_level.0,
                            vec_store.hnsw_params.clone().get().num_layers,
                        )
                    {
                        let mut z = traverse_find_nearest(
                            vec_store.clone(),
                            nref.1.clone(),
                            fvec.clone(),
                            hops + 1,
                            skipm,
                            cur_level,
                            skip_hop,
                        )?;
                        z.push((nref.1.clone(), dist));
                        tasks.push(z);
                    } else {
                        tasks.push(vec![(nref.1.clone(), dist)]);
                    }
                }
            }
        }
    }

    let mut nn: Vec<_> = tasks.into_iter().flatten().collect();
    nn.sort_by(|a, b| b.1.get_value().partial_cmp(&a.1.get_value()).unwrap());
    let mut seen = HashSet::new();
    nn.retain(|(lazy_node, _)| {
        if let LazyItem::Valid { data: node_arc, .. } = &lazy_node {
            if let Some(node) = node_arc.clone().get() {
                let mut prop_arc = node.prop.clone();
                let prop_state = prop_arc.get();
                if let PropState::Ready(node_prop) = &*prop_state {
                    seen.insert(node_prop.id.clone())
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    });

    Ok(nn.into_iter().take(5).collect())
}

pub fn create_index_in_collection(vec_store: Arc<VectorStore>) -> Result<(), WaCustomError> {
    let collection_path: Arc<Path> = Path::new(&vec_store.database_name).into();

    // TODO(a-rustacean): store the min & max in the `VectorStore` in the collection creation step and use here
    let min = -1.0;
    let max = 1.0;
    let vec = (0..vec_store.dim)
        .map(|_| {
            let mut rng = thread_rng();

            let number: f32 = rng.gen_range(min..max);
            number
        })
        .collect::<Vec<f32>>();
    let vec_hash = VectorId::Int(-1);
    let mut quantization_metric_arc = vec_store.quantization_metric.clone();
    let quantization_metric = quantization_metric_arc.get();

    let vector_list =
        Arc::new(quantization_metric.quantize(&vec, *vec_store.storage_type.clone().get())?);

    let prop_file = Arc::new(
        OpenOptions::new()
            .create(true)
            .truncate(true) // removes all the previous data from the file
            .append(true)
            .open(collection_path.join("prop.data"))
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
    );

    let hash = vec_store.root_vec.get_current_version();

    let location = write_prop_to_file(&vec_hash, vector_list.clone(), &prop_file)?;

    let prop = ArcShift::new(PropState::Ready(Arc::new(NodeProp {
        id: vec_hash,
        value: vector_list,
        location: Some(location),
    })));

    let root = LazyItem::new(
        hash,
        0,
        MergedNode {
            hnsw_level: HNSWLevel(0),
            prop: prop.clone(),
            neighbors: EagerLazyItemSet::new(),
            parent: LazyItemRef::new_invalid(),
            child: LazyItemRef::new_invalid(),
        },
    );

    let mut prev: LazyItemRef<MergedNode> = LazyItemRef::from_lazy(root.clone());

    let mut nodes = Vec::new();
    for l in (1..=vec_store.hnsw_params.clone().get().num_layers).rev() {
        let current_node = Arc::new(MergedNode {
            hnsw_level: HNSWLevel(l),
            prop: prop.clone(),
            neighbors: EagerLazyItemSet::new(),
            parent: prev.clone(),
            child: LazyItemRef::new_invalid(),
        });

        let lazy_node = LazyItem::from_arc(hash, 0, current_node.clone());
        let nn = LazyItemRef::from_arc(hash, 0, current_node.clone());

        if let Some(prev_node) = prev.item.get().get_lazy_data().unwrap().get() {
            current_node.set_parent(prev.clone().item.get().clone());
            prev_node.set_child(lazy_node.clone());
        }
        prev = nn.clone();

        nodes.push(nn.clone());
    }

    for nn in nodes.iter_mut() {
        persist_node_update_loc(vec_store.index_manager.clone(), &mut nn.item)?;
    }

    vec_store.index_manager.flush_all()?;

    // The whole index is empty now
    vec_store.root_vec.item.clone().update(root);
    vec_store.set_auto_config_flag(false);
    vec_store.set_configured_flag(true);

    Ok(())
}

fn delete_node_update_neighbours(
    vec_store: Arc<VectorStore>,
    item: LazyItem<MergedNode>,
    skipm: HashSet<VectorId>,
    version_id: Hash,
    version_number: u16,
    hnsw_level: HNSWLevel,
) -> Result<(), WaCustomError> {
    let node = item.get_data(vec_store.cache.clone());
    for nbr in node.neighbors.iter() {
        let mut skipm = skipm.clone();
        let nbr_node = nbr.1.get_data(vec_store.cache.clone());
        let (nbr_id, nbr_vec) = match nbr_node.get_prop() {
            PropState::Ready(prop) => (prop.id.clone(), prop.value.clone()),
            PropState::Pending(_) => {
                // TODO: load prop
                return Err(WaCustomError::NodeError("PropState is Pending".to_string()));
            }
        };
        skipm.insert(nbr_id);
        let mut nbr_nbrs = traverse_find_nearest(
            vec_store.clone(),
            vec_store
                .root_vec
                .item
                .clone()
                .get()
                .get_latest_version(vec_store.cache.clone())
                .0,
            nbr_vec,
            0,
            &mut skipm,
            HNSWLevel(vec_store.hnsw_params.clone().get().num_layers),
            false,
        )?;

        nbr_nbrs.truncate(20);
        let nbr_nbrs_set = IdentitySet::from_iter(
            nbr_nbrs
                .into_iter()
                .map(|(node, dist)| EagerLazyItem(dist, node)),
        );

        if let Some(version) = nbr.1.get_version(vec_store.cache.clone(), version_number) {
            let nbr_current_version_data = version.get_data(vec_store.cache.clone());
            nbr_current_version_data
                .neighbors
                .items
                .clone()
                .update(nbr_nbrs_set);
        } else {
            let (new_version, mut new_neighbours) = create_node_extract_neighbours(
                version_id,
                version_number,
                hnsw_level,
                nbr_node.prop.clone(),
                nbr_node.parent.clone(),
                nbr_node.child.clone(),
            );
            new_neighbours.items.update(nbr_nbrs_set);
            nbr.1.add_version(vec_store.cache.clone(), new_version);
            let mut exec_queue = vec_store.exec_queue_nodes.clone();
            exec_queue
                .transactional_update(|queue| {
                    let mut new_queue = queue.clone();
                    new_queue.push(ArcShift::new(nbr.1.clone()));
                    new_queue
                })
                .unwrap();
        }
    }
    Ok(())
}

pub fn delete_vector_by_id(
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Result<(), WaCustomError> {
    let (current_version, current_version_number) = vec_store
        .vcs
        .add_next_version("main")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    vec_store.set_current_version(current_version);
    store_current_version(&vec_store.lmdb, current_version)?;

    let vec_raw = get_embedding_by_id(vec_store.clone(), vector_id.clone())?;
    let quantized = Arc::new(vec_store.quantization_metric.clone().get().quantize(
        &vec_raw.raw_vec,
        vec_store.storage_type.clone().get().clone(),
    )?);
    let mut skipm = HashSet::new();
    skipm.insert(vec_raw.hash_vec);
    let items = traverse_find_nearest(
        vec_store.clone(),
        vec_store
            .root_vec
            .item
            .clone()
            .get()
            .get_latest_version(vec_store.cache.clone())
            .0,
        quantized,
        0,
        &mut skipm,
        HNSWLevel(vec_store.hnsw_params.clone().get().num_layers),
        false,
    )?;

    let mut maybe_item = None;

    for (item, _) in items {
        let data = item.get_data(vec_store.cache.clone());
        let prop = match data.get_prop() {
            PropState::Ready(prop) => prop.id.clone(),
            PropState::Pending(_) => {
                // TODO: load prop
                return Err(WaCustomError::NodeError("PropState is Pending".to_string()));
            }
        };

        if prop == vector_id {
            maybe_item = Some(item);
        }
    }

    let Some(mut item) = maybe_item else {
        return Err(WaCustomError::NodeError(
            "Node not found in graph".to_string(),
        ));
    };

    for level in 0..=vec_store.hnsw_params.clone().get().num_layers {
        delete_node_update_neighbours(
            vec_store.clone(),
            item.clone(),
            skipm.clone(),
            current_version,
            *current_version_number as u16,
            HNSWLevel(level),
        )?;

        item = item
            .get_data(vec_store.cache.clone())
            .parent
            .clone()
            .item
            .get()
            .clone();
    }

    auto_commit_transaction(vec_store)?;

    Ok(())
}

pub fn delete_vector_by_id_in_transaction(
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
    transaction_id: Hash,
) -> Result<(), WaCustomError> {
    let version_hash = vec_store
        .vcs
        .get_version_hash(&transaction_id)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?
        .ok_or(WaCustomError::DatabaseError(
            "VersionHash not found for transaction".to_string(),
        ))?;
    let current_version_number = *version_hash.version as u16;

    let vec_raw = get_embedding_by_id(vec_store.clone(), vector_id.clone())?;
    let quantized = Arc::new(vec_store.quantization_metric.clone().get().quantize(
        &vec_raw.raw_vec,
        vec_store.storage_type.clone().get().clone(),
    )?);
    let mut skipm = HashSet::new();
    skipm.insert(vec_raw.hash_vec);
    let items = traverse_find_nearest(
        vec_store.clone(),
        vec_store
            .root_vec
            .item
            .clone()
            .get()
            .get_latest_version(vec_store.cache.clone())
            .0,
        quantized,
        0,
        &mut skipm,
        HNSWLevel(vec_store.hnsw_params.clone().get().num_layers),
        false,
    )?;

    let mut maybe_item = None;

    for (item, _) in items {
        let data = item.get_data(vec_store.cache.clone());
        let prop = match data.get_prop() {
            PropState::Ready(prop) => prop.id.clone(),
            PropState::Pending(_) => {
                // TODO: load prop
                return Err(WaCustomError::NodeError("PropState is Pending".to_string()));
            }
        };

        if prop == vector_id {
            maybe_item = Some(item);
        }
    }

    let Some(mut item) = maybe_item else {
        return Err(WaCustomError::NodeError(
            "Node not found in graph".to_string(),
        ));
    };

    for level in 0..=vec_store.hnsw_params.clone().get().num_layers {
        delete_node_update_neighbours(
            vec_store.clone(),
            item.clone(),
            skipm.clone(),
            transaction_id,
            current_version_number,
            HNSWLevel(level),
        )?;

        item = item
            .get_data(vec_store.cache.clone())
            .parent
            .clone()
            .item
            .get()
            .clone();
    }

    auto_commit_transaction(vec_store)?;

    Ok(())
}

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
