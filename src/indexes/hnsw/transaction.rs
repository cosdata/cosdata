use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc, Arc,
    },
    thread,
};

use lmdb::{Transaction, WriteFlags};

use crate::{
    macros::key,
    models::{
        common::{TSHashTable, WaCustomError},
        embedding_persist::{write_dense_embedding, EmbeddingOffset},
        prob_node::{ProbNode, SharedNode},
        types::VectorId,
        versioning::{Hash, Version},
    },
};

use super::{types::RawDenseVectorEmbedding, HNSWIndex};

pub struct HNSWIndexTransaction {
    pub id: Hash,
    pub version_number: u16,
    pub lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    raw_embedding_serializer_thread_handle: thread::JoinHandle<Result<(), WaCustomError>>,
    pub raw_embedding_channel: mpsc::Sender<RawDenseVectorEmbedding>,
    level_0_node_offset_counter: AtomicU32,
    node_offset_counter: AtomicU32,
    node_size: u32,
    level_0_node_size: u32,
}

unsafe impl Send for HNSWIndexTransaction {}
unsafe impl Sync for HNSWIndexTransaction {}

impl HNSWIndexTransaction {
    pub fn new(hnsw_index: Arc<HNSWIndex>) -> Result<Self, WaCustomError> {
        let branch_info = hnsw_index
            .vcs
            .get_branch_info("main")
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get main branch info: {}", err))
            })?
            .unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = hnsw_index
            .vcs
            .generate_hash("main", Version::from(version_number))
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get transaction hash: {}", err))
            })?;

        let (raw_embedding_channel, rx) = mpsc::channel();

        let raw_embedding_serializer_thread_handle = {
            let bufman = hnsw_index.vec_raw_manager.get(id)?;
            let hnsw_index = hnsw_index.clone();

            thread::spawn(move || {
                let mut offsets = Vec::new();
                for raw_emb in rx {
                    let offset = write_dense_embedding(bufman.clone(), &raw_emb)?;
                    let embedding_key = key!(e:raw_emb.hash_vec);
                    offsets.push((embedding_key, offset));
                }

                let env = hnsw_index.lmdb.env.clone();
                let db = hnsw_index.lmdb.db.clone();

                let mut txn = env.begin_rw_txn().map_err(|e| {
                    WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e))
                })?;
                for (key, offset) in offsets {
                    let offset = EmbeddingOffset {
                        version: id,
                        offset,
                    };
                    let offset_serialized = offset.serialize();

                    txn.put(*db, &key, &offset_serialized, WriteFlags::empty())
                        .map_err(|e| {
                            WaCustomError::DatabaseError(format!("Failed to put data: {}", e))
                        })?;
                }

                txn.commit().map_err(|e| {
                    WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
                })?;
                bufman.flush()?;
                Ok(())
            })
        };

        let hnsw_params = hnsw_index.hnsw_params.read().unwrap();

        Ok(Self {
            id,
            lazy_item_versions_table: Arc::new(TSHashTable::new(16)),
            raw_embedding_channel,
            raw_embedding_serializer_thread_handle,
            version_number: version_number as u16,
            node_offset_counter: AtomicU32::new(0),
            level_0_node_offset_counter: AtomicU32::new(0),
            node_size: ProbNode::get_serialized_size(hnsw_params.neighbors_count) as u32,
            level_0_node_size: ProbNode::get_serialized_size(hnsw_params.level_0_neighbors_count)
                as u32,
        })
    }

    pub fn post_raw_embedding(&self, raw_emb: RawDenseVectorEmbedding) {
        self.raw_embedding_channel.send(raw_emb).unwrap();
    }

    pub fn pre_commit(self, hnsw_index: Arc<HNSWIndex>) -> Result<(), WaCustomError> {
        hnsw_index.cache.flush_all()?;
        drop(self.raw_embedding_channel);
        self.raw_embedding_serializer_thread_handle
            .join()
            .unwrap()?;
        Ok(())
    }

    pub fn get_new_node_offset(&self) -> u32 {
        self.node_offset_counter
            .fetch_add(self.node_size, Ordering::SeqCst)
    }

    pub fn get_new_level_0_node_offset(&self) -> u32 {
        self.level_0_node_offset_counter
            .fetch_add(self.level_0_node_size, Ordering::SeqCst)
    }
}
