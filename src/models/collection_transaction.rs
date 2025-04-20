use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc, Arc,
    },
    thread,
};

use lmdb::{Transaction, WriteFlags};

use crate::{
    config_loader::Config,
    indexes::{hnsw::types::RawDenseVectorEmbedding, IndexOps},
    macros::key,
};

use super::{
    collection::Collection,
    common::{TSHashTable, WaCustomError},
    embedding_persist::{write_dense_embedding, EmbeddingOffset},
    prob_node::{ProbNode, SharedNode},
    types::VectorId,
    versioning::{Hash, Version},
};

pub struct CollectionTransaction {
    pub id: Hash,
    pub version_number: u16,
    pub lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    raw_dense_embedding_serializer_thread_handle:
        Option<thread::JoinHandle<Result<(), WaCustomError>>>,
    raw_dense_embedding_channel: Option<mpsc::Sender<RawDenseVectorEmbedding>>,
    level_0_node_offset_counter: AtomicU32,
    node_offset_counter: AtomicU32,
    node_size: u32,
    level_0_node_size: u32,
}

impl CollectionTransaction {
    pub fn new(collection: Arc<Collection>) -> Result<Self, WaCustomError> {
        let branch_info = collection.vcs.get_branch_info("main")?.unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = collection
            .vcs
            .generate_hash("main", Version::from(version_number))?;

        let level_0_node_offset_counter = AtomicU32::new(0);
        let node_offset_counter = AtomicU32::new(0);

        let (
            node_size,
            level_0_node_size,
            raw_dense_embedding_serializer_thread_handle,
            raw_dense_embedding_channel,
        ) = if let Some(hnsw_index) = &*collection.hnsw_index.read().unwrap() {
            let (raw_embedding_channel, rx) = mpsc::channel();
            let raw_embedding_serializer_thread_handle = {
                let bufman = hnsw_index.vec_raw_manager.get(id)?;
                let collection = collection.clone();

                thread::spawn(move || {
                    let mut offsets = Vec::new();
                    for raw_emb in rx {
                        let offset = write_dense_embedding(&bufman, &raw_emb)?;
                        let embedding_key = key!(e:raw_emb.hash_vec);
                        offsets.push((embedding_key, offset));
                    }

                    let env = collection.lmdb.env.clone();
                    let db = collection.lmdb.db.clone();

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
            (
                ProbNode::get_serialized_size(hnsw_params.neighbors_count) as u32,
                ProbNode::get_serialized_size(hnsw_params.level_0_neighbors_count) as u32,
                Some(raw_embedding_serializer_thread_handle),
                Some(raw_embedding_channel),
            )
        } else {
            (0, 0, None, None)
        };

        Ok(Self {
            id,
            version_number,
            level_0_node_offset_counter,
            node_offset_counter,
            node_size,
            level_0_node_size,
            lazy_item_versions_table: Arc::new(TSHashTable::new(16)),
            raw_dense_embedding_serializer_thread_handle,
            raw_dense_embedding_channel,
        })
    }

    pub fn post_raw_dense_embedding(&self, raw_emb: RawDenseVectorEmbedding) {
        if let Some(channel) = &self.raw_dense_embedding_channel {
            channel.send(raw_emb).unwrap();
        }
    }

    pub fn pre_commit(self, collection: &Collection, config: &Config) -> Result<(), WaCustomError> {
        if let Some(hnsw_index) = &*collection.hnsw_index.read().unwrap() {
            hnsw_index.pre_commit_transaction(collection, &self, config)?;
        }
        if let Some(inverted_index) = &*collection.inverted_index.read().unwrap() {
            inverted_index.pre_commit_transaction(collection, &self, config)?;
        }
        if let Some(tf_idf_index) = &*collection.tf_idf_index.read().unwrap() {
            tf_idf_index.pre_commit_transaction(collection, &self, config)?;
        }
        drop(self.raw_dense_embedding_channel);
        if let Some(handle) = self.raw_dense_embedding_serializer_thread_handle {
            handle.join().unwrap()?;
        }

        Ok(())
    }

    pub fn get_new_node_offset(&self) -> u32 {
        self.node_offset_counter
            .fetch_add(self.node_size, Ordering::Relaxed)
    }

    pub fn get_new_level_0_node_offset(&self) -> u32 {
        self.level_0_node_offset_counter
            .fetch_add(self.level_0_node_size, Ordering::Relaxed)
    }
}
