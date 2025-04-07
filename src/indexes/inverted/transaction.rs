use std::{
    sync::{mpsc, Arc},
    thread,
};

use lmdb::{Transaction, WriteFlags};

use crate::{
    macros::key,
    models::{
        common::WaCustomError,
        embedding_persist::{write_sparse_embedding, EmbeddingOffset},
        versioning::{Hash, Version},
    },
};

use super::{types::RawSparseVectorEmbedding, InvertedIndex};

pub struct InvertedIndexTransaction {
    pub id: Hash,
    pub version_number: u16,
    raw_embedding_serializer_thread_handle: thread::JoinHandle<Result<(), WaCustomError>>,
    pub raw_embedding_channel: mpsc::Sender<RawSparseVectorEmbedding>,
}

impl InvertedIndexTransaction {
    pub fn new(inverted_index: Arc<InvertedIndex>) -> Result<Self, WaCustomError> {
        let branch_info = inverted_index
            .vcs
            .get_branch_info("main")
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get main branch info: {}", err))
            })?
            .unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = inverted_index
            .vcs
            .generate_hash("main", Version::from(version_number))
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get transaction hash: {}", err))
            })?;

        let (raw_embedding_channel, rx) = mpsc::channel();

        let raw_embedding_serializer_thread_handle = {
            let bufman = inverted_index.vec_raw_manager.get(id)?;

            thread::spawn(move || {
                let mut offsets = Vec::new();
                for raw_emb in rx {
                    let offset = write_sparse_embedding(&bufman, &raw_emb)?;
                    let embedding_key = key!(e:raw_emb.hash_vec);
                    offsets.push((embedding_key, offset));
                }

                let env = inverted_index.lmdb.env.clone();
                let db = inverted_index.lmdb.db.clone();

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

        Ok(Self {
            id,
            raw_embedding_channel,
            raw_embedding_serializer_thread_handle,
            version_number: version_number as u16,
        })
    }

    pub fn post_raw_embedding(&self, raw_emb: RawSparseVectorEmbedding) {
        self.raw_embedding_channel.send(raw_emb).unwrap();
    }

    pub fn pre_commit(self, inverted_index: &InvertedIndex) -> Result<(), WaCustomError> {
        drop(self.raw_embedding_channel);
        inverted_index.root.serialize()?;
        inverted_index.root.cache.dim_bufman.flush()?;
        inverted_index.root.cache.data_bufmans.flush_all()?;
        self.raw_embedding_serializer_thread_handle
            .join()
            .unwrap()?;
        Ok(())
    }
}
