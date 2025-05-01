use std::{
    fs,
    sync::{mpsc, Arc},
    thread::{self, JoinHandle},
};

use crate::config_loader::Config;

use super::{
    buffered_io::BufIoError,
    collection::{Collection, IndexingState},
    collection_transaction::BackgroundCollectionTransaction,
    common::WaCustomError,
    meta_persist::update_background_version,
    versioning::Hash,
    wal::{VectorOp, WALFile},
};

pub struct IndexingManager {
    thread: Option<JoinHandle<Result<(), WaCustomError>>>,
    channel: mpsc::Sender<(Hash, u16)>,
}

impl IndexingManager {
    pub fn new(collection: Arc<Collection>, config: Config) -> Self {
        let (sender, receiver) = mpsc::channel();

        let thread = thread::spawn(move || {
            for (version_id, version_number) in receiver {
                *collection.indexing_state.write() = IndexingState::Indexing;
                let txn = BackgroundCollectionTransaction::from_version_id_and_number(
                    collection.clone(),
                    version_id,
                    version_number,
                )?;
                let wal = WALFile::new(&collection.get_path(), version_id)?;
                thread::scope(|s| {
                    while let Some(op) = wal.read()? {
                        match op {
                            VectorOp::Upsert(embeddings) => {
                                s.spawn(|| collection.index_embeddings(embeddings, &txn, &config));
                            }
                            VectorOp::Delete(_vector_id) => unimplemented!(),
                        }
                    }
                    Ok::<_, WaCustomError>(())
                })?;
                txn.pre_commit(&collection, &config)?;
                update_background_version(&collection.lmdb, version_id)?;
                *collection.indexing_state.write() = IndexingState::Completed;
                fs::remove_file(collection.get_path().join(format!("{}.wal", *version_id)))
                    .map_err(BufIoError::Io)?;
            }
            Ok(())
        });

        Self {
            thread: Some(thread),
            channel: sender,
        }
    }

    pub fn trigger(&self, version_id: Hash, version_number: u16) {
        self.channel.send((version_id, version_number)).unwrap()
    }
}

impl Drop for IndexingManager {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
