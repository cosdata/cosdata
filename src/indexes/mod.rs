use std::{hash::Hasher, sync::RwLock};

use lmdb::{Transaction, WriteFlags};
use siphasher::sip::SipHasher24;

use crate::{
    config_loader::Config,
    models::{
        collection::Collection, collection_transaction::CollectionTransaction,
        common::WaCustomError, types::MetaDb,
    },
};

pub(crate) mod hnsw;
pub(crate) mod inverted;
pub(crate) mod tf_idf;

pub trait IndexOps {
    type InputEmbedding;
    type Data: serde::Serialize + serde::de::DeserializeOwned;

    fn run_upload(
        &self,
        collection: &Collection,
        embeddings: Vec<Self::InputEmbedding>,
        transaction: &CollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        let Some(embeddings) = self.sample_embeddings(&collection.lmdb, embeddings, config)? else {
            return Ok(());
        };

        self.index_embeddings(collection, embeddings, transaction, config)
    }

    fn index_embeddings(
        &self,
        collection: &Collection,
        embeddings: Vec<Self::InputEmbedding>,
        transaction: &CollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError>;

    fn force_index(
        &self,
        collection: &Collection,
        transaction: &CollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        if !self.is_configured() {
            let mut embeddings_guard = self.embeddings_collected().write().unwrap();
            self.finalize_sampling(&collection.lmdb, config, &embeddings_guard)?;
            let embeddings = std::mem::take(&mut *embeddings_guard);
            self.index_embeddings(collection, embeddings, transaction, config)?;
        }
        Ok(())
    }

    fn sample_embeddings(
        &self,
        lmdb: &MetaDb,
        sample_embeddings: Vec<Self::InputEmbedding>,
        config: &Config,
    ) -> Result<Option<Vec<Self::InputEmbedding>>, WaCustomError> {
        if self.is_configured() {
            return Ok(Some(sample_embeddings));
        }

        let collected_count = self.increment_collected_count(sample_embeddings.len());
        let sample_threshold = self.sample_threshold();

        if collected_count < sample_threshold {
            for embedding in &sample_embeddings {
                self.sample_embedding(embedding);
            }

            let mut collected_embeddings = self.embeddings_collected().write().unwrap();
            collected_embeddings.extend(sample_embeddings);
            if collected_embeddings.len() < sample_threshold {
                return Ok(None);
            }

            self.finalize_sampling(lmdb, config, &collected_embeddings)?;

            Ok(Some(std::mem::take(&mut *collected_embeddings)))
        } else {
            while !self.is_configured() {
                drop(self.embeddings_collected().read().unwrap());
            }
            Ok(Some(sample_embeddings))
        }
    }

    fn finalize_sampling(
        &self,
        lmdb: &MetaDb,
        config: &Config,
        embeddings: &[Self::InputEmbedding],
    ) -> Result<(), WaCustomError>;

    fn sample_embedding(&self, embedding: &Self::InputEmbedding);

    fn embeddings_collected(&self) -> &RwLock<Vec<Self::InputEmbedding>>;

    fn increment_collected_count(&self, count: usize) -> usize;

    fn sample_threshold(&self) -> usize;

    // is this index configured? true is the sampling is done
    fn is_configured(&self) -> bool;

    // save everything to disk
    fn flush(&self, collection: &Collection) -> Result<(), WaCustomError>;

    fn pre_commit_transaction(
        &self,
        collection: &Collection,
        transaction: &CollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        self.force_index(collection, transaction, config)?;
        self.flush(collection)
    }

    fn get_key_for_name(name: &str) -> u64 {
        let mut hasher = SipHasher24::new();
        hasher.write(name.as_bytes());
        hasher.finish()
    }

    fn get_data(&self) -> Self::Data;

    fn persist(
        &self,
        collection_name: &str,
        env: &lmdb::Environment,
        db: lmdb::Database,
    ) -> Result<(), WaCustomError> {
        let data = self.get_data();
        let key = Self::get_key_for_name(collection_name).to_le_bytes();
        let val = serde_cbor::to_vec(&data)
            .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

        let mut txn = env.begin_rw_txn()?;
        txn.put(db, &key, &val, WriteFlags::empty())?;
        txn.commit()?;
        Ok(())
    }

    fn load_data(
        env: &lmdb::Environment,
        db: lmdb::Database,
        collection_name: &str,
    ) -> Result<Option<Self::Data>, WaCustomError> {
        let txn = env.begin_ro_txn()?;
        let key = Self::get_key_for_name(collection_name).to_be_bytes();
        let data_bytes = match txn.get(db, &key) {
            Ok(bytes) => Ok(bytes),
            Err(lmdb::Error::NotFound) => return Ok(None),
            Err(err) => Err(err),
        }?;
        let data = serde_cbor::from_slice(data_bytes)
            .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

        Ok(data)
    }

    fn delete(
        env: &lmdb::Environment,
        db: lmdb::Database,
        collection_name: &str,
    ) -> Result<(), WaCustomError> {
        let key = Self::get_key_for_name(collection_name).to_le_bytes();
        let mut txn = env.begin_rw_txn()?;
        txn.del(db, &key, None)?;
        txn.commit()?;
        Ok(())
    }
}
