use rayon::prelude::*;

use std::{
    fs::OpenOptions,
    hash::{Hash, Hasher},
    sync::{Arc, RwLock},
};

use siphasher::sip::SipHasher24;

use crate::{
    config_loader::Config,
    models::{
        buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
        collection::{Collection, RawVectorEmbedding},
        common::WaCustomError,
        paths::get_data_path,
        serializer::{CborDeserialize, CborSerialize},
        tree_map::{TreeMap, TreeMapKey},
        types::{DocumentId, InternalId, MetaDb, VectorId},
        versioning::VersionNumber,
    },
};

pub(crate) mod hnsw;
pub(crate) mod inverted;
pub(crate) mod key_value;
pub(crate) mod tf_idf;
pub(crate) mod usv;

#[derive(PartialEq, Eq, Hash)]
pub enum IndexType {
    Hnsw,
    Inverted,
    TfIdf,
    KeyValue,
    Usv,
}

pub type InternalSearchResult = (
    InternalId,
    Option<VectorId>,
    Option<DocumentId>,
    f32,
    Option<String>,
);

pub type SearchResult = (VectorId, Option<DocumentId>, f32, Option<String>);

pub trait IndexOps: Send + Sync {
    type IndexingInput: Send + Sync;
    type SearchInput: Send + Sync;
    type SearchOptions: Send + Sync;

    fn validate_embedding(&self, embedding: Self::IndexingInput) -> Result<(), WaCustomError>;

    fn run_upload(
        &self,
        collection: &Collection,
        embeddings: Vec<Self::IndexingInput>,
        version: VersionNumber,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        let Some(embeddings) = self.sample_embeddings(&collection.lmdb, embeddings, config)? else {
            return Ok(());
        };

        self.index_embeddings(collection, embeddings, version, config)
    }

    fn index_embeddings(
        &self,
        collection: &Collection,
        embeddings: Vec<Self::IndexingInput>,
        version: VersionNumber,
        config: &Config,
    ) -> Result<(), WaCustomError>;

    fn delete_embedding(
        &self,
        id: InternalId,
        raw_emb: &RawVectorEmbedding,
        version: VersionNumber,
        config: &Config,
    ) -> Result<(), WaCustomError>;

    fn force_index(
        &self,
        collection: &Collection,
        version: VersionNumber,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        if !self.is_configured() {
            let mut embeddings_guard = self.embeddings_collected().write().unwrap();
            self.finalize_sampling(&collection.lmdb, config, &embeddings_guard)?;
            let embeddings = std::mem::take(&mut *embeddings_guard);
            self.index_embeddings(collection, embeddings, version, config)?;
        }
        Ok(())
    }

    fn sample_embeddings(
        &self,
        lmdb: &MetaDb,
        sample_embeddings: Vec<Self::IndexingInput>,
        config: &Config,
    ) -> Result<Option<Vec<Self::IndexingInput>>, WaCustomError> {
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
        embeddings: &[Self::IndexingInput],
    ) -> Result<(), WaCustomError>;

    fn sample_embedding(&self, embedding: &Self::IndexingInput);

    fn embeddings_collected(&self) -> &RwLock<Vec<Self::IndexingInput>>;

    fn increment_collected_count(&self, count: usize) -> usize;

    fn sample_threshold(&self) -> usize;

    // is this index configured? true if the sampling is done
    fn is_configured(&self) -> bool;

    // save everything to disk
    fn flush(&self, collection: &Collection, version: VersionNumber) -> Result<(), WaCustomError>;

    fn pre_commit_transaction(
        &self,
        collection: &Collection,
        version: VersionNumber,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        self.force_index(collection, version, config)?;
        self.flush(collection, version)
    }

    fn get_key_for_name(name: &str) -> u64 {
        let mut hasher = SipHasher24::new();
        hasher.write(name.as_bytes());
        hasher.finish()
    }

    fn get_data(&self) -> Option<IndexData>;

    fn persist(
        &self,
        index_data_map: &IndexDataMap,
        collection_name: &str,
    ) -> Result<(), WaCustomError> {
        if let Some(data) = self.get_data() {
            index_data_map
                .insert(collection_name, data)
                .map_err(|e| WaCustomError::BufIo(Arc::new(e)))?;
        }
        Ok(())
    }

    // fn load_data<'a>(
    //     index_data_map: &'a IndexDataMap,
    //     collection_name: &str,
    //     index_type: IndexType,
    // ) -> Option<&'a IndexData> {
    //     index_data_map.get(collection_name, index_type)
    // }

    // // @TODO: To be implemented in terms of TreeMap
    // fn delete(
    //     index_data_map: &'a IndexDataMap,
    //     collection_name: &str,
    //     index_type: IndexType,
    // ) -> Result<(), WaCustomError> {
    //     index_data_map.remove
    //     Ok(())
    // }

    fn search_internal(
        &self,
        collection: &Collection,
        query: Self::SearchInput,
        options: &Self::SearchOptions,
        config: &Config,
        return_raw_text: bool,
    ) -> Result<Vec<InternalSearchResult>, WaCustomError>;

    fn remap_search_results(
        &self,
        collection: &Collection,
        results: Vec<InternalSearchResult>,
        return_raw_text: bool,
    ) -> Result<Vec<SearchResult>, WaCustomError> {
        results
            .into_iter()
            .map(|(internal_id, id, document_id, score, text)| {
                Ok(if let Some(id) = id {
                    (id, document_id, score, text)
                } else {
                    let raw_emb = collection
                        .get_raw_emb_by_internal_id(&internal_id)
                        .ok_or_else(|| {
                            WaCustomError::NotFound("raw embedding not found".to_string())
                        })?
                        .clone();
                    (
                        raw_emb.id.clone(),
                        raw_emb.document_id.clone(),
                        score,
                        if return_raw_text {
                            raw_emb.text.clone()
                        } else {
                            None
                        },
                    )
                })
            })
            .collect()
    }

    fn search(
        &self,
        collection: &Collection,
        query: Self::SearchInput,
        options: &Self::SearchOptions,
        config: &Config,
        return_raw_text: bool,
    ) -> Result<Vec<SearchResult>, WaCustomError> {
        let results = self.search_internal(collection, query, options, config, return_raw_text)?;
        self.remap_search_results(collection, results, return_raw_text)
    }

    fn batch_search(
        &self,
        collection: &Collection,
        queries: Vec<Self::SearchInput>,
        options: &Self::SearchOptions,
        config: &Config,
        return_raw_text: bool,
    ) -> Result<Vec<Vec<SearchResult>>, WaCustomError> {
        queries
            .into_par_iter()
            .map(|query| self.search(collection, query, options, config, return_raw_text))
            .collect()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum IndexData {
    Hnsw(hnsw::HNSWIndexData),
    Inverted(inverted::InvertedIndexData),
    TfIdf(tf_idf::TFIDFIndexData),
    Usv(usv::USVIndexData),
}

impl IndexData {
    fn index_type(&self) -> IndexType {
        match self {
            Self::Hnsw(_) => IndexType::Hnsw,
            Self::Inverted(_) => IndexType::Inverted,
            Self::TfIdf(_) => IndexType::TfIdf,
            Self::Usv(_) => IndexType::Usv,
        }
    }
}

impl CborSerialize for IndexData {}
impl CborDeserialize for IndexData {}

/// Index is identified by collection name (string) and type of index
#[derive(PartialEq, Eq, Hash)]
struct IndexId(String, IndexType);

impl TreeMapKey for IndexId {
    fn key(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Persistent storage for index data, implemented using TreeMap
pub struct IndexDataMap {
    inner: TreeMap<IndexId, IndexData>,
}

impl IndexDataMap {
    /// Loads the IndexDataMap from disk (if it exists) or
    /// creates a new one (if it's the first run).
    ///
    /// # Panics
    /// The method panics if:
    ///
    ///   1. fails to open/read from files or
    ///   2. fails to deserialize existing data
    ///   3. fails to serialize empty TreeMap on first run
    pub fn load_or_create() -> Self {
        let data_path = get_data_path();
        let file_path = data_path.join("indexes.dim");
        let is_file_exists = std::fs::exists(&file_path).expect("Failed to check if file exists");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(!is_file_exists)
            .open(file_path)
            .unwrap();

        let dim_bufman = BufferManager::new(file, 8192).unwrap();
        let data_bufmans = BufferManagerFactory::new(
            data_path.into(),
            |root, version: &VersionNumber| root.join(format!("indexes.{}.data", **version)),
            8192,
        );

        // @NOTE: The TreeMap can be deserialized from disk only if
        // it's initialized and serialized at least once, else it
        // results in stack overflow. Hence, deserialize if the file
        // exists, otherwise create a new TreeMap and immediately
        // serialize it.
        let inner = if is_file_exists {
            TreeMap::deserialize(dim_bufman, data_bufmans)
                .expect("Failed to deserialize TreeMap for CollectionMetadataMap")
        } else {
            let map = TreeMap::new(dim_bufman, data_bufmans);
            // Immediately serialize it so that the file is created
            map.serialize().expect("Failed to serialize TreeMap");
            map
        };
        Self { inner }
    }

    pub fn get(&self, collection_name: &str, index_type: IndexType) -> Option<&IndexData> {
        let index_id = IndexId(collection_name.to_owned(), index_type);
        self.inner.get_latest(&index_id)
    }

    pub fn insert(&self, collection_name: &str, data: IndexData) -> Result<(), BufIoError> {
        let index_type = data.index_type();
        let index_id = IndexId(collection_name.to_owned(), index_type);
        self.inner.insert(0.into(), &index_id, data);
        self.inner.serialize()?;
        Ok(())
    }

    pub fn remove(&self, collection_name: &str, index_type: IndexType) -> Result<(), BufIoError> {
        let index_id = IndexId(collection_name.to_owned(), index_type);
        self.inner.delete(0.into(), &index_id);
        self.inner.serialize()?;
        Ok(())
    }
}
