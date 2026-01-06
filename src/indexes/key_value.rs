use std::{fs::OpenOptions, path::PathBuf};

use crate::{
    config_loader::Config,
    models::{
        buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
        collection::Collection,
        common::WaCustomError,
        tree_map::TreeMap,
        types::InternalId,
        versioning::VersionNumber,
    },
};

use super::{IndexData, IndexOps};

pub struct KeyValueInputPair {
    pub id: InternalId,
    pub value: Vec<u8>,
}

pub struct KeyValueIndex {
    pub(crate) tree_map: TreeMap<u32, Vec<u8>>,
}

impl KeyValueIndex {
    pub fn new(root_path: PathBuf) -> Result<Self, BufIoError> {
        let dim_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root_path.join("tree-map.dim"))?;
        let dim_bufman = BufferManager::new(dim_file, 8192)?;
        let data_bufmans = BufferManagerFactory::new(
            root_path.into(),
            |root, ver: &VersionNumber| root.join(format!("{}.data", **ver)),
            8192,
        );
        let tree_map = TreeMap::new(dim_bufman, data_bufmans);

        Ok(Self { tree_map })
    }

    pub fn insert(&self, version: VersionNumber, key: InternalId, value: Vec<u8>) {
        self.tree_map.insert(version, &*key, value);
    }

    pub fn lookup(&self, key: &InternalId) -> Option<Vec<u8>> {
        self.tree_map.get_latest(key).cloned()
    }
}

impl IndexOps for KeyValueIndex {
    type IndexingInput = KeyValueInputPair;
    type SearchInput = ();
    type SearchOptions = ();

    fn validate_embedding(&self, _embedding: Self::IndexingInput) -> Result<(), WaCustomError> {
        Ok(())
    }

    fn index_embeddings(
        &self,
        _collection: &Collection,
        embeddings: Vec<Self::IndexingInput>,
        version: VersionNumber,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
        for emb in embeddings {
            self.insert(version, emb.id, emb.value);
        }

        Ok(())
    }

    fn delete_embedding(
        &self,
        _id: InternalId,
        _raw_emb: &crate::models::collection::RawVectorEmbedding,
        _version: VersionNumber,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
        // TODO(a-rustacean): impl delete?
        Ok(())
    }

    fn finalize_sampling(
        &self,
        _lmdb: &crate::models::types::MetaDb,
        _config: &Config,
        _embeddings: &[Self::IndexingInput],
    ) -> Result<(), WaCustomError> {
        Ok(())
    }

    fn sample_embedding(&self, _embedding: &Self::IndexingInput) {}

    fn embeddings_collected(&self) -> &std::sync::RwLock<Vec<Self::IndexingInput>> {
        unreachable!()
    }

    fn increment_collected_count(&self, _count: usize) -> usize {
        0
    }

    fn sample_threshold(&self) -> usize {
        0
    }

    fn is_configured(&self) -> bool {
        true
    }

    fn flush(
        &self,
        _collection: &Collection,
        _version: VersionNumber,
    ) -> Result<(), WaCustomError> {
        self.tree_map.serialize()?;
        Ok(())
    }

    fn get_data(&self) -> Option<IndexData> {
        None
    }

    fn search_internal(
        &self,
        _collection: &Collection,
        _query: Self::SearchInput,
        _options: &Self::SearchOptions,
        _config: &Config,
        _return_raw_text: bool,
    ) -> Result<Vec<super::InternalSearchResult>, WaCustomError> {
        Err(WaCustomError::NotImplemented(
            "Regular index search cannot be used with key value index".to_string(),
        ))
    }
}
