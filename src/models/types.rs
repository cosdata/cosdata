use super::{
    buffered_io::BufferManagerFactory,
    cache_loader::HNSWIndexCache,
    collection::Collection,
    crypto::{DoubleSHA256Hash, SingleSHA256Hash},
    inverted_index::InvertedIndexRoot,
    meta_persist::{
        lmdb_init_collections_db, lmdb_init_db, load_collections, retrieve_current_version,
        retrieve_values_upper_bound,
    },
    paths::get_data_path,
    prob_lazy_load::lazy_item::FileIndex,
    prob_node::ProbNode,
    versioning::VersionControl,
};
use crate::{
    args::CosdataArgs,
    config_loader::Config,
    distance::{
        cosine::{CosineDistance, CosineSimilarity},
        dotproduct::DotProductDistance,
        euclidean::EuclideanDistance,
        hamming::HammingDistance,
        DistanceError, DistanceFunction,
    },
    indexes::{
        hnsw::{data::HNSWIndexData, HNSWIndex},
        inverted::{data::InvertedIndexData, InvertedIndex},
    },
    models::{
        buffered_io::BufIoError, common::*, meta_persist::retrieve_values_range, versioning::*,
    },
    quantization::{
        product::ProductQuantization, scalar::ScalarQuantization, Quantization, QuantizationError,
        StorageType,
    },
    storage::Storage,
};
use dashmap::DashMap;
use lmdb::{Cursor, Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher24;
use std::{
    fmt,
    fs::{create_dir_all, OpenOptions},
    hash::{Hash as StdHash, Hasher},
    path::Path,
    ptr,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicUsize},
        Arc, RwLock,
    },
    time::Instant,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HNSWLevel(pub u8);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FileOffset(pub u32);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BytesToRead(pub u32);

pub type PropPersistRef = (FileOffset, BytesToRead);

#[derive(Debug, Clone, PartialEq)]
pub struct NodeProp {
    pub id: VectorId,
    pub value: Arc<Storage>,
    pub location: PropPersistRef,
}

impl StdHash for NodeProp {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id.hash(state);
    }
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct VectorId(pub u64);

impl VectorId {
    pub fn get_hash(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum MetricResult {
    CosineSimilarity(CosineSimilarity),
    // @DOUBT: how can we obtain `CosineDistance`?
    CosineDistance(CosineDistance),
    EuclideanDistance(EuclideanDistance),
    HammingDistance(HammingDistance),
    // @DOUBT: dot product shows similarity between two vectors, not distance,
    // should rename it to `DotProduct`?
    DotProductDistance(DotProductDistance),
}

impl PartialOrd for MetricResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for MetricResult {}

impl Ord for MetricResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self {
            Self::CosineSimilarity(val) => val.0.total_cmp(&other.get_value()),
            Self::CosineDistance(val) => other.get_value().total_cmp(&val.0),
            Self::EuclideanDistance(val) => other.get_value().total_cmp(&val.0),
            Self::HammingDistance(val) => other.get_value().total_cmp(&val.0),
            Self::DotProductDistance(val) => val.0.total_cmp(&other.get_value()),
        }
    }
}

impl MetricResult {
    // gets the bare numerical value stored in the type
    pub fn get_value(&self) -> f32 {
        match self {
            MetricResult::CosineSimilarity(value) => value.0,
            MetricResult::CosineDistance(value) => value.0,
            MetricResult::EuclideanDistance(value) => value.0,
            MetricResult::HammingDistance(value) => value.0,
            MetricResult::DotProductDistance(value) => value.0,
        }
    }

    pub fn get_tag_and_value(&self) -> (u8, f32) {
        match self {
            Self::CosineSimilarity(value) => (0, value.0),
            Self::CosineDistance(value) => (1, value.0),
            Self::EuclideanDistance(value) => (2, value.0),
            Self::HammingDistance(value) => (3, value.0),
            Self::DotProductDistance(value) => (4, value.0),
        }
    }

    pub fn min(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Cosine => Self::CosineSimilarity(CosineSimilarity(-1.0)),
            DistanceMetric::Euclidean => {
                Self::EuclideanDistance(EuclideanDistance(f32::NEG_INFINITY))
            }
            DistanceMetric::Hamming => Self::HammingDistance(HammingDistance(f32::NEG_INFINITY)),
            DistanceMetric::DotProduct => {
                Self::DotProductDistance(DotProductDistance(f32::NEG_INFINITY))
            }
        }
    }

    pub fn max(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Cosine => Self::CosineSimilarity(CosineSimilarity(2.0)), // take care of precision issues
            DistanceMetric::Euclidean => Self::EuclideanDistance(EuclideanDistance(f32::INFINITY)),
            DistanceMetric::Hamming => Self::HammingDistance(HammingDistance(f32::INFINITY)),
            DistanceMetric::DotProduct => {
                Self::DotProductDistance(DotProductDistance(f32::INFINITY))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Hamming,
    DotProduct,
}

impl DistanceFunction for DistanceMetric {
    type Item = MetricResult;
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<Self::Item, DistanceError> {
        match self {
            Self::Cosine => {
                let value = CosineSimilarity(0.0).calculate(x, y)?;
                Ok(MetricResult::CosineSimilarity(value))
            }
            Self::Euclidean => {
                let value = EuclideanDistance(0.0).calculate(x, y)?;
                Ok(MetricResult::EuclideanDistance(value))
            }
            Self::Hamming => {
                let value = HammingDistance(0.0).calculate(x, y)?;
                Ok(MetricResult::HammingDistance(value))
            }
            Self::DotProduct => {
                let value = DotProductDistance(0.0).calculate(x, y)?;
                Ok(MetricResult::DotProductDistance(value))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMetric {
    Scalar,
    Product(ProductQuantization),
}

impl Quantization for QuantizationMetric {
    fn quantize(
        &self,
        vector: &[f32],
        storage_type: StorageType,
        range: (f32, f32),
    ) -> Result<Storage, QuantizationError> {
        match self {
            Self::Scalar => ScalarQuantization.quantize(vector, storage_type, range),
            Self::Product(product) => product.quantize(vector, storage_type, range),
        }
    }

    fn train(&mut self, vectors: &[&[f32]]) -> Result<(), QuantizationError> {
        match self {
            Self::Scalar => ScalarQuantization.train(vectors),
            Self::Product(product) => product.train(vectors),
        }
    }
}

// Implementing the std::fmt::Display trait for VectorId
impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct MetaDb {
    pub env: Arc<Environment>,
    pub db: Arc<Database>,
}

impl MetaDb {
    pub fn from_env(env: Arc<Environment>, collection_name: &str) -> lmdb::Result<Self> {
        let db = Arc::new(env.create_db(Some(collection_name), DatabaseFlags::empty())?);

        Ok(Self { env, db })
    }
}

pub struct CollectionsMap {
    /// holds an in-memory map of all dense indexes for all collections
    inner_hnsw_indexes: DashMap<String, Arc<HNSWIndex>>,
    inner_inverted_indexes: DashMap<String, Arc<InvertedIndex>>,
    inner_collections: DashMap<String, Arc<Collection>>,
    lmdb_env: Arc<Environment>,
    // made it public temporarily
    // just to be able to persist collections from outside CollectionsMap
    pub(crate) lmdb_collections_db: Database,
    lmdb_hnsw_index_db: Database,
    #[allow(dead_code)]
    lmdb_inverted_index_db: Database,
}

impl CollectionsMap {
    fn new(env: Arc<Environment>) -> lmdb::Result<Self> {
        let collections_db = lmdb_init_collections_db(&env)?;
        let hnsw_index_db = lmdb_init_db(&env, "hnsw_indexes")?;
        let inverted_index_db = lmdb_init_db(&env, "inverted_indexes")?;
        let res = Self {
            inner_hnsw_indexes: DashMap::new(),
            inner_inverted_indexes: DashMap::new(),
            inner_collections: DashMap::new(),
            lmdb_env: env,
            lmdb_collections_db: collections_db,
            lmdb_hnsw_index_db: hnsw_index_db,
            lmdb_inverted_index_db: inverted_index_db,
        };
        Ok(res)
    }

    /// Loads collections map from lmdb
    ///
    /// In doing so, the root vec for all collections' dense indexes are loaded into
    /// memory, which also ends up warming the cache (NodeRegistry)
    fn load(env: Arc<Environment>, config: &Config) -> Result<Self, WaCustomError> {
        let collections_map =
            Self::new(env.clone()).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let collections = load_collections(
            &collections_map.lmdb_env,
            collections_map.lmdb_collections_db,
        )
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let root_path = get_data_path().join("collections");

        for coll in collections {
            let coll = Arc::new(coll);
            collections_map
                .inner_collections
                .insert(coll.name.clone(), coll.clone());

            // if collection has dense index load it from the lmdb
            if coll.dense_vector.enabled {
                let hnsw_index = collections_map.load_hnsw_index(&coll, &root_path, config)?;
                collections_map
                    .inner_hnsw_indexes
                    .insert(coll.name.clone(), Arc::new(hnsw_index));
            }

            // if collection has inverted index load it from the lmdb
            if coll.sparse_vector.enabled {
                let inverted_index =
                    collections_map.load_inverted_index(&coll, &root_path, config)?;
                collections_map
                    .inner_inverted_indexes
                    .insert(coll.name.clone(), Arc::new(inverted_index));
            }
        }
        Ok(collections_map)
    }

    /// loads and initiates the dense index of a collection from lmdb
    ///
    /// In doing so, the root vec for all collections' dense indexes are loaded into
    /// memory, which also ends up warming the cache (NodeRegistry)
    fn load_hnsw_index(
        &self,
        coll: &Collection,
        root_path: &Path,
        config: &Config,
    ) -> Result<HNSWIndex, WaCustomError> {
        let collection_path: Arc<Path> = root_path.join(&coll.name).into();
        let index_path = collection_path.join("dense_hnsw");

        let hnsw_index_data =
            HNSWIndexData::load(&self.lmdb_env, self.lmdb_hnsw_index_db, &coll.get_key())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
        let prop_file = RwLock::new(
            OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(index_path.join("prop.data"))
                .unwrap(),
        );

        let node_size = ProbNode::get_serialized_size(hnsw_index_data.hnsw_params.neighbors_count);
        let level_0_node_size =
            ProbNode::get_serialized_size(hnsw_index_data.hnsw_params.level_0_neighbors_count);

        let bufman_size = node_size * 1000;
        let level_0_bufman_size = level_0_node_size * 1000;

        let index_manager = Arc::new(BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            bufman_size,
        ));
        let level_0_index_manager = Arc::new(BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}_0.index", **ver)),
            level_0_bufman_size,
        ));
        let vec_raw_manager = BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
            8192,
        );
        let distance_metric = Arc::new(RwLock::new(hnsw_index_data.distance_metric));
        let cache = HNSWIndexCache::new(
            index_manager.clone(),
            level_0_index_manager.clone(),
            prop_file,
            distance_metric.clone(),
        );

        let db = Arc::new(
            self.lmdb_env
                .create_db(Some(&coll.name), DatabaseFlags::empty())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
        );

        let FileIndex {
            offset: root_offset,
            version_number: root_version_number,
            version_id: root_version_id,
        } = hnsw_index_data.file_index;

        let root_node_region_offset = root_offset.0 - (root_offset.0 % bufman_size as u32);
        let load_start = Instant::now();
        let region = cache.load_region(
            root_node_region_offset,
            root_version_number,
            root_version_id,
            node_size as u32,
            false,
        )?;

        let root = region[(root_offset.0 - root_node_region_offset) as usize / node_size];

        let vcs = VersionControl::from_existing(self.lmdb_env.clone(), db.clone());

        let mut versions = vcs
            .get_branch_versions("main")
            .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

        println!(
            "versions: {:?}",
            versions.iter().map(|(hash, _)| **hash).collect::<Vec<_>>()
        );

        // root node region is already loaded
        let mut num_regions_queued = 1;
        let mut regions_to_load = Vec::new();

        while !versions.is_empty() {
            let num_regions_to_load = (config.num_regions_to_load_on_restart - num_regions_queued)
                .div_ceil(versions.len())
                / 2;
            let (version_id, version_hash) = versions.remove(0);
            // level n
            let bufman = index_manager.get(version_id)?;
            for i in 0..num_regions_to_load.min((bufman.file_size() as usize).div_ceil(bufman_size))
            {
                let region_start = (bufman_size * i) as u32;
                if version_id == root_version_id && region_start == root_node_region_offset {
                    continue;
                }
                regions_to_load.push((
                    region_start,
                    *version_hash.version as u16,
                    version_id,
                    node_size as u32,
                    false,
                ));
                num_regions_queued += 1;
            }
            // level 0
            let bufman = level_0_index_manager.get(version_id)?;
            for i in 0..num_regions_to_load
                .min((bufman.file_size() as usize).div_ceil(level_0_bufman_size))
            {
                let region_start = (level_0_bufman_size * i) as u32;
                regions_to_load.push((
                    region_start,
                    *version_hash.version as u16,
                    version_id,
                    level_0_node_size as u32,
                    true,
                ));
                num_regions_queued += 1;
            }
        }

        regions_to_load
            .into_par_iter()
            .map(
                |(region_start, version_number, version_id, node_size, is_level_0)| {
                    cache.load_region(
                        region_start,
                        version_number,
                        version_id,
                        node_size,
                        is_level_0,
                    )?;
                    Ok(())
                },
            )
            .collect::<Result<Vec<_>, BufIoError>>()?;

        let load_time = load_start.elapsed();
        println!("Loaded regions in: {:?}", load_time);

        let lmdb = MetaDb {
            env: self.lmdb_env.clone(),
            db,
        };
        let current_version = retrieve_current_version(&lmdb)?;
        let values_range = retrieve_values_range(&lmdb)?;
        let hnsw_index = HNSWIndex::new(
            coll.name.clone(),
            root,
            hnsw_index_data.levels_prob,
            hnsw_index_data.dim,
            lmdb,
            current_version,
            hnsw_index_data.quantization_metric,
            distance_metric,
            hnsw_index_data.storage_type,
            vcs,
            hnsw_index_data.hnsw_params,
            cache,
            vec_raw_manager,
            values_range.unwrap_or((-1.0, 1.0)),
            hnsw_index_data.sample_threshold,
            values_range.is_some(),
        );

        Ok(hnsw_index)
    }

    /// loads and initiates the inverted index of a collection from lmdb
    ///
    /// In doing so, the root vec for all collections'  inverted indexes are loaded into
    /// memory, which also ends up warming the cache (NodeRegistry)
    fn load_inverted_index(
        &self,
        coll: &Collection,
        root_path: &Path,
        config: &Config,
    ) -> Result<InvertedIndex, WaCustomError> {
        let collection_path: Arc<Path> = root_path.join(&coll.name).into();
        let index_path = collection_path.join("sparse_inverted_index");

        let vec_raw_manager = BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
            8192,
        );

        let db = Arc::new(
            self.lmdb_env
                .create_db(Some(&coll.name), DatabaseFlags::empty())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
        );

        let inverted_index_data =
            InvertedIndexData::load(&self.lmdb_env, self.lmdb_inverted_index_db, &coll.get_key())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let vcs = VersionControl::from_existing(self.lmdb_env.clone(), db.clone());
        let lmdb = MetaDb {
            env: self.lmdb_env.clone(),
            db,
        };
        let current_version = retrieve_current_version(&lmdb)?;
        let values_upper_bound = retrieve_values_upper_bound(&lmdb)?;
        let inverted_index = InvertedIndex {
            name: coll.name.clone(),
            description: inverted_index_data.description,
            auto_create_index: inverted_index_data.auto_create_index,
            metadata_schema: inverted_index_data.metadata_schema,
            max_vectors: inverted_index_data.max_vectors,
            root: InvertedIndexRoot::deserialize(
                index_path,
                inverted_index_data.quantization_bits,
                config.inverted_index_data_file_parts,
            )?,
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: AtomicPtr::new(ptr::null_mut()),
            vcs,
            values_upper_bound: RwLock::new(values_upper_bound.unwrap_or(1.0)),
            is_configured: AtomicBool::new(values_upper_bound.is_some()),
            vectors: RwLock::new(Vec::new()),
            vectors_collected: AtomicUsize::new(0),
            sampling_data: crate::indexes::inverted::types::SamplingData::default(),
            sample_threshold: inverted_index_data.sample_threshold,
            vec_raw_manager,
            early_terminate_threshold: inverted_index_data.early_terminate_threshold,
        };

        Ok(inverted_index)
    }

    pub fn insert_hnsw_index(
        &self,
        name: &str,
        hnsw_index: Arc<HNSWIndex>,
    ) -> Result<(), WaCustomError> {
        HNSWIndexData::persist(&self.lmdb_env, self.lmdb_hnsw_index_db, &hnsw_index)?;
        self.inner_hnsw_indexes
            .insert(name.to_owned(), hnsw_index.clone());
        Ok(())
    }

    pub fn insert_inverted_index(
        &self,
        name: &str,
        index: Arc<InvertedIndex>,
    ) -> Result<(), WaCustomError> {
        InvertedIndexData::persist(&self.lmdb_env, self.lmdb_inverted_index_db, &index)?;
        self.inner_inverted_indexes
            .insert(name.to_owned(), index.clone());
        Ok(())
    }

    /// inserts a collection into the collections map
    #[allow(dead_code)]
    pub fn insert_collection(&self, collection: Arc<Collection>) -> Result<(), WaCustomError> {
        self.inner_collections
            .insert(collection.name.to_owned(), collection);
        Ok(())
    }

    /// Returns the `HNSWIndex` by collection's name
    ///
    /// If not found, None is returned
    ///
    /// Note that it tried to look up the DenseIndex in the DashMap
    /// only and doesn't check LMDB. This is because of the assumption
    /// that at startup, all DenseIndexes will be loaded from LMDB
    /// into the in-memory DashMap and when a new DenseIndex is
    /// added, it will be written to the DashMap as well.
    ///
    /// @TODO: As a future improvement, we can fallback to checking if
    /// the DenseIndex exists in LMDB and caching it. But it's not
    /// required for the current use case.
    #[allow(dead_code)]
    pub fn get_hnsw_index(&self, name: &str) -> Option<Arc<HNSWIndex>> {
        self.inner_hnsw_indexes.get(name).map(|index| index.clone())
    }

    /// Returns the `InvertedIndex` by collection's name
    ///
    /// If not found, None is returned
    ///
    /// Note that it tried to look up the InvertedIndex in the DashMap
    /// only and doesn't check LMDB. This is because of the assumption
    /// that at startup, all InvertedIndexes will be loaded from LMDB
    /// into the in-memory DashMap and when a new InvertedIndex is
    /// added, it will be written to the DashMap as well.
    ///
    /// @TODO: As a future improvement, we can fallback to checking if
    /// the InvertedIndex exists in LMDB and caching it. But it's not
    /// required for the current use case.
    pub fn get_inverted_index(&self, name: &str) -> Option<Arc<InvertedIndex>> {
        self.inner_inverted_indexes
            .get(name)
            .map(|index| index.clone())
    }

    /// Returns the `Collection` by collection's name
    ///
    /// If not found, None is returned
    ///
    /// Note that it tried to look up the Collections in the DashMap
    /// only and doesn't check LMDB. This is because of the assumption
    /// that at startup, all collections will be loaded from LMDB
    /// into the in-memory DashMap and when a new collection is
    /// added, it will be written to the DashMap as well.
    ///
    /// @TODO: As a future improvement, we can fallback to checking if
    /// the Collection exists in LMDB and caching it. But it's not
    /// required for the current use case.
    #[allow(dead_code)]
    pub fn get_collection(&self, name: &str) -> Option<Arc<Collection>> {
        self.inner_collections.get(name).map(|index| index.clone())
    }

    #[allow(dead_code)]
    pub fn remove_hnsw_index(
        &self,
        name: &str,
    ) -> Result<Option<(String, Arc<HNSWIndex>)>, WaCustomError> {
        match self.inner_hnsw_indexes.remove(name) {
            Some((key, hnsw_index)) => {
                HNSWIndexData::delete_index(&self.lmdb_env, self.lmdb_hnsw_index_db, &hnsw_index)
                    .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
                Ok(Some((key, hnsw_index)))
            }
            None => Ok(None),
        }
    }

    #[allow(dead_code)]
    pub fn remove_inverted_index(
        &self,
        name: &str,
    ) -> Result<Option<(String, Arc<InvertedIndex>)>, WaCustomError> {
        match self.inner_inverted_indexes.remove(name) {
            Some((key, inverted_index)) => {
                InvertedIndexData::delete_index(
                    &self.lmdb_env,
                    self.lmdb_hnsw_index_db,
                    &inverted_index,
                )
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
                Ok(Some((key, inverted_index)))
            }
            None => Ok(None),
        }
    }

    /// removes a collection from the in-memory map
    ///
    /// returns the removed collection in case of success
    ///
    /// returns error if not found
    #[allow(dead_code)]
    pub fn remove_collection(&self, name: &str) -> Result<Arc<Collection>, WaCustomError> {
        match self.inner_collections.remove(name) {
            Some((_, collection)) => Ok(collection),
            None => {
                // collection not found, return an error response
                Err(WaCustomError::NotFound("collection".into()))
            }
        }
    }

    #[allow(dead_code)]
    pub fn iter_hnsw_indexes(
        &self,
    ) -> dashmap::iter::Iter<
        String,
        Arc<HNSWIndex>,
        std::hash::RandomState,
        DashMap<String, Arc<HNSWIndex>>,
    > {
        self.inner_hnsw_indexes.iter()
    }

    #[allow(dead_code)]
    pub fn iter_inverted_indexes(
        &self,
    ) -> dashmap::iter::Iter<
        String,
        Arc<InvertedIndex>,
        std::hash::RandomState,
        DashMap<String, Arc<InvertedIndex>>,
    > {
        self.inner_inverted_indexes.iter()
    }

    /// returns an iterator
    #[allow(dead_code)]
    pub fn iter_collections(
        &self,
    ) -> dashmap::iter::Iter<
        String,
        Arc<Collection>,
        std::hash::RandomState,
        DashMap<String, Arc<Collection>>,
    > {
        self.inner_collections.iter()
    }
}

pub struct UsersMap {
    env: Arc<Environment>,
    users_db: Database,
    // (username, user details)
    map: DashMap<String, User>,
}

impl UsersMap {
    pub fn new(env: Arc<Environment>) -> lmdb::Result<Self> {
        let users_db = env.create_db(Some("users"), DatabaseFlags::empty())?;
        let txn = env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(users_db)?;
        let map = DashMap::new();

        for (username, user_bytes) in cursor.iter() {
            let username = String::from_utf8(username.to_vec()).unwrap();
            let user = User::deserialize(user_bytes).unwrap();
            map.insert(username, user);
        }

        drop(cursor);
        txn.abort();

        Ok(Self { env, users_db, map })
    }

    pub fn add_user(&self, username: String, password_hash: DoubleSHA256Hash) -> lmdb::Result<()> {
        let user = User {
            username: username.clone(),
            password_hash,
        };
        let user_bytes = user.serialize();
        let username_bytes = username.as_bytes();

        let mut txn = self.env.begin_rw_txn()?;
        txn.put(
            self.users_db,
            &username_bytes,
            &user_bytes,
            WriteFlags::empty(),
        )?;
        txn.commit()?;

        self.map.insert(username, user);

        Ok(())
    }

    pub fn get_user(&self, username: &str) -> Option<User> {
        self.map.get(username).map(|user| user.value().clone())
    }
}

#[derive(Clone)]
pub struct User {
    pub username: String,
    pub password_hash: DoubleSHA256Hash,
}

impl User {
    fn serialize(&self) -> Vec<u8> {
        let username_bytes = self.username.as_bytes();
        let mut buf = Vec::with_capacity(32 + username_bytes.len());
        buf.extend_from_slice(&self.password_hash.0);
        buf.extend_from_slice(username_bytes);
        buf
    }

    fn deserialize(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < 32 {
            return Err("Input must be at least 32 bytes".to_string());
        }
        let mut password_hash = [0u8; 32];
        password_hash.copy_from_slice(&buf[..32]);
        let username_bytes = buf[32..].to_vec();
        let username = String::from_utf8(username_bytes).map_err(|err| err.to_string())?;
        Ok(Self {
            username,
            password_hash: DoubleSHA256Hash(password_hash),
        })
    }
}

pub struct SessionDetails {
    pub created_at: u64,
    pub expires_at: u64,
    pub user: User,
}

// Define the AppEnv struct
pub struct AppEnv {
    pub collections_map: CollectionsMap,
    pub users_map: UsersMap,
    pub persist: Arc<Environment>,
    // Single hash, must not be persisted to disk, only the double hash must be
    // written to disk
    pub admin_key: SingleSHA256Hash,
    pub active_sessions: Arc<DashMap<String, SessionDetails>>,
}

fn get_admin_key(env: Arc<Environment>, args: CosdataArgs) -> lmdb::Result<SingleSHA256Hash> {
    let db = env.create_db(Some("security_metadata"), DatabaseFlags::empty())?;
    let mut txn = env.begin_rw_txn()?;
    let admin_key_from_lmdb = match txn.get(db, &"admin_key") {
        Ok(key) => Some(DoubleSHA256Hash(key.try_into().unwrap())),
        Err(lmdb::Error::NotFound) => None,
        Err(err) => return Err(err),
    };
    let admin_key_hash = if let Some(admin_key_from_lmdb) = admin_key_from_lmdb {
        txn.abort();
        let arg_admin_key = args.admin_key;
        let arg_admin_key_hash = SingleSHA256Hash::from_str(&arg_admin_key).unwrap();
        let arg_admin_key_double_hash = arg_admin_key_hash.hash_again();
        if !admin_key_from_lmdb.verify_eq(&arg_admin_key_double_hash) {
            log::error!("Invalid admin key!");
            std::process::exit(1);
        }
        arg_admin_key_hash
    } else {
        let arg_admin_key = args.admin_key;
        let arg_admin_key_hash = SingleSHA256Hash::from_str(&arg_admin_key).unwrap();
        let arg_admin_key_double_hash = arg_admin_key_hash.hash_again();
        txn.put(
            db,
            &"admin_key",
            &arg_admin_key_double_hash.0,
            WriteFlags::empty(),
        )?;
        txn.commit()?;
        arg_admin_key_hash
    };
    Ok(admin_key_hash)
}

pub fn get_app_env(config: &Config, args: CosdataArgs) -> Result<Arc<AppEnv>, WaCustomError> {
    let db_path = get_data_path().join("_mdb"); // TODO: prefix the customer & database name

    // Ensure the directory exists
    create_dir_all(&db_path).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    // Initialize the environment
    let env = Environment::new()
        .set_max_dbs(10)
        .set_map_size(1048576000) // Set the maximum size of the database to 1GB
        .open(&db_path)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let env_arc = Arc::new(env);

    let admin_key = get_admin_key(env_arc.clone(), args)
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

    let collections_map = CollectionsMap::load(env_arc.clone(), config)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let users_map = UsersMap::new(env_arc.clone())
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

    // Fake user, for testing APIs
    let username = "admin".to_string();
    let password = "admin";
    let password_hash = DoubleSHA256Hash::from_str(password).unwrap();
    users_map
        .add_user(username, password_hash)
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

    Ok(Arc::new(AppEnv {
        collections_map,
        users_map,
        persist: env_arc,
        admin_key,
        active_sessions: Arc::new(DashMap::new()),
    }))
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct SparseVector {
    pub vector_id: u32,
    pub entries: Vec<(u32, f32)>,
}

impl SparseVector {
    #[allow(unused)]
    pub fn new(vector_id: u32, entries: Vec<(u32, f32)>) -> Self {
        Self { vector_id, entries }
    }
}

// Dimension type for a sparse vector, based on the posting list length.
// A dimension being common or rare is relative, and is defined based on the
// query vector.
#[derive(Debug, Clone, Copy)]
pub enum SparseQueryVectorDimensionType {
    Common,
    Rare,
}

// A sparse query vector, which attaches a dimension type to each dimension
// based on the posting list length. This is used to optimize sparse ANN search
// using a cuckoo filter.
#[derive(Debug, Clone)]
pub struct SparseQueryVector {
    pub entries: Vec<(u32, SparseQueryVectorDimensionType, f32)>,
}

impl SparseQueryVector {
    pub fn new(entries: Vec<(u32, SparseQueryVectorDimensionType, f32)>) -> Self {
        Self { entries }
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::cosine::CosineSimilarity;

    use super::MetricResult;

    #[test]
    fn test_metric_result_ordering() {
        let mut metric_results = vec![
            MetricResult::CosineSimilarity(CosineSimilarity(6.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(5.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(4.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(3.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(2.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(1.0)),
        ];

        let correctly_ordered_metric_results = vec![
            MetricResult::CosineSimilarity(CosineSimilarity(1.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(2.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(3.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(4.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(5.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(6.0)),
        ];

        metric_results.sort();

        assert_eq!(metric_results, correctly_ordered_metric_results);
    }
}
