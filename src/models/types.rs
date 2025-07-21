use super::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::HNSWIndexCache,
    collection::{Collection, CollectionMetadata},
    collection_transaction::ImplicitTransaction,
    crypto::{DoubleSHA256Hash, SingleSHA256Hash},
    indexing_manager::IndexingManager,
    inverted_index::InvertedIndexRoot,
    meta_persist::{
        lmdb_init_collections_db, lmdb_init_db, load_collections, retrieve_average_document_length,
        retrieve_background_version, retrieve_current_version, retrieve_highest_internal_id,
        retrieve_values_upper_bound,
    },
    paths::get_data_path,
    prob_node::ProbNode,
    tf_idf_index::TFIDFIndexRoot,
    tree_map::{TreeMap, TreeMapKey, TreeMapVec},
    versioning::{VersionControl, VersionNumber},
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
        hnsw::{
            offset_counter::{HNSWIndexFileOffsetCounter, IndexFileId},
            HNSWIndex,
        },
        inverted::InvertedIndex,
        tf_idf::TFIDFIndex,
        IndexOps,
    },
    metadata::{schema::MetadataDimensions, QueryFilterDimensions, HIGH_WEIGHT},
    models::{
        buffered_io::{BufferManager, FilelessBufferManager},
        common::*,
        lazy_item::{FileIndex, LazyItem},
        meta_persist::retrieve_values_range,
        prob_node::{LatestNode, SharedLatestNode},
        serializer::hnsw::RawDeserialize,
    },
    quantization::{
        product::ProductQuantization, scalar::ScalarQuantization, Quantization, QuantizationError,
        StorageType,
    },
    storage::Storage,
};
use crossbeam::channel;
use dashmap::DashMap;
use lmdb::{Cursor, Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use rayon::ThreadPool;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher24;
use std::{
    fmt,
    fs::{self, create_dir_all, OpenOptions},
    hash::{Hash as StdHash, Hasher},
    io::Write,
    ops::{Deref, Div, Mul},
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicUsize},
        Arc, RwLock,
    },
    thread,
    time::Instant,
};
use tempfile::tempfile;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HNSWLevel(pub u8);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FileOffset(pub u32);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BytesToRead(pub u32);

pub type PropPersistRef = (FileOffset, BytesToRead);

#[derive(Debug, PartialEq)]
pub struct NodePropValue {
    pub id: InternalId,
    pub vec: Arc<Storage>,
    pub location: PropPersistRef,
}

impl StdHash for NodePropValue {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id.hash(state);
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub mag: f32,
    pub mbits: Vec<i32>,
}

impl From<MetadataDimensions> for Metadata {
    fn from(dims: MetadataDimensions) -> Self {
        let total = dims
            .iter()
            .map(|d| {
                let x = *d as f32;
                x * x
            })
            .sum::<f32>();
        // @NOTE: As `MetadataDimensions` have high weight values, we
        // need to handle overflow during intermediate addition when
        // calculating the euclidean norm
        let mag = total.min(f32::MAX).sqrt();
        Self { mag, mbits: dims }
    }
}

impl From<&QueryFilterDimensions> for Metadata {
    fn from(dims: &QueryFilterDimensions) -> Self {
        let dims_i32 = dims.iter().map(|d| *d as i32).collect::<Vec<i32>>();
        // @NOTE: Unlike `MetadataDimensions`, `QueryFilterDimensions`
        // will have -1, 0, 1 values so no need to worry about
        // overflow during summation
        let mag = dims
            .iter()
            .map(|d| {
                let x = *d as f32;
                x * x
            })
            .sum::<f32>()
            .sqrt();
        Self {
            mag,
            mbits: dims_i32,
        }
    }
}

impl PartialEq for Metadata {
    fn eq(&self, other: &Self) -> bool {
        self.mag.to_bits() == other.mag.to_bits() && self.mbits == other.mbits
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataId(pub u8);

#[derive(Debug, PartialEq)]
pub struct NodePropMetadata {
    pub replica_id: InternalId,
    pub vec: Arc<Metadata>,
    pub location: PropPersistRef,
}

/// Kinds of nodes in the HNSW/dense index
///
/// They are called 'replica nodes' because when metadata fields are
/// supported by the index, one embedding may result in multiple nodes
/// getting added to the HNSW graph.
#[derive(Debug)]
pub enum ReplicaNodeKind {
    /// These are "static" nodes created at the time of index
    /// initialization under the pseudo root node (See `RootNodeKind`
    /// for more types of root nodes)
    Pseudo,
    /// Nodes corresponding to the vector embeddings with metadata
    /// dimensions either absent or default value (all 0s)
    Base,
    /// Nodes corresponding to the vector embeddings with metadata
    /// dimensions set
    Metadata,
}

impl ReplicaNodeKind {
    /// Returns the kind of root node that this kind of replica must
    /// be indexed under
    pub fn root_node_kind(&self) -> RootNodeKind {
        match self {
            Self::Pseudo => RootNodeKind::Pseudo,
            Self::Base => RootNodeKind::Main,
            Self::Metadata => RootNodeKind::Pseudo,
        }
    }
}

/// Kinds of root nodes in HNSW/dense index
#[derive(Debug)]
pub enum RootNodeKind {
    Pseudo,
    Main,
}

#[derive(Debug)]
pub struct VectorData<'a> {
    // Vector id (use specified one and not the internal replica
    // id). It's not being used any where but occasionally useful for
    // debugging. In case it's a query vector, `id` expected to be
    // None.
    pub id: Option<&'a InternalId>,
    pub quantized_vec: &'a Storage,
    pub metadata: Option<&'a Metadata>,
}

impl<'a> VectorData<'a> {
    pub fn without_metadata(id: Option<&'a InternalId>, qvec: &'a Storage) -> Self {
        Self {
            id,
            quantized_vec: qvec,
            metadata: None,
        }
    }

    pub fn replica_node_kind(&self) -> ReplicaNodeKind {
        match self.metadata {
            Some(m) => {
                if m.mag == 0.0 {
                    ReplicaNodeKind::Base
                } else {
                    match self.id {
                        Some(id) => {
                            if ((u32::MAX - 257)..=(u32::MAX - 2)).contains(&**id) {
                                ReplicaNodeKind::Pseudo
                            } else {
                                ReplicaNodeKind::Metadata
                            }
                        }
                        None => ReplicaNodeKind::Metadata,
                    }
                }
            }
            None => ReplicaNodeKind::Base,
        }
    }

    pub fn is_pseudo_root(&self) -> bool {
        match self.metadata {
            Some(m) => m.mbits == vec![HIGH_WEIGHT; m.mbits.len()],
            None => false,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Hash, Eq, PartialEq)]
pub struct VectorId(String);

impl From<String> for VectorId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl Deref for VectorId {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<VectorId> for String {
    fn from(id: VectorId) -> Self {
        id.0
    }
}

impl TreeMapKey for VectorId {
    fn key(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq)]
pub struct DocumentId(String);

impl From<String> for DocumentId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl Deref for DocumentId {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<DocumentId> for String {
    fn from(id: DocumentId) -> Self {
        id.0
    }
}

impl TreeMapKey for DocumentId {
    fn key(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
    Serialize,
    Deserialize,
)]
pub struct InternalId(u32);

impl InternalId {
    /// Increments an InternalId by 1 and returns a new instance
    ///
    /// The caller needs to ensure it doesn't result in overflow
    pub fn inc(&self) -> Self {
        InternalId::from(self.0 + 1)
    }
}

impl From<u32> for InternalId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl Deref for InternalId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<InternalId> for u32 {
    fn from(id: InternalId) -> Self {
        id.0
    }
}

impl TreeMapKey for InternalId {
    fn key(&self) -> u64 {
        self.0 as u64
    }
}

impl Div<u32> for InternalId {
    type Output = Self;

    fn div(self, rhs: u32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl Mul<u32> for InternalId {
    type Output = Self;

    fn mul(self, rhs: u32) -> Self::Output {
        Self(self.0 * rhs)
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
    fn calculate(
        &self,
        x: &VectorData,
        y: &VectorData,
        is_indexing: bool,
    ) -> Result<Self::Item, DistanceError> {
        match self {
            Self::Cosine => {
                let value = CosineSimilarity(0.0).calculate(x, y, is_indexing)?;
                Ok(MetricResult::CosineSimilarity(value))
            }
            Self::Euclidean => {
                let value = EuclideanDistance(0.0).calculate(x, y, is_indexing)?;
                Ok(MetricResult::EuclideanDistance(value))
            }
            Self::Hamming => {
                let value = HammingDistance(0.0).calculate(x, y, is_indexing)?;
                Ok(MetricResult::HammingDistance(value))
            }
            Self::DotProduct => {
                let value = DotProductDistance(0.0).calculate(x, y, is_indexing)?;
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
    pub db: Database,
}

impl MetaDb {
    pub fn from_env(env: Arc<Environment>, collection_name: &str) -> lmdb::Result<Self> {
        let db = env.create_db(Some(collection_name), DatabaseFlags::empty())?;

        Ok(Self { env, db })
    }
}

pub struct CollectionsMap {
    inner_collections: DashMap<String, Arc<Collection>>,
    lmdb_env: Arc<Environment>,
    // made it public temporarily
    // just to be able to persist collections from outside CollectionsMap
    pub(crate) lmdb_collections_db: Database,
    lmdb_hnsw_index_db: Database,
    lmdb_inverted_index_db: Database,
    lmdb_tf_idf_index_db: Database,
}

impl CollectionsMap {
    fn new(env: Arc<Environment>) -> lmdb::Result<Self> {
        let collections_db = lmdb_init_collections_db(&env)?;
        let hnsw_index_db = lmdb_init_db(&env, "hnsw_indexes")?;
        let inverted_index_db = lmdb_init_db(&env, "inverted_indexes")?;
        let tf_idf_index_db = lmdb_init_db(&env, "tf_idf_indexes")?;
        let res = Self {
            inner_collections: DashMap::new(),
            lmdb_env: env,
            lmdb_collections_db: collections_db,
            lmdb_hnsw_index_db: hnsw_index_db,
            lmdb_inverted_index_db: inverted_index_db,
            lmdb_tf_idf_index_db: tf_idf_index_db,
        };
        Ok(res)
    }

    /// Loads collections map from lmdb
    fn load(
        env: Arc<Environment>,
        config: Arc<Config>,
        threadpool: Arc<ThreadPool>,
    ) -> Result<Self, WaCustomError> {
        let collections_map =
            Self::new(env.clone()).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let collections = load_collections(
            &collections_map.lmdb_env,
            collections_map.lmdb_collections_db,
        )
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        for collection_meta in collections {
            let lmdb = MetaDb::from_env(collections_map.lmdb_env.clone(), &collection_meta.name)?;
            let current_version = retrieve_current_version(&lmdb)?;
            let vcs = VersionControl::from_existing(lmdb.env.clone(), lmdb.db);

            // if collection has dense index load it from the lmdb
            let hnsw_index = if collection_meta.dense_vector.enabled {
                collections_map
                    .load_hnsw_index(
                        &collection_meta,
                        &lmdb,
                        &config,
                        collection_meta
                            .metadata_schema
                            .as_ref()
                            .map_or(1, |schema| schema.max_num_replicas()),
                        current_version,
                    )
                    .unwrap()
                    .map(Arc::new)
            } else {
                None
            };

            // if collection has inverted index load it from the lmdb
            let inverted_index = if collection_meta.sparse_vector.enabled {
                collections_map
                    .load_inverted_index(&collection_meta, &lmdb)?
                    .map(Arc::new)
            } else {
                None
            };

            let tf_idf_index = if collection_meta.tf_idf_options.enabled {
                collections_map
                    .load_tf_idf_index(&collection_meta, &lmdb)?
                    .map(Arc::new)
            } else {
                None
            };

            let collection_path: Arc<Path> =
                get_collections_path().join(&collection_meta.name).into();

            let internal_to_external_map_dim_file = OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(false)
                .create(true)
                .open(collection_path.join("itoe.dim"))
                .map_err(BufIoError::Io)?;

            let internal_to_external_map_dim_bufman =
                BufferManager::new(internal_to_external_map_dim_file, 8192)
                    .map_err(BufIoError::Io)?;

            let internal_to_external_map_data_bufmans = BufferManagerFactory::new(
                collection_path.clone(),
                |root, version: &VersionNumber| root.join(format!("itoe.{}.data", **version)),
                8192,
            );

            let external_to_internal_map_dim_file = OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(false)
                .create(true)
                .open(collection_path.join("etoi.dim"))
                .map_err(BufIoError::Io)?;

            let external_to_internal_map_dim_bufman =
                BufferManager::new(external_to_internal_map_dim_file, 8192)
                    .map_err(BufIoError::Io)?;

            let external_to_internal_map_data_bufmans = BufferManagerFactory::new(
                collection_path.clone(),
                |root, version: &VersionNumber| root.join(format!("etoi.{}.data", **version)),
                8192,
            );

            let document_to_internals_map_dim_file = OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(false)
                .create(true)
                .open(collection_path.join("dtoi.dim"))
                .map_err(BufIoError::Io)?;

            let document_to_internals_map_dim_bufman =
                BufferManager::new(document_to_internals_map_dim_file, 8192)
                    .map_err(BufIoError::Io)?;

            let document_to_internals_map_data_bufmans = BufferManagerFactory::new(
                collection_path.clone(),
                |root, version: &VersionNumber| root.join(format!("dtoi.{}.data", **version)),
                8192,
            );

            let transaction_status_map_dim_file = OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(false)
                .create(true)
                .open(collection_path.join("txn_status.dim"))
                .map_err(BufIoError::Io)?;

            let transaction_status_map_dim_bufman =
                BufferManager::new(transaction_status_map_dim_file, 8192)
                    .map_err(BufIoError::Io)?;

            let transaction_status_map_data_bufmans = BufferManagerFactory::new(
                collection_path.clone(),
                |root, version: &VersionNumber| root.join(format!("txn_status.{}.data", **version)),
                8192,
            );

            let id_counter_value = retrieve_highest_internal_id(&lmdb)?.unwrap_or_default();

            let collection = Arc::new(Collection {
                meta: collection_meta,
                lmdb,
                current_version: parking_lot::RwLock::new(current_version),
                last_allotted_version: parking_lot::RwLock::new(current_version),
                current_explicit_transaction: parking_lot::RwLock::new(None),
                current_implicit_transaction: parking_lot::RwLock::new(
                    ImplicitTransaction::default(),
                ),
                vcs,
                internal_to_external_map: TreeMap::deserialize(
                    internal_to_external_map_dim_bufman,
                    internal_to_external_map_data_bufmans,
                )?,
                external_to_internal_map: TreeMap::deserialize(
                    external_to_internal_map_dim_bufman,
                    external_to_internal_map_data_bufmans,
                )?,
                document_to_internals_map: TreeMapVec::deserialize(
                    document_to_internals_map_dim_bufman,
                    document_to_internals_map_data_bufmans,
                )?,
                transaction_status_map: TreeMap::deserialize(
                    transaction_status_map_dim_bufman,
                    transaction_status_map_data_bufmans,
                )?,
                internal_id_counter: AtomicU32::new(id_counter_value),
                hnsw_index: parking_lot::RwLock::new(hnsw_index),
                inverted_index: parking_lot::RwLock::new(inverted_index),
                tf_idf_index: parking_lot::RwLock::new(tf_idf_index),
                indexing_manager: parking_lot::RwLock::new(None),
                is_indexing: AtomicBool::new(false),
            });

            *collection.indexing_manager.write() = Some(IndexingManager::new(
                collection.clone(),
                config.clone(),
                threadpool.clone(),
            ));

            let background_version = retrieve_background_version(&collection.lmdb)?;

            if background_version != current_version {
                for version in *background_version..*current_version {
                    let version = VersionNumber::from(version + 1);
                    IndexingManager::index_version_on_restart(
                        &collection,
                        &config,
                        &threadpool,
                        version,
                    )
                    .unwrap();
                }
            }

            collections_map
                .inner_collections
                .insert(collection.meta.name.clone(), collection);
        }
        Ok(collections_map)
    }

    /// loads and initiates the dense index of a collection from lmdb
    ///
    /// In doing so, the root vec for all collections' dense indexes are loaded into
    /// memory, which also ends up warming the cache (NodeRegistry)
    fn load_hnsw_index(
        &self,
        collection_meta: &CollectionMetadata,
        lmdb: &MetaDb,
        config: &Config,
        max_replicas_per_node: u8,
        current_version: VersionNumber,
    ) -> Result<Option<HNSWIndex>, WaCustomError> {
        let collection_path: Arc<Path> = get_collections_path().join(&collection_meta.name).into();
        let index_path = collection_path.join("dense_hnsw");

        // Check if the path exists before proceeding
        if !index_path.exists() {
            return Ok(None);
        }

        let Some(hnsw_index_data) = HNSWIndex::load_data(
            &self.lmdb_env,
            self.lmdb_hnsw_index_db,
            &collection_meta.name,
        )?
        else {
            return Ok(None);
        };
        let prop_file_path = index_path.join("prop.data");
        let prop_file_result = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&prop_file_path);

        let prop_file = match prop_file_result {
            Ok(file) => RwLock::new(file),
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to open properties file {:?}: {}",
                    prop_file_path, e
                )));
            }
        };

        let index_manager = BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &IndexFileId| root.join(format!("{}.index", **ver)),
            8192,
        );

        let latest_version_links_bufman = if config.enable_context_history {
            FilelessBufferManager::from_versioned(
                8192,
                &index_path,
                |path| {
                    let file_name = path.file_name()?.to_str()?;
                    let parts: Vec<&str> = file_name.split(".").collect();
                    if parts.len() != 2 {
                        return None;
                    }
                    let region_version_combined = parts[0];
                    let parts: Vec<&str> = region_version_combined.split("-").collect();

                    if parts.len() != 2 {
                        return None;
                    }

                    let region_id = parts[0].parse().ok()?;
                    let version = VersionNumber::from(parts[1].parse::<u32>().ok()?);

                    Some((version, region_id))
                },
                current_version,
            )?
        } else {
            let mut file = OpenOptions::new()
                .read(true)
                .open(index_path.join("nodes.ptr"))
                .map_err(BufIoError::Io)?;
            FilelessBufferManager::from_file(&mut file, 8192)?
        };
        let distance_metric = Arc::new(RwLock::new(hnsw_index_data.distance_metric));
        let cache = HNSWIndexCache::new(
            index_manager,
            latest_version_links_bufman,
            index_path.clone(),
            config.enable_context_history,
            prop_file,
            distance_metric.clone(),
        );

        let values_range_result = retrieve_values_range(lmdb);
        let values_range = match values_range_result {
            Ok(vr) => vr,
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to retrieve values range: {}",
                    e
                )));
            }
        };

        let first_index_file_path = index_path.join("0.index");
        let (file_id, offset) = if first_index_file_path.exists() {
            let mut file_id = 0;
            let mut offset = fs::metadata(first_index_file_path)
                .map_err(BufIoError::Io)?
                .len() as u32;
            loop {
                let index_file_path = index_path.join(format!("{}.index", file_id));
                if index_file_path.exists() {
                    file_id += 1;
                    offset = fs::metadata(index_file_path).map_err(BufIoError::Io)?.len() as u32;
                } else {
                    break;
                }
            }

            (file_id, offset)
        } else {
            (0, 0)
        };
        let offset_counter = HNSWIndexFileOffsetCounter::from_offset_and_file_id(
            offset,
            file_id,
            cache.latest_version_links_bufman.file_size() as u32,
            config.index_file_min_size,
            hnsw_index_data.hnsw_params.level_0_neighbors_count,
            hnsw_index_data.hnsw_params.neighbors_count,
        );

        let start = Instant::now();
        let mut pending_items = FxHashMap::default();
        let mut latest_version_links = FxHashMap::default();

        let root_ptr_offset = hnsw_index_data.root_vec_ptr_offset;
        let latest_version_links_cursor = cache.latest_version_links_bufman.open_cursor()?;
        let temp_file = tempfile().map_err(BufIoError::Io)?;
        let dummy_bufman = BufferManager::new(temp_file, 8192).map_err(BufIoError::Io)?;
        let root_file_index = SharedLatestNode::deserialize_raw(
            &dummy_bufman, // not used
            &cache.latest_version_links_bufman,
            u64::MAX, // not used
            latest_version_links_cursor,
            root_ptr_offset,
            IndexFileId::invalid(),
            &cache,
        )?;
        let bufman = cache.bufmans.get(root_file_index.file_id)?;
        let cursor = bufman.open_cursor()?;
        let root_node_raw = ProbNode::deserialize_raw(
            &bufman,
            &cache.latest_version_links_bufman,
            cursor,
            latest_version_links_cursor,
            root_file_index.offset,
            root_file_index.file_id,
            &cache,
        )?;
        let root_node = ProbNode::build_from_raw(
            root_node_raw,
            &cache,
            &dummy_bufman,
            &mut pending_items,
            &mut latest_version_links,
            latest_version_links_cursor,
        )?;
        let root = LazyItem::new(root_node, root_file_index.file_id, root_file_index.offset);
        cache
            .registry
            .insert(HNSWIndexCache::combine_index(&root_file_index), root);
        let root_ptr = LatestNode::new(root, root_ptr_offset);
        latest_version_links.insert(root_ptr_offset, root_ptr);
        bufman.close_cursor(cursor)?;

        let pseudo_root_ptr = match hnsw_index_data.pseudo_root_vec_ptr_offset {
            Some(pseudo_root_ptr_offset) => {
                let latest_version_links_cursor =
                    cache.latest_version_links_bufman.open_cursor()?;
                let pseudo_root_file_index = SharedLatestNode::deserialize_raw(
                    &dummy_bufman, // not used
                    &cache.latest_version_links_bufman,
                    u64::MAX, // not used
                    latest_version_links_cursor,
                    pseudo_root_ptr_offset,
                    IndexFileId::invalid(),
                    &cache,
                )?;
                let bufman = cache.bufmans.get(pseudo_root_file_index.file_id)?;
                let cursor = bufman.open_cursor()?;
                let pseudo_root_node_raw = ProbNode::deserialize_raw(
                    &bufman,
                    &cache.latest_version_links_bufman,
                    cursor,
                    latest_version_links_cursor,
                    pseudo_root_file_index.offset,
                    pseudo_root_file_index.file_id,
                    &cache,
                )?;
                let pseudo_root_node = ProbNode::build_from_raw(
                    pseudo_root_node_raw,
                    &cache,
                    &dummy_bufman,
                    &mut pending_items,
                    &mut latest_version_links,
                    latest_version_links_cursor,
                )?;
                let pseudo_root = LazyItem::new(
                    pseudo_root_node,
                    pseudo_root_file_index.file_id,
                    pseudo_root_file_index.offset,
                );
                cache.registry.insert(
                    HNSWIndexCache::combine_index(&pseudo_root_file_index),
                    pseudo_root,
                );
                let pseudo_root_ptr = LatestNode::new(pseudo_root, pseudo_root_ptr_offset);
                latest_version_links.insert(pseudo_root_ptr_offset, pseudo_root_ptr);
                bufman.close_cursor(cursor)?;
                Some(pseudo_root_ptr)
            }
            None => None,
        };

        let (file_index_sender, file_index_receiver) =
            channel::unbounded::<FileIndex<IndexFileId>>();
        let (raw_node_sender, raw_node_receiver) =
            channel::unbounded::<(<ProbNode as RawDeserialize>::Raw, FileIndex<IndexFileId>)>();

        thread::scope(|s| {
            let mut handles = Vec::new();
            let file_index_receiver = Arc::new(file_index_receiver);
            let cache = &cache;
            for _ in 0..8 {
                let file_index_receiver = file_index_receiver.clone();
                let cursors = (0..=offset_counter.file_id)
                    .map(|file_id| cache.bufmans.get(IndexFileId::from(file_id))?.open_cursor())
                    .collect::<Result<Vec<_>, _>>()?;
                let latest_version_links_cursor =
                    cache.latest_version_links_bufman.open_cursor()?;
                let raw_node_sender = raw_node_sender.clone();
                let handle = s.spawn(move || {
                    for file_index in &*file_index_receiver {
                        let bufman = cache.bufmans.get(file_index.file_id).unwrap();
                        let node = ProbNode::deserialize_raw(
                            &bufman,
                            &cache.latest_version_links_bufman,
                            cursors[*file_index.file_id as usize],
                            latest_version_links_cursor,
                            file_index.offset,
                            file_index.file_id,
                            cache,
                        )
                        .unwrap_or_else(|_| {
                            panic!("failed to load node at file index: {:?}", file_index)
                        });
                        raw_node_sender.send((node, file_index)).unwrap();
                    }

                    for (file_id, cursor) in cursors.into_iter().enumerate() {
                        let file_id = IndexFileId::from(file_id as u32);
                        cache.bufmans.get(file_id)?.close_cursor(cursor)?;
                    }

                    cache
                        .latest_version_links_bufman
                        .close_cursor(latest_version_links_cursor)?;

                    Ok::<_, WaCustomError>(())
                });
                handles.push(handle);
            }

            while !pending_items.is_empty() {
                let keys = pending_items.keys().cloned().collect::<Vec<_>>();
                let len = keys.len();
                for file_index in keys {
                    file_index_sender.send(file_index).unwrap();
                }

                for _ in 0..len {
                    let (raw, file_index) = raw_node_receiver.recv().unwrap();

                    let node = ProbNode::build_from_raw(
                        raw,
                        cache,
                        &dummy_bufman,
                        &mut pending_items,
                        &mut latest_version_links,
                        latest_version_links_cursor,
                    )?;
                    let lazy_item = pending_items.remove(&file_index).unwrap();
                    let lazy_item_ref = unsafe { &*lazy_item };
                    lazy_item_ref.set_data(node);
                    cache
                        .registry
                        .insert(HNSWIndexCache::combine_index(&file_index), lazy_item);
                }
            }

            drop(file_index_sender);

            for handle in handles {
                handle.join().unwrap()?;
            }

            Ok::<_, WaCustomError>(())
        })?;

        println!("Dense index loaded in {:?}", start.elapsed());
        cache
            .latest_version_links_bufman
            .close_cursor(latest_version_links_cursor)?;

        let hnsw_index = HNSWIndex::new(
            root_ptr,
            pseudo_root_ptr,
            hnsw_index_data.levels_prob,
            hnsw_index_data.dim,
            hnsw_index_data.quantization_metric,
            distance_metric,
            hnsw_index_data.storage_type,
            hnsw_index_data.hnsw_params,
            cache,
            values_range.unwrap_or((-1.0, 1.0)),
            hnsw_index_data.sample_threshold,
            values_range.is_some(),
            max_replicas_per_node,
            offset_counter,
        );

        Ok(Some(hnsw_index))
    }

    /// loads and initiates the inverted index of a collection from lmdb
    fn load_inverted_index(
        &self,
        collection_meta: &CollectionMetadata,
        lmdb: &MetaDb,
    ) -> Result<Option<InvertedIndex>, WaCustomError> {
        let collection_path: Arc<Path> = get_collections_path().join(&collection_meta.name).into();
        let index_path = collection_path.join("sparse_inverted_index");

        if !index_path.exists() {
            return Ok(None);
        }

        let Some(inverted_index_data) = InvertedIndex::load_data(
            &self.lmdb_env,
            self.lmdb_inverted_index_db,
            &collection_meta.name,
        )?
        else {
            return Ok(None);
        };

        let values_upper_bound = retrieve_values_upper_bound(lmdb)?;
        let inverted_index = InvertedIndex {
            root: InvertedIndexRoot::deserialize(
                index_path,
                inverted_index_data.quantization_bits,
            )?,
            values_upper_bound: RwLock::new(values_upper_bound.unwrap_or(1.0)),
            is_configured: AtomicBool::new(values_upper_bound.is_some()),
            vectors: RwLock::new(Vec::new()),
            vectors_collected: AtomicUsize::new(0),
            sampling_data: crate::indexes::inverted::types::SamplingData::default(),
            sample_threshold: inverted_index_data.sample_threshold,
        };

        Ok(Some(inverted_index))
    }

    /// loads and initiates the TF-IDF index of a collection from lmdb
    fn load_tf_idf_index(
        &self,
        collection_meta: &CollectionMetadata,
        lmdb: &MetaDb,
    ) -> Result<Option<TFIDFIndex>, WaCustomError> {
        let collection_path: Arc<Path> = get_collections_path().join(&collection_meta.name).into();
        let index_path = collection_path.join("tf_idf_index");

        if !index_path.exists() {
            return Ok(None);
        }

        let Some(inverted_index_data) = TFIDFIndex::load_data(
            &self.lmdb_env,
            self.lmdb_tf_idf_index_db,
            &collection_meta.name,
        )?
        else {
            return Ok(None);
        };

        let average_document_length = retrieve_average_document_length(lmdb)?;
        let inverted_index = TFIDFIndex {
            root: TFIDFIndexRoot::deserialize(index_path)?,
            average_document_length: RwLock::new(average_document_length.unwrap_or(1.0)),
            is_configured: AtomicBool::new(average_document_length.is_some()),
            documents: RwLock::new(Vec::new()),
            documents_collected: AtomicUsize::new(0),
            sampling_data: crate::indexes::tf_idf::SamplingData::default(),
            sample_threshold: inverted_index_data.sample_threshold,
            k1: inverted_index_data.k1,
            b: inverted_index_data.b,
        };

        Ok(Some(inverted_index))
    }

    pub fn insert_hnsw_index(
        &self,
        collection: &Collection,
        hnsw_index: Arc<HNSWIndex>,
    ) -> Result<(), WaCustomError> {
        hnsw_index.persist(
            &collection.meta.name,
            &self.lmdb_env,
            self.lmdb_hnsw_index_db,
        )?;
        *collection.hnsw_index.write() = Some(hnsw_index);
        Ok(())
    }

    pub fn insert_inverted_index(
        &self,
        collection: &Collection,
        inverted_index: Arc<InvertedIndex>,
    ) -> Result<(), WaCustomError> {
        inverted_index.persist(
            &collection.meta.name,
            &self.lmdb_env,
            self.lmdb_inverted_index_db,
        )?;
        *collection.inverted_index.write() = Some(inverted_index);
        Ok(())
    }

    pub fn insert_tf_idf_index(
        &self,
        collection: &Collection,
        tf_idf_index: Arc<TFIDFIndex>,
    ) -> Result<(), WaCustomError> {
        tf_idf_index.persist(
            &collection.meta.name,
            &self.lmdb_env,
            self.lmdb_tf_idf_index_db,
        )?;
        *collection.tf_idf_index.write() = Some(tf_idf_index);
        Ok(())
    }

    /// inserts a collection into the collections map
    #[allow(dead_code)]
    pub fn insert_collection(&self, collection: Arc<Collection>) -> Result<(), WaCustomError> {
        self.inner_collections
            .insert(collection.meta.name.to_owned(), collection);
        Ok(())
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
    pub fn get_collection(&self, name: &str) -> Option<Arc<Collection>> {
        self.inner_collections.get(name).map(|index| index.clone())
    }

    pub fn remove_hnsw_index(&self, name: &str) -> Result<Option<Arc<HNSWIndex>>, WaCustomError> {
        match self.inner_collections.get(name) {
            Some(collection) => match collection.hnsw_index.write().take() {
                Some(hnsw_index) => {
                    HNSWIndex::delete(&self.lmdb_env, self.lmdb_hnsw_index_db, name)?;
                    Ok(Some(hnsw_index))
                }
                None => Ok(None),
            },
            None => Ok(None),
        }
    }

    pub fn remove_inverted_index(
        &self,
        name: &str,
    ) -> Result<Option<Arc<InvertedIndex>>, WaCustomError> {
        match self.inner_collections.get(name) {
            Some(collection) => match collection.inverted_index.write().take() {
                Some(inverted_index) => {
                    InvertedIndex::delete(&self.lmdb_env, self.lmdb_inverted_index_db, name)?;
                    Ok(Some(inverted_index))
                }
                None => Ok(None),
            },
            None => Ok(None),
        }
    }

    pub fn remove_tf_idf_index(
        &self,
        name: &str,
    ) -> Result<Option<Arc<TFIDFIndex>>, WaCustomError> {
        match self.inner_collections.get(name) {
            Some(collection) => match collection.tf_idf_index.write().take() {
                Some(tf_idf_index) => {
                    TFIDFIndex::delete(&self.lmdb_env, self.lmdb_tf_idf_index_db, name)?;
                    Ok(Some(tf_idf_index))
                }
                None => Ok(None),
            },
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
    // Create meta database if it doesn't exist
    let init_txn = env.begin_rw_txn()?;
    unsafe { init_txn.create_db(Some("meta"), DatabaseFlags::empty())? };
    init_txn.commit()?;

    let txn = env.begin_ro_txn()?;
    let db = unsafe { txn.open_db(Some("meta"))? };

    let admin_key_from_lmdb = match txn.get(db, &"admin_key") {
        Ok(bytes) => {
            let mut hash_array = [0u8; 32];
            // Copy bytes from the database to the fixed-size array
            if bytes.len() >= 32 {
                hash_array.copy_from_slice(&bytes[..32]);
                Some(DoubleSHA256Hash(hash_array))
            } else {
                log::error!("Invalid admin key format in database");
                return Err(lmdb::Error::Other(7));
            }
        }
        Err(lmdb::Error::NotFound) => None,
        Err(e) => return Err(e),
    };
    txn.abort();

    let admin_key_hash = if let Some(admin_key_from_lmdb) = admin_key_from_lmdb {
        // Database already exists, verify admin key
        let arg_admin_key = args.admin_key;
        let arg_admin_key_hash = SingleSHA256Hash::from_str(&arg_admin_key).unwrap();
        let arg_admin_key_double_hash = arg_admin_key_hash.hash_again();
        if !admin_key_from_lmdb.verify_eq(&arg_admin_key_double_hash) {
            log::error!("Invalid admin key!");
            return Err(lmdb::Error::Other(5));
        }
        arg_admin_key_hash
    } else {
        // First-time setup
        let arg_admin_key = args.admin_key;
        let arg_admin_key_hash = SingleSHA256Hash::from_str(&arg_admin_key).unwrap();
        let arg_admin_key_double_hash = arg_admin_key_hash.hash_again();

        // Store the admin key double hash in the database
        let mut txn = env.begin_rw_txn()?;
        let db = unsafe { txn.open_db(Some("meta"))? };
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

pub fn get_collections_path() -> PathBuf {
    get_data_path().join("collections")
}

pub fn get_app_env(
    config: Arc<Config>,
    threadpool: Arc<ThreadPool>,
    args: CosdataArgs,
) -> Result<Arc<AppEnv>, WaCustomError> {
    // Check both possible db path locations
    let db_path_1 = get_data_path().join("_mdb");
    let db_path_2 = get_data_path().join("data/_mdb");

    // Use whichever path exists, or default to db_path_2
    let db_path = if db_path_1.exists() {
        //println!("Using existing database at {}", db_path_1.display());
        db_path_1
    } else if db_path_2.exists() {
        //println!("Using existing database at {}", db_path_2.display());
        db_path_2
    } else {
        //println!("Creating new database at {}", db_path_2.display());
        db_path_2 // Default for first-time setup
    };

    // Check if this is first-time setup
    let is_first_time = !db_path.exists();

    // If this is first time and confirmation is required
    if is_first_time && !args.skip_confirmation && !args.confirmed {
        // Interactive prompt for confirmation
        print!("Re-enter admin key: ");
        std::io::stdout().flush().unwrap();

        let mut confirmation = String::new();
        std::io::stdin().read_line(&mut confirmation).map_err(|e| {
            WaCustomError::ConfigError(format!("Failed to read confirmation: {}", e))
        })?;

        // Remove trailing newline
        confirmation = confirmation.trim().to_string();

        if confirmation != args.admin_key {
            return Err(WaCustomError::ConfigError(
                "Admin key and confirmation do not match".to_string(),
            ));
        }

        println!("Admin key confirmed successfully.");
    }

    // Create a modified args with confirmed flag set
    let mut confirmed_args = args.clone();
    confirmed_args.confirmed = true;

    // Ensure parent directories exist first if needed
    if let Some(parent) = db_path.parent() {
        create_dir_all(parent).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    }

    // Ensure the database directory exists
    create_dir_all(&db_path).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    // Initialize the environment
    let env = Environment::new()
        .set_max_dbs(10)
        .set_map_size(1048576000) // Set the maximum size of the database to 1GB
        .open(&db_path)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let env_arc = Arc::new(env);

    let admin_key = get_admin_key(env_arc.clone(), confirmed_args)
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

    // Add more resilient error handling for collections_map loading
    let collections_map = CollectionsMap::load(env_arc.clone(), config, threadpool)?;

    let users_map = match UsersMap::new(env_arc.clone()) {
        Ok(map) => map,
        Err(err) => {
            println!("Warning: Failed to load users map: {}", err);
            return Err(WaCustomError::DatabaseError(err.to_string()));
        }
    };

    // Use the admin key as the password instead of hardcoded "admin"
    let username = "admin".to_string();
    let password = args.admin_key.clone();
    let password_hash = DoubleSHA256Hash::from_str(&password).unwrap();

    // Don't fail if user already exists
    match users_map.add_user(username, password_hash) {
        Ok(_) => {}
        Err(err) => {
            println!(
                "Note: Could not add admin user (may already exist): {}",
                err
            );
        }
    };

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
