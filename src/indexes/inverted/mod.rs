pub(crate) mod types;

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use serde::{Deserialize, Serialize};
use snowball_stemmer::Stemmer;
use twox_hash::XxHash32;

use super::{tf_idf::tokenize, IndexOps, InternalSearchResult, Matches, SearchResult};
use crate::{
    config_loader::Config,
    models::{
        buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
        collection::{Collection, RawVectorEmbedding},
        common::WaCustomError,
        inverted_index::InvertedIndexRoot,
        serializer::tree_map::TreeMapSerialize,
        sparse_ann_query::SparseAnnQueryBasic,
        tree_map::{TreeMapKey, TreeMapVec},
        types::{DocumentId, InternalId, MetaDb, SparseVector},
        versioned_vec::{VersionedVec, VersionedVecItem},
        versioning::VersionNumber,
    },
};
use std::{
    f32::consts::PI,
    fs::OpenOptions,
    hash::{DefaultHasher, Hash, Hasher},
    ops::Deref,
    path::PathBuf,
    sync::RwLock,
};

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct ZoneId(String);

impl From<String> for ZoneId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl Deref for ZoneId {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TreeMapKey for ZoneId {
    fn key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write(self.0.as_bytes());
        hasher.finish()
    }
}

pub struct GeoFencedDocument {
    pub id: InternalId,
    pub weight: f32, // specified by the API
    pub coordinates: (f32, f32),
}

impl VersionedVecItem for GeoFencedDocument {
    type Storage = u128;
    type Id = u32;

    fn id(storage: &Self::Storage) -> Self::Id {
        (*storage >> 96) as u32
    }

    fn into_storage(self) -> Self::Storage {
        ((*self.id as u128) << 96)
            | ((self.weight.to_bits() as u128) << 64)
            | ((self.coordinates.0.to_bits() as u128) << 32)
            | (self.coordinates.1.to_bits() as u128)
    }

    fn from_storage(storage: Self::Storage) -> Self {
        let id = InternalId::from((storage >> 96) as u32);
        let weight = f32::from_bits((storage >> 64) as u32);
        let coordinates = (
            f32::from_bits((storage >> 32) as u32),
            f32::from_bits(storage as u32),
        );

        Self {
            id,
            weight,
            coordinates,
        }
    }
}

pub struct SparseSearchOptions {
    pub top_k: Option<usize>,
    pub early_terminate_threshold: Option<f32>,
    pub sort_by_distance: bool,
    pub coordinates: (f32, f32),
    pub zones: Vec<ZoneId>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct InvertedIndexData {
    pub quantization_bits: u8,
}

pub struct GeoFenceInputEmbedding(
    pub InternalId,
    pub RawVectorEmbedding,
    pub FxHashMap<String, String>,
);

pub struct SparseSearchInput(pub String);

const MAX_TOKEN_LEN: usize = 40;
const MAX_CARDINALITY: usize = 20;

pub struct InvertedIndex {
    pub root: InvertedIndexRoot,
    pub fields_values: TreeMapVec<String, u64>,
    pub zones: TreeMapVec<ZoneId, GeoFencedDocument>,
    pub fields: RwLock<VersionedVec<String>>,
    pub fields_bufmans: BufferManagerFactory<VersionNumber>,
}

unsafe impl Send for InvertedIndex {}
unsafe impl Sync for InvertedIndex {}

impl InvertedIndex {
    pub fn new(
        root_path: PathBuf,
        version: VersionNumber,
        quantization_bits: u8,
    ) -> Result<Self, BufIoError> {
        let root = InvertedIndexRoot::new(root_path.clone(), quantization_bits)?;

        let zones_dim_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(root_path.join("zones.dim"))
            .map_err(BufIoError::Io)?;
        let zones_dim_bufman = BufferManager::new(zones_dim_file, 8192).map_err(BufIoError::Io)?;

        let zones_data_bufmans = BufferManagerFactory::new(
            root_path.clone().into(),
            |root, version: &VersionNumber| root.join(format!("zones.{}.data", **version)),
            8192,
        );

        let fields_values_dim_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(root_path.join("fields_values.dim"))
            .map_err(BufIoError::Io)?;
        let fields_values_dim_bufman =
            BufferManager::new(fields_values_dim_file, 8192).map_err(BufIoError::Io)?;

        let fields_values_data_bufmans = BufferManagerFactory::new(
            root_path.clone().into(),
            |root, version: &VersionNumber| root.join(format!("fields_values.{}.data", **version)),
            8192,
        );

        let fields_bufmans = BufferManagerFactory::new(
            root_path.clone().into(),
            |root, version: &VersionNumber| root.join(format!("{}.fields", **version)),
            8192,
        );

        Ok(Self {
            root,
            zones: TreeMapVec::new(zones_dim_bufman, zones_data_bufmans),
            fields_values: TreeMapVec::new(fields_values_dim_bufman, fields_values_data_bufmans),
            fields: RwLock::new(VersionedVec::new(version)),
            fields_bufmans,
        })
    }

    fn update_fields_values(&self, version: VersionNumber, values: &FxHashMap<String, String>) {
        let stemmer = Stemmer::create();

        for (field, value) in values {
            // Check cardinality early
            if let Some(values) = self.fields_values.get(field) {
                if values.len() > MAX_CARDINALITY {
                    continue;
                }
            }

            // Acquire write lock only if necessary
            {
                let mut fields = self.fields.write().unwrap();
                if !fields.iter().any(|f| &f == field) {
                    fields.push(version, field.clone());
                }
            }

            let tokens: FxHashSet<_> = tokenize(value.as_str())
                .into_iter()
                .filter(|token| token.len() <= MAX_TOKEN_LEN)
                .map(|token| token.to_lowercase())
                .filter(|lower| !STOPWORDS.contains(&lower.as_str()))
                .map(|lower| stemmer.stem(&lower).to_string())
                .map(|stemmed| hash_word(&stemmed))
                .collect();

            for hash in tokens {
                // Re-check cardinality to avoid exceeding limit
                if let Some(values) = self.fields_values.get(field) {
                    if values.len() > MAX_CARDINALITY {
                        break;
                    }
                }

                let mut needs_insert = true;
                if let Some(mut values) = self.fields_values.get_mut(field) {
                    needs_insert = false;
                    if !values.iter().any(|v| v == hash) {
                        values.push(version, hash);
                    }
                }

                if needs_insert {
                    self.fields_values.push(version, field, hash);
                }
            }
        }
    }

    pub fn create_geofenced_document(
        &self,
        id: InternalId,
        zone_id: ZoneId,
        weight: f32,
        values: FxHashMap<String, String>,
        coordinates: (f32, f32),
        version: VersionNumber,
    ) -> Result<(), BufIoError> {
        self.update_fields_values(version, &values);

        self.zones.push(
            version,
            &zone_id,
            GeoFencedDocument {
                id,
                weight,
                coordinates,
            },
        );

        let terms = process_geo_fence_values(values, zone_id);

        for (term, score) in terms {
            self.root
                .insert(term, (score * 2.0).min(1.0), *id, version, 1.0)?;
        }

        Ok(())
    }

    pub fn insert(
        &self,
        id: InternalId,
        document: RawVectorEmbedding,
        values: FxHashMap<String, String>,
        version: VersionNumber,
    ) -> Result<(), BufIoError> {
        self.update_fields_values(version, &values);
        let terms = process_geo_fence_values(
            values,
            document.geo_fence_metadata.as_ref().unwrap().zone.clone(),
        );

        for (term, score) in terms {
            self.root.insert(term, score, *id, version, 1.0)?;
        }

        let terms = process_geo_fence_values(
            document.geo_fence_values.unwrap(),
            document.geo_fence_metadata.unwrap().zone,
        );

        for (term, score) in terms {
            self.root.insert(term, score, *id, version, 1.0)?;
        }

        Ok(())
    }
}

impl IndexOps for InvertedIndex {
    type IndexingInput = GeoFenceInputEmbedding;
    type SearchInput = SparseSearchInput;
    type SearchOptions = SparseSearchOptions;
    type Data = InvertedIndexData;

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
        embeddings
            .into_iter()
            .try_for_each(|GeoFenceInputEmbedding(id, document, values)| {
                self.insert(id, document, values, version)
            })?;
        Ok(())
    }

    fn delete_embedding(
        &self,
        _id: InternalId,
        _raw_emb: &RawVectorEmbedding,
        _version: VersionNumber,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
        todo!();
    }

    fn sample_embedding(&self, _embedding: &Self::IndexingInput) {
        todo!()
    }

    fn finalize_sampling(
        &self,
        _lmdb: &MetaDb,
        _config: &Config,
        _embeddings: &[Self::IndexingInput],
    ) -> Result<(), WaCustomError> {
        Ok(())
    }

    fn embeddings_collected(&self) -> &RwLock<Vec<Self::IndexingInput>> {
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

    fn flush(&self, _: &Collection, _version: VersionNumber) -> Result<(), WaCustomError> {
        self.root.serialize()?;
        self.zones.serialize()?;
        self.fields_values.serialize()?;
        self.root.cache.flush_all()?;
        let bufman = self.fields_bufmans.get(VersionNumber::from(0))?;
        let cursor = bufman.open_cursor()?;
        bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        let fields = self.fields.read().unwrap();
        bufman.update_u32_with_cursor(cursor, *fields.version)?;
        let offset =
            fields.serialize(&self.fields_values.dim_bufman, &self.fields_bufmans, cursor)?;
        drop(fields);
        bufman.seek_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, offset)?;
        bufman.close_cursor(cursor)?;
        self.fields_bufmans.flush_all()?;
        Ok(())
    }

    fn get_data(&self) -> Self::Data {
        Self::Data {
            quantization_bits: self.root.root.quantization_bits,
        }
    }

    fn search_internal(
        &self,
        _collection: &Collection,
        query: Self::SearchInput,
        options: &Self::SearchOptions,
        config: &Config,
        _return_raw_text: bool,
    ) -> Result<Vec<InternalSearchResult>, WaCustomError> {
        let stemmer = Stemmer::create();

        let mut filtered_tokens = Vec::new();

        for token in tokenize(&query.0) {
            if token.len() > MAX_TOKEN_LEN {
                continue;
            }

            let lower = token.to_lowercase();

            if STOPWORDS.contains(&lower.as_str()) {
                continue;
            }

            let stemmed = stemmer.stem(&lower).to_string();

            let hash = hash_word(&stemmed);

            filtered_tokens.push((stemmed, hash));
        }

        let mut sparse_pairs = Vec::new();

        let mut terms_info = FxHashMap::default();

        for field in self.fields.read().unwrap().iter() {
            if let Some(values) = self.fields_values.get(&field.to_string()) {
                let len = values.len();

                for (stemmed, hash) in &filtered_tokens {
                    if len < MAX_CARDINALITY && !values.iter().any(|value| value == *hash) {
                        continue;
                    }

                    for zone in &options.zones {
                        let mut hasher = XxHash32::with_seed(0);
                        zone.hash(&mut hasher);
                        field.hash(&mut hasher);
                        stemmed.hash(&mut hasher);
                        let hash = hasher.finish_32();
                        sparse_pairs.push((hash, 1.0));
                        terms_info.insert(hash, (field.clone(), stemmed));
                    }
                }
            }
        }

        let sparse_vec = SparseVector {
            vector_id: u32::MAX,
            entries: sparse_pairs,
        };

        let top_k = options.top_k.map(|top_k| top_k * 5);

        let results = SparseAnnQueryBasic::new(sparse_vec).search_with_matches(
            &self.root,
            self.root.root.quantization_bits,
            1.0,
            options
                .early_terminate_threshold
                .unwrap_or(config.search.early_terminate_threshold),
            if config.rerank_sparse_with_raw_values {
                config.sparse_raw_values_reranking_factor
            } else {
                1
            },
            top_k,
        )?;

        let alpha = 0.8_f32; // Use f32 for consistency with scores
        let max_score = ((1u32 << self.root.root.quantization_bits) - 1).pow(2) as f32;

        Ok(results
            .into_iter()
            .map(|result| {
                // First pass: Count token frequencies
                let mut token_counts = FxHashMap::<String, u32>::default();
                for (term_id, _) in &result.matches {
                    if let Some((_, token)) = terms_info.get(term_id) {
                        *token_counts.entry((*token).clone()).or_insert(0) += 1;
                    }
                }

                // Second pass: Process matches and calculate new total score
                let mut total_adjusted_score = 0.0;
                let mut matches: Matches = FxHashMap::default();

                for (term_id, raw_score) in result.matches {
                    if let Some((field, token)) = terms_info.get(&term_id).cloned() {
                        // Normalize raw score (0.0-1.0 range)
                        let normalized_score = raw_score as f32 / max_score;

                        // Get frequency count for this token
                        let count = token_counts.get(token).copied().unwrap_or(1) as f32;

                        let adjusted_score = normalized_score * count.powf(alpha);

                        // Add to total adjusted score
                        total_adjusted_score += adjusted_score;

                        // Store in matches map
                        matches
                            .entry(field)
                            .or_default()
                            .push((token.clone(), adjusted_score));
                    }
                }

                (
                    InternalId::from(result.vector_id),
                    None,
                    None,
                    total_adjusted_score,
                    None,
                    Some(matches),
                )
            })
            .collect())
    }

    fn remap_search_results(
        &self,
        collection: &Collection,
        results: Vec<InternalSearchResult>,
        options: &Self::SearchOptions,
        return_raw_text: bool,
    ) -> Result<Vec<SearchResult>, WaCustomError> {
        let documents = options
            .zones
            .iter()
            .flat_map(|zone| {
                self.zones
                    .get(zone)
                    .map(|vec| vec.iter().collect::<Vec<_>>())
            })
            .flatten()
            .flat_map(|document| {
                collection
                    .get_raw_emb_by_internal_id(&document.id)
                    .map(|emb| {
                        (
                            DocumentId::from((*emb.id).clone()),
                            emb.geo_fence_metadata.as_ref().unwrap().coordinates,
                        )
                    })
            })
            .collect::<FxHashMap<_, _>>();

        let results = results
            .into_iter()
            .map(|(internal_id, id, document_id, score, text, matches)| {
                Ok::<_, WaCustomError>(if let Some(id) = id {
                    (id, document_id, score, text, matches)
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
                        matches,
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();

        let mut results = if options.sort_by_distance {
            let mut results = results
                .filter_map(|result| {
                    result.1.and_then(|doc_id| {
                        documents.get(&doc_id).map(|coords| {
                            (
                                result.0,
                                Some(doc_id),
                                result.2,
                                result.3,
                                haversine_distance(
                                    coords.0,
                                    coords.1,
                                    options.coordinates.0,
                                    options.coordinates.1,
                                ),
                                result.4,
                            )
                        })
                    })
                })
                .collect::<Vec<_>>();

            results.sort_unstable_by(|a, b| b.4.total_cmp(&a.4));

            results
                .into_iter()
                .map(|result| (result.0, result.1, result.2, result.3, result.5))
                .collect()
        } else {
            let mut results = results
                .filter(|result| {
                    result
                        .1
                        .as_ref()
                        .is_some_and(|doc_id| documents.contains_key(doc_id))
                })
                .collect::<Vec<_>>();

            results.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));

            results
        };

        if let Some(top_k) = options.top_k {
            results.truncate(top_k);
        }

        Ok(results)
    }
}

/// Convert degrees to radians
fn deg2rad(deg: f32) -> f32 {
    deg * PI / 180.0
}

/// Haversine distance in kilometers between two lat/lng coordinates
pub fn haversine_distance(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let r = 6_371.0; // Earth radius in kilometers

    let dlat = deg2rad(lat2 - lat1);
    let dlon = deg2rad(lon2 - lon1);

    let lat1 = deg2rad(lat1);
    let lat2 = deg2rad(lat2);

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);

    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    r * c
}

const STOPWORDS: [&str; 35] = [
    "a", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no",
    "not", "of", "on", "or", "s", "such", "t", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "www",
];

fn process_geo_fence_values(values: FxHashMap<String, String>, zone: ZoneId) -> Vec<(u32, f32)> {
    let stemmer = Stemmer::create();

    let mut terms = Vec::new();

    for (field, value) in values {
        let num_tokens = count_tokens(&value);
        let alpha = 0.8_f32;
        let score = 1.0 / (num_tokens as f32).powf(alpha);

        let mut tokens = FxHashSet::default();

        for token in tokenize(&value) {
            if token.len() > MAX_TOKEN_LEN {
                continue;
            }

            let lower = token.to_lowercase();

            if STOPWORDS.contains(&lower.as_str()) {
                continue;
            }

            let stemmed = stemmer.stem(&lower).to_string();

            if !tokens.insert(stemmed.clone()) {
                continue;
            }

            let mut hasher = XxHash32::with_seed(0);
            zone.hash(&mut hasher);
            field.hash(&mut hasher);
            stemmed.hash(&mut hasher);

            let token_hash = hasher.finish_32();

            terms.push((token_hash, score));
        }
    }

    terms
}

fn hash_word(word: &str) -> u64 {
    let mut hasher = FxHasher::default();
    word.hash(&mut hasher);
    hasher.finish() & (u64::MAX >> 1)
}

pub fn count_tokens(input: &str) -> u32 {
    let mut count = 0;

    for token in tokenize(input) {
        if token.len() > MAX_TOKEN_LEN {
            continue;
        }

        let lower = token.to_lowercase();
        if STOPWORDS.contains(&lower.as_str()) {
            continue;
        }
        count += 1;
    }

    count
}
