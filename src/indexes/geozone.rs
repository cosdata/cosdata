use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::{
    inverted::{types::SparsePair, SparseInputEmbedding, SparseSearchInput},
    IndexOps, InternalSearchResult, SearchResult,
};
use crate::{
    config_loader::Config,
    models::{
        buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
        collection::{Collection, RawVectorEmbedding},
        common::WaCustomError,
        inverted_index::InvertedIndexRoot,
        sparse_ann_query::{SparseAnnQueryBasic, SparseAnnResult},
        tree_map::{TreeMapKey, TreeMapVec},
        types::{DocumentId, InternalId, MetaDb, SparseVector},
        versioned_vec::VersionedVecItem,
        versioning::VersionNumber,
    },
};
use std::{
    fs::OpenOptions,
    hash::{DefaultHasher, Hasher},
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

    fn id(storage: Self::Storage) -> Self::Id {
        (storage >> 96) as u32
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

pub struct GeoZoneSearchOptions {
    pub top_k: Option<usize>,
    pub early_terminate_threshold: Option<f32>,
    pub sort_by_distance: bool,
    pub coordinates: (f32, f32),
    pub zones: Vec<ZoneId>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct GeoZoneIndexData {
    pub quantization_bits: u8,
}

pub struct GeoZoneIndex {
    pub root: InvertedIndexRoot,
    pub zones: TreeMapVec<ZoneId, GeoFencedDocument>,
}

unsafe impl Send for GeoZoneIndex {}
unsafe impl Sync for GeoZoneIndex {}

impl GeoZoneIndex {
    pub fn new(root_path: PathBuf, quantization_bits: u8) -> Result<Self, BufIoError> {
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
            root_path.into(),
            |root, version: &VersionNumber| root.join(format!("zones.{}.data", **version)),
            8192,
        );

        Ok(Self {
            root,
            zones: TreeMapVec::new(zones_dim_bufman, zones_data_bufmans),
        })
    }

    pub fn create_geofenced_document(
        &self,
        id: InternalId,
        zone_id: ZoneId,
        weight: f32,
        pairs: Vec<SparsePair>,
        coordinates: (f32, f32),
        version: VersionNumber,
    ) -> Result<(), BufIoError> {
        for pair in pairs {
            self.root.insert(pair.0, pair.1, *id, version, 1.0)?;
        }

        self.zones.push(
            version,
            &zone_id,
            GeoFencedDocument {
                id,
                weight,
                coordinates,
            },
        );

        Ok(())
    }

    pub fn insert(
        &self,
        id: InternalId,
        pairs: Vec<SparsePair>,
        version: VersionNumber,
    ) -> Result<(), BufIoError> {
        for pair in pairs {
            self.root.insert(pair.0, pair.1, *id, version, 1.0)?;
        }

        Ok(())
    }
}

impl IndexOps for GeoZoneIndex {
    type IndexingInput = SparseInputEmbedding;
    type SearchInput = SparseSearchInput;
    type SearchOptions = GeoZoneSearchOptions;
    type Data = GeoZoneIndexData;

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
            .try_for_each(|SparseInputEmbedding(id, pairs)| self.insert(id, pairs, version))?;
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
        self.root.cache.flush_all()?;
        Ok(())
    }

    fn get_data(&self) -> Self::Data {
        Self::Data {
            quantization_bits: self.root.root.quantization_bits,
        }
    }

    fn search_internal(
        &self,
        collection: &Collection,
        query: Self::SearchInput,
        options: &Self::SearchOptions,
        config: &Config,
        return_raw_text: bool,
    ) -> Result<Vec<InternalSearchResult>, WaCustomError> {
        let sparse_vec = SparseVector {
            vector_id: u32::MAX,
            entries: query.0.iter().map(|pair| (pair.0, pair.1)).collect(),
        };

        let top_k = options.top_k.map(|top_k| top_k * 5);

        let results = SparseAnnQueryBasic::new(sparse_vec).sequential_search(
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

        if config.rerank_sparse_with_raw_values {
            finalize_sparse_ann_results(collection, results, &query.0, top_k, return_raw_text)
        } else {
            Ok(results
                .into_iter()
                .map(|result| {
                    (
                        InternalId::from(result.vector_id),
                        None,
                        None,
                        result.similarity as f32,
                        None,
                    )
                })
                .collect())
        }
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
            .map(|(internal_id, id, document_id, score, text)| {
                Ok::<_, WaCustomError>(if let Some(id) = id {
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
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();

        let results = if options.sort_by_distance {
            let mut results = results
                .filter_map(|result| {
                    result.1.and_then(|doc_id| {
                        documents.get(&doc_id).map(|coords| {
                            (
                                result.0,
                                Some(doc_id),
                                result.2,
                                result.3,
                                euclidean_distance(*coords, options.coordinates),
                            )
                        })
                    })
                })
                .collect::<Vec<_>>();

            results.sort_unstable_by(|a, b| a.4.total_cmp(&b.4));

            results
                .into_iter()
                .map(|result| (result.0, result.1, result.2, result.3))
                .collect()
        } else {
            results
                .filter(|result| {
                    result
                        .1
                        .as_ref()
                        .is_some_and(|doc_id| documents.contains_key(doc_id))
                })
                .collect::<Vec<_>>()
        };

        Ok(results)
    }
}

fn finalize_sparse_ann_results(
    collection: &Collection,
    intermediate_results: Vec<SparseAnnResult>,
    query: &[SparsePair],
    k: Option<usize>,
    return_raw_text: bool,
) -> Result<Vec<InternalSearchResult>, WaCustomError> {
    let mut results = Vec::with_capacity(k.unwrap_or(intermediate_results.len()));

    for result in intermediate_results {
        let internal_id = InternalId::from(result.vector_id);

        let raw_embedding_ref = collection
            .get_raw_emb_by_internal_id(&internal_id)
            .ok_or_else(|| WaCustomError::NotFound("raw embedding not found".to_string()))?;

        let sparse_pairs = raw_embedding_ref.sparse_values.clone().ok_or_else(|| {
            WaCustomError::NotFound("sparse values is missing in raw embedding".to_string())
        })?;
        let map: std::collections::HashMap<u32, f32> =
            sparse_pairs.iter().map(|sp| (sp.0, sp.1)).collect();

        let mut dp = 0.0;
        for pair in query {
            if let Some(val) = map.get(&pair.0) {
                dp += val * pair.1;
            }
        }

        results.push((
            internal_id,
            Some(raw_embedding_ref.id.clone()),
            raw_embedding_ref.document_id.clone(),
            dp,
            if return_raw_text {
                raw_embedding_ref.text.clone()
            } else {
                None
            },
        ));
    }

    // Sort by descending order
    results.sort_unstable_by(|(_, _, _, a, _), (_, _, _, b, _)| b.total_cmp(a));

    if let Some(k_val) = k {
        results.truncate(k_val);
    }

    Ok(results)
}

fn euclidean_distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}
