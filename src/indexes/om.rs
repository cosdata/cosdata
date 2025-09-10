use std::{
    hash::{Hash, Hasher},
    path::Path,
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc, Arc, RwLock,
    },
    thread,
    time::Instant,
};

use parking_lot::MappedRwLockReadGuard;
use rustc_hash::FxHasher;

use crate::{
    config_loader::Config,
    models::{
        buffered_io::{BufIoError, BufferManager},
        collection::{Collection, OmVectorEmbedding, RawVectorEmbedding},
        common::{TSHashTable, WaCustomError},
        om_tree_map::{OmTreeMapKey, ShardIndex, ShardedOmTreeMap},
        serializer::InlineSerialize,
        types::{InternalId, MetaDb},
        versioning::VersionNumber,
    },
};

use super::{IndexOps, InternalSearchResult};

// hash(metric=value)
#[derive(Debug, Clone, Copy)]
pub struct MetricHash(u32);

#[derive(Debug, Clone, Copy)]
pub struct FullHash(u64);

impl OmTreeMapKey for MetricHash {
    fn primary_key(&self) -> u16 {
        (self.0 >> 12) as u16
    }

    fn secondary_key(&self) -> u64 {
        (self.0 & 0xFFF) as u64
    }
}

impl ShardIndex for MetricHash {
    fn shard_index(&self) -> usize {
        (self.0 >> 28) as usize
    }
}

impl OmTreeMapKey for FullHash {
    fn primary_key(&self) -> u16 {
        self.0 as u16
    }

    fn secondary_key(&self) -> u64 {
        self.0
    }
}

impl ShardIndex for FullHash {
    fn shard_index(&self) -> usize {
        (self.0 >> 60) as usize
    }
}

impl InlineSerialize for FullHash {
    const SIZE: usize = 8;

    fn serialize(&self, vec: &mut Vec<u8>, _bufman: &BufferManager) -> Result<(), BufIoError> {
        vec.extend(self.0.to_le_bytes());
        Ok(())
    }

    fn deserialize(bufman: &BufferManager, cursor: u64) -> Result<Self, BufIoError> {
        bufman.read_u64_with_cursor(cursor).map(Self)
    }
}

pub struct OmIndex {
    pub id_to_values_map: Arc<ShardedOmTreeMap<FullHash, f32>>,
    pub label_to_ids_map: Arc<ShardedOmTreeMap<MetricHash, u32>>,
    pub id_mapping: TSHashTable<u32, FullHash>,
    pub id_counter: AtomicU32,
    pub serialization_channel: mpsc::Sender<VersionNumber>,
}

impl OmIndex {
    pub fn new(index_path: &Path) -> Result<Self, WaCustomError> {
        let id_to_values_map = Arc::new(ShardedOmTreeMap::new(index_path.into(), 16, "itov")?);
        let label_to_ids_map = Arc::new(ShardedOmTreeMap::new(index_path.into(), 16, "ltoi")?);

        let (tx, rx) = mpsc::channel::<VersionNumber>();

        {
            let id_to_values_map = id_to_values_map.clone();
            let label_to_ids_map = label_to_ids_map.clone();

            thread::spawn(move || {
                for version in rx {
                    println!("serializing version: {}", *version);
                    rayon::join(
                        || {
                            let start = Instant::now();
                            id_to_values_map.serialize(version).unwrap();
                            println!("serialized key value store in {:?}", start.elapsed());
                        },
                        || {
                            let start = Instant::now();
                            label_to_ids_map.serialize(version).unwrap();
                            println!("serialized inverted index in {:?}", start.elapsed());
                        },
                    );
                    println!("serialized version: {}", *version);
                }
            });
        }

        Ok(Self {
            id_to_values_map,
            label_to_ids_map,
            id_mapping: TSHashTable::new(64),
            id_counter: AtomicU32::new(0),
            serialization_channel: tx,
        })
    }

    pub fn insert(&self, version: VersionNumber, key: &[u32], value: f32) {
        let hash = compute_hash(key);
        let newly_created = self.id_to_values_map.push(version, &hash, value);

        if newly_created {
            let id = self.id_counter.fetch_add(1, Ordering::AcqRel);
            self.id_mapping.insert(id, hash);
            for hash in key {
                let metric_hash = MetricHash(*hash);
                self.label_to_ids_map.push_sorted(version, &metric_hash, id);
            }
        }
    }

    pub fn sum(&self, labels: &[u32]) -> f32 {
        let mut vecs = Vec::new();

        for &label in labels {
            let metric_hash = MetricHash(label);
            if let Some(vec) = self.label_to_ids_map.get(&metric_hash) {
                vecs.push(vec);
            };
        }

        let keys = intersect_sorted_vecs(vecs);

        let mut sum = 0.0;

        for id in keys {
            let Some(hash) = self.id_mapping.lookup(&id) else {
                continue;
            };
            let Some(val) = self.id_to_values_map.get(&hash) else {
                continue;
            };

            for &val in &*val {
                sum += val;
            }
        }

        sum
    }
}

fn intersect_sorted_vecs(vecs: Vec<MappedRwLockReadGuard<'_, Vec<u32>>>) -> Vec<u32> {
    if vecs.is_empty() {
        return Vec::new();
    }
    let mut vecs = vecs;

    // Sort vectors by length to process shortest first
    vecs.sort_by_key(|v| v.len());

    let mut candidate = vecs.remove(0).clone();
    let mut temp = Vec::new();

    for vec in vecs {
        let (mut i, mut j) = (0, 0);
        while i < candidate.len() && j < vec.len() {
            match candidate[i].cmp(&vec[j]) {
                std::cmp::Ordering::Equal => {
                    temp.push(candidate[i]);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        std::mem::swap(&mut candidate, &mut temp);
        temp.clear();
        if candidate.is_empty() {
            break;
        }
    }

    candidate
}

fn compute_hash(key: &[u32]) -> FullHash {
    let mut hasher = FxHasher::default();
    key.hash(&mut hasher);
    FullHash(hasher.finish())
}

impl IndexOps for OmIndex {
    type IndexingInput = OmVectorEmbedding;
    type SearchInput = Vec<u32>;
    type SearchOptions = ();
    type Data = ();

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
            self.insert(version, &emb.key, emb.value);
        }
        Ok(())
    }

    fn delete_embedding(
        &self,
        _id: InternalId,
        _raw_emb: &RawVectorEmbedding,
        _version: VersionNumber,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
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

    fn sample_embedding(&self, _embedding: &Self::IndexingInput) {}

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

    fn flush(&self, _collection: &Collection, version: VersionNumber) -> Result<(), WaCustomError> {
        self.serialization_channel.send(version).unwrap();
        self.id_counter.store(0, Ordering::Release);
        self.id_mapping.clear();
        Ok(())
    }

    fn get_data(&self) -> Self::Data {}

    fn search_internal(
        &self,
        _collection: &Collection,
        _query: Self::SearchInput,
        _options: &Self::SearchOptions,
        _config: &Config,
        _return_raw_text: bool,
    ) -> Result<Vec<InternalSearchResult>, WaCustomError> {
        todo!()
    }
}
