use rayon::iter::{IntoParallelIterator, ParallelIterator};

use std::{
    hash::Hasher,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        RwLock,
    },
};

use rustc_hash::FxHashMap;
use snowball_stemmer::Stemmer;
use twox_hash::XxHash32;

use crate::{
    config_loader::Config,
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        collection::Collection,
        collection_transaction::CollectionTransaction,
        common::WaCustomError,
        meta_persist::{store_average_document_length, store_highest_internal_id},
        tf_idf_index::TFIDFIndexRoot,
        tree_map::TreeMap,
        types::{MetaDb, VectorId},
        versioning::Hash,
    },
};

use super::IndexOps;

#[derive(Default)]
pub struct SamplingData {
    pub total_documents_length: AtomicU64,
    pub total_documents_count: AtomicU32,
}

pub struct TFIDFInputEmbedding(pub VectorId, pub String);

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TFIDFIndexData {
    pub sample_threshold: usize,
    pub store_raw_text: bool,
    pub k1: f32,
    pub b: f32,
}
pub struct TFIDFIndex {
    pub root: TFIDFIndexRoot,
    pub average_document_length: RwLock<f32>,
    pub is_configured: AtomicBool,
    pub documents: RwLock<Vec<TFIDFInputEmbedding>>,
    pub documents_collected: AtomicUsize,
    pub sampling_data: SamplingData,
    pub sample_threshold: usize,
    pub vec_raw_manager: BufferManagerFactory<u8>,
    pub vec_raw_map: TreeMap<(VectorId, Option<String>)>,
    pub document_id_counter: AtomicU32,
    pub store_raw_text: bool,
    pub k1: f32,
    pub b: f32,
}

unsafe impl Send for TFIDFIndex {}
unsafe impl Sync for TFIDFIndex {}

impl TFIDFIndex {
    pub fn new(
        root_path: PathBuf,
        vec_raw_manager: BufferManagerFactory<u8>,
        data_file_parts: u8,
        sample_threshold: usize,
        store_raw_text: bool,
        k1: f32,
        b: f32,
    ) -> Result<Self, BufIoError> {
        let root = TFIDFIndexRoot::new(root_path, data_file_parts)?;

        Ok(Self {
            root,
            average_document_length: RwLock::new(1.0),
            is_configured: AtomicBool::new(false),
            sampling_data: SamplingData::default(),
            documents: RwLock::new(Vec::new()),
            documents_collected: AtomicUsize::new(0),
            sample_threshold,
            vec_raw_manager,
            vec_raw_map: TreeMap::new(),
            document_id_counter: AtomicU32::new(0),
            store_raw_text,
            k1,
            b,
        })
    }

    pub fn insert(&self, version: Hash, ext_id: VectorId, text: String) -> Result<(), BufIoError> {
        self.root
            .total_documents_count
            .fetch_add(1, Ordering::Relaxed);

        let document_id = self.document_id_counter.fetch_add(1, Ordering::Relaxed);
        let terms = process_text(
            &text,
            40,
            *self.average_document_length.read().unwrap(),
            self.k1,
            self.b,
        );

        for (term_hash, tf) in terms {
            self.root.insert(term_hash, tf, document_id, version)?;
        }

        self.vec_raw_map.insert(
            version,
            document_id as u64,
            (ext_id, self.store_raw_text.then_some(text)),
        );

        Ok(())
    }
}

impl IndexOps for TFIDFIndex {
    type InputEmbedding = TFIDFInputEmbedding;
    type Data = TFIDFIndexData;

    fn index_embeddings(
        &self,
        _collection: &Collection,
        embeddings: Vec<Self::InputEmbedding>,
        transaction: &CollectionTransaction,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
        embeddings
            .into_par_iter()
            .try_for_each(|TFIDFInputEmbedding(id, text)| self.insert(transaction.id, id, text))?;
        Ok(())
    }

    fn sample_embedding(&self, embedding: &Self::InputEmbedding) {
        let len = count_tokens(&embedding.1, 40);
        self.sampling_data
            .total_documents_length
            .fetch_add(len as u64, Ordering::Relaxed);
        self.sampling_data
            .total_documents_count
            .fetch_add(1, Ordering::Relaxed);
    }

    fn finalize_sampling(
        &self,
        lmdb: &MetaDb,
        _config: &Config,
        _embeddings: &[Self::InputEmbedding],
    ) -> Result<(), WaCustomError> {
        let total_documents_count = self
            .sampling_data
            .total_documents_count
            .load(Ordering::Relaxed);
        let total_documents_length = self
            .sampling_data
            .total_documents_length
            .load(Ordering::Relaxed);

        let avg_length = total_documents_length as f32 / total_documents_count as f32;
        *self.average_document_length.write().unwrap() = avg_length;
        self.is_configured.store(true, Ordering::Release);
        store_average_document_length(lmdb, avg_length)?;
        Ok(())
    }

    fn embeddings_collected(&self) -> &RwLock<Vec<Self::InputEmbedding>> {
        &self.documents
    }

    fn increment_collected_count(&self, count: usize) -> usize {
        self.documents_collected.fetch_add(count, Ordering::SeqCst)
    }

    fn sample_threshold(&self) -> usize {
        self.sample_threshold
    }

    fn is_configured(&self) -> bool {
        self.is_configured.load(Ordering::Acquire)
    }

    fn flush(&self, collection: &Collection) -> Result<(), WaCustomError> {
        store_highest_internal_id(
            &collection.lmdb,
            self.document_id_counter.load(Ordering::Relaxed),
        )?;
        self.vec_raw_map
            .serialize(&self.vec_raw_manager, self.root.data_file_parts)?;
        self.root.serialize()?;
        self.root.cache.flush_all()?;
        Ok(())
    }

    fn get_data(&self) -> Self::Data {
        Self::Data {
            sample_threshold: self.sample_threshold,
            store_raw_text: self.store_raw_text,
            k1: self.k1,
            b: self.b,
        }
    }
}

const STOPWORDS: [&str; 35] = [
    "a", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no",
    "not", "of", "on", "or", "s", "such", "t", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "www",
];

pub fn tokenize(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = None;

    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() || c == '_' {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            result.push(&text[s..i]);
            start = None;
        }
    }

    if let Some(s) = start {
        result.push(&text[s..]);
    }

    result
}

pub fn process_text(
    input: &str,
    max_token_len: usize,
    average_document_length: f32,
    k1: f32,
    b: f32,
) -> Vec<(u32, f32)> {
    // Create an English stemmer.
    let stemmer = Stemmer::create();
    // Create a fast hash map for counting; FxHashMap is chosen for performance.
    let mut freq: FxHashMap<u32, u32> = FxHashMap::default();
    let document_length = count_tokens(input, max_token_len);

    // Split input by whitespace to minimize unnecessary cloning.
    for token in tokenize(input) {
        if token.len() > max_token_len {
            continue;
        }

        // Convert to lowercase.
        // Since to_lowercase allocates a new String, this is necessary for case insensitive matching.
        let lower = token.to_lowercase();

        // Skip token if it is a stopword.
        if STOPWORDS.contains(&lower.as_str()) {
            continue;
        }

        // Stem the token.
        // Note: stemmer.stem returns a Cow<&str>; we call to_string() to get an owned String.
        let stemmed = stemmer.stem(&lower).to_string();

        // Hash the stemmed token using xxhash32.
        let mut hasher = XxHash32::with_seed(0);
        hasher.write(stemmed.as_bytes());
        let token_hash = hasher.finish() as u32;

        // Increment the count for this hash.
        *freq.entry(token_hash).or_insert(0) += 1;
    }

    // Convert the hash map into a Vec of (hash, count) pairs.
    freq.into_iter()
        .map(|(hash, count)| {
            (
                hash,
                compute_bm25_term_frequency(count, document_length, average_document_length, k1, b),
            )
        })
        .collect()
}

fn compute_bm25_term_frequency(
    count: u32,
    document_length: u32,
    average_document_length: f32,
    k1: f32,
    b: f32,
) -> f32 {
    count as f32 * (k1 + 1.0)
        / (count as f32 + k1 * (1.0 - b + b * (document_length as f32 / average_document_length)))
}

pub fn count_tokens(input: &str, max_token_len: usize) -> u32 {
    let mut count = 0;

    for token in tokenize(input) {
        if token.len() > max_token_len {
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
