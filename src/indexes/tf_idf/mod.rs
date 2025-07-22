use super::{IndexOps, InternalSearchResult};
use crate::{
    config_loader::Config,
    models::{
        buffered_io::BufIoError,
        collection::{Collection, RawVectorEmbedding},
        common::WaCustomError,
        meta_persist::store_average_document_length,
        sparse_ann_query::SparseAnnQueryBasic,
        tf_idf_index::TFIDFIndexRoot,
        types::{InternalId, MetaDb, SparseVector},
        versioning::VersionNumber,
    },
};
use rustc_hash::FxHashMap;
use snowball_stemmer::Stemmer;
use std::{
    hash::Hasher,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        RwLock,
    },
};
use twox_hash::XxHash32;

#[derive(Default)]
pub struct SamplingData {
    pub total_documents_length: AtomicU64,
    pub total_documents_count: AtomicU32,
}

pub struct TFIDFInputEmbedding(pub InternalId, pub String);

pub struct TFIDFSearchInput(pub String);

pub struct TFIDFSearchOptions {
    pub top_k: Option<usize>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TFIDFIndexData {
    pub sample_threshold: usize,
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
    pub k1: f32,
    pub b: f32,
}

unsafe impl Send for TFIDFIndex {}
unsafe impl Sync for TFIDFIndex {}

impl TFIDFIndex {
    pub fn new(
        root_path: PathBuf,
        sample_threshold: usize,
        k1: f32,
        b: f32,
    ) -> Result<Self, BufIoError> {
        let root = TFIDFIndexRoot::new(root_path)?;

        Ok(Self {
            root,
            average_document_length: RwLock::new(1.0),
            is_configured: AtomicBool::new(false),
            sampling_data: SamplingData::default(),
            documents: RwLock::new(Vec::new()),
            documents_collected: AtomicUsize::new(0),
            sample_threshold,
            k1,
            b,
        })
    }

    pub fn insert(
        &self,
        version: VersionNumber,
        id: InternalId,
        text: String,
    ) -> Result<(), BufIoError> {
        self.root
            .total_documents_count
            .fetch_add(1, Ordering::Relaxed);

        let terms = process_text(
            &text,
            40,
            *self.average_document_length.read().unwrap(),
            self.k1,
            self.b,
        );

        let id = id.into();

        for (term_hash, tf) in terms {
            self.root.insert(term_hash, tf, id, version)?;
        }

        Ok(())
    }

    pub fn mark_embedding_as_deleted(
        &self,
        version: VersionNumber,
        id: InternalId,
        text: &str,
    ) -> Result<(), BufIoError> {
        self.root
            .total_documents_count
            .fetch_sub(1, Ordering::Relaxed);

        let terms = process_text(
            text,
            40,
            *self.average_document_length.read().unwrap(),
            self.k1,
            self.b,
        );

        let id = id.into();

        for (term_hash, _) in terms {
            self.root.delete(term_hash, id, version)?;
        }

        Ok(())
    }
}

impl IndexOps for TFIDFIndex {
    type IndexingInput = TFIDFInputEmbedding;
    type SearchInput = TFIDFSearchInput;
    type SearchOptions = TFIDFSearchOptions;
    type Data = TFIDFIndexData;

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
            .try_for_each(|TFIDFInputEmbedding(id, text)| self.insert(version, id, text))?;
        Ok(())
    }

    fn delete_embedding(
        &self,
        id: InternalId,
        raw_emb: &RawVectorEmbedding,
        version: VersionNumber,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
        let Some(text) = &raw_emb.text else {
            return Ok(());
        };
        self.mark_embedding_as_deleted(version, id, text)?;
        Ok(())
    }

    fn sample_embedding(&self, embedding: &Self::IndexingInput) {
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
        _embeddings: &[Self::IndexingInput],
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

    fn embeddings_collected(&self) -> &RwLock<Vec<Self::IndexingInput>> {
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

    fn flush(
        &self,
        _collection: &Collection,
        _version: VersionNumber,
    ) -> Result<(), WaCustomError> {
        self.root.serialize()?;
        self.root.cache.flush_all()?;
        Ok(())
    }

    fn get_data(&self) -> Self::Data {
        Self::Data {
            sample_threshold: self.sample_threshold,
            k1: self.k1,
            b: self.b,
        }
    }

    fn search_internal(
        &self,
        _collection: &Collection,
        query: Self::SearchInput,
        options: &Self::SearchOptions,
        _config: &Config,
        _return_raw_text: bool,
    ) -> Result<Vec<InternalSearchResult>, WaCustomError> {
        let entries = process_text(
            &query.0,
            40,
            *self.average_document_length.read().unwrap(),
            self.k1,
            self.b,
        );

        let sparse_vec = SparseVector {
            vector_id: u32::MAX,
            entries,
        };

        let results =
            SparseAnnQueryBasic::new(sparse_vec).search_bm25(&self.root, options.top_k)?;

        Ok(results
            .into_iter()
            .map(|result| {
                (
                    InternalId::from(result.document_id),
                    None,
                    None,
                    result.score,
                    None,
                )
            })
            .collect())
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
