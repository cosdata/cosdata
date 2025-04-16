pub(crate) mod data;
pub(crate) mod transaction;
use lmdb::Transaction;

use std::{
    hash::Hasher,
    path::PathBuf,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        RwLock,
    },
};

use rustc_hash::FxHashMap;
use snowball_stemmer::Stemmer;
use transaction::TFIDFIndexTransaction;
use twox_hash::XxHash32;

use crate::macros::key;
use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    tf_idf_index::TFIDFIndexRoot,
    tree_map::TreeMap,
    types::{MetaDb, VectorId},
    versioning::{Hash, VersionControl},
};

#[derive(Default)]
pub struct SamplingData {
    pub total_documents_length: AtomicU64,
    pub total_documents_count: AtomicU32,
}

pub struct TFIDFIndex {
    pub name: String,
    pub description: Option<String>,
    pub max_vectors: Option<i32>,
    pub root: TFIDFIndexRoot,
    pub lmdb: MetaDb,
    pub current_version: RwLock<Hash>,
    pub current_open_transaction: AtomicPtr<TFIDFIndexTransaction>,
    pub vcs: VersionControl,
    pub average_document_length: RwLock<f32>,
    pub is_configured: AtomicBool,
    pub documents: RwLock<Vec<(VectorId, String)>>,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        description: Option<String>,
        root_path: PathBuf,
        max_vectors: Option<i32>,
        lmdb: MetaDb,
        current_version: Hash,
        vcs: VersionControl,
        vec_raw_manager: BufferManagerFactory<u8>,
        data_file_parts: u8,
        sample_threshold: usize,
        store_raw_text: bool,
        k1: f32,
        b: f32,
    ) -> Result<Self, BufIoError> {
        let root = TFIDFIndexRoot::new(root_path, data_file_parts)?;

        Ok(Self {
            name,
            description,
            max_vectors,
            root,
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: AtomicPtr::new(ptr::null_mut()),
            vcs,
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

    pub fn set_current_version(&self, version: Hash) {
        *self.current_version.write().unwrap() = version;
    }

    pub fn contains_vector_id(&self, vector_id_u32: u32) -> bool {
        let env = self.lmdb.env.clone();
        let db = *self.lmdb.db;
        let txn = match env.begin_ro_txn() {
            Ok(txn) => txn,
            Err(e) => {
                log::error!("LMDB RO txn failed for IDF contains_vector_id check: {}", e);
                return false;
            }
        };

        let vector_id_obj = VectorId(vector_id_u32 as u64);
        let embedding_key = key!(e: &vector_id_obj);

        let found = match txn.get(db, &embedding_key) {
            Ok(_) => true,
            Err(lmdb::Error::NotFound) => false,
            Err(e) => {
                log::error!(
                    "LMDB error during IDF contains_vector_id get for {}: {}",
                    vector_id_u32,
                    e
                );
                false
            }
        };

        txn.abort();
        found
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
