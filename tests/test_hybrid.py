import csv
import getpass
import json
import math
import os
import pickle
import psutil
import re
import numpy as np
import sys
import threading
import time
import unicodedata
import urllib3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
from tqdm import tqdm

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from dotenv import load_dotenv
from fastembed import TextEmbedding
from py_rust_stemmers import SnowballStemmer
import requests

from cosdata import Client

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class CPUMonitor:
    def __init__(self):
        self.process = None
        self.monitoring = False
        self.cpu_samples = []
        self.monitor_thread = None

    def find_cosdata_process(self):
        for proc in psutil.process_iter(["pid", "name", "cmdline", "exe"]):
            try:
                # Check process name
                if proc.info["name"] == "cosdata":
                    return psutil.Process(proc.info["pid"])

                # Check command line arguments
                if proc.info["cmdline"]:
                    cmdline_str = " ".join(proc.info["cmdline"])
                    if "cosdata" in cmdline_str and "--admin-key" in cmdline_str:
                        return psutil.Process(proc.info["pid"])

                # Check executable path
                if proc.info.get("exe") and "cosdata" in proc.info["exe"]:
                    return psutil.Process(proc.info["pid"])

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return None

    def start_monitoring(self, sample_interval=0.1):
        self.process = self.find_cosdata_process()
        if not self.process:
            print("Warning: Could not find cosdata process for CPU monitoring")
            print("Available processes with 'cosdata' in name or cmdline:")
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if "cosdata" in proc.info["name"].lower() or (
                        proc.info["cmdline"]
                        and any(
                            "cosdata" in str(arg).lower()
                            for arg in proc.info["cmdline"]
                        )
                    ):
                        print(
                            f"  PID {proc.info['pid']}: {proc.info['name']} - {' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 'N/A'}"
                        )
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    pass
            return False

        print(
            f"Found cosdata process: PID {self.process.pid}, name: {self.process.name()}"
        )
        self.monitoring = True
        self.cpu_samples = []

        def monitor_cpu():
            # Get initial CPU reading to establish baseline
            if self.process and self.process.is_running():
                try:
                    initial_cpu = self.process.cpu_percent()
                    print(f"Initial CPU reading: {initial_cpu}%")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            while self.monitoring:
                try:
                    if self.process and self.process.is_running():
                        cpu = self.process.cpu_percent()
                        self.cpu_samples.append(cpu)
                        if (
                            len(self.cpu_samples) % 50 == 0
                        ):  # Log every 5 seconds at 0.1s intervals
                            print(
                                f"CPU monitoring: {len(self.cpu_samples)} samples, latest: {cpu}%"
                            )
                    time.sleep(sample_interval)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print("Lost connection to cosdata process during monitoring")
                    break

        self.monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
        self.monitor_thread.start()
        return True

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        print(f"CPU monitoring stopped. Collected {len(self.cpu_samples)} samples")
        if not self.cpu_samples:
            return {"max_cpu": 0.0, "avg_cpu": 0.0, "samples": 0}

        max_cpu = max(self.cpu_samples)
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
        print(
            f"CPU stats: max={max_cpu:.1f}%, avg={avg_cpu:.1f}%, samples={len(self.cpu_samples)}"
        )

        return {
            "max_cpu": max_cpu,
            "avg_cpu": avg_cpu,
            "samples": len(self.cpu_samples),
        }


class QPSTracker:
    def __init__(self):
        self.query_times = []
        self.total_queries = 0
        self.failed_queries = 0
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()

    def start_tracking(self):
        self.start_time = time.time()
        self.query_times = []
        self.total_queries = 0
        self.failed_queries = 0

    def record_query_batch(self, batch_size, batch_duration, failed_count=0):
        with self.lock:
            self.total_queries += batch_size
            self.failed_queries += failed_count
            avg_query_time = batch_duration / batch_size if batch_size > 0 else 0
            self.query_times.extend([avg_query_time] * (batch_size - failed_count))

    def stop_tracking(self):
        self.end_time = time.time()
        return self.get_metrics()

    def get_metrics(self):
        if self.start_time is None or self.end_time is None:
            return {
                "qps": 0.0,
                "avg_query_time": 0.0,
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "total_duration": 0.0,
            }

        total_duration = self.end_time - self.start_time
        successful_queries = self.total_queries - self.failed_queries

        qps = successful_queries / total_duration if total_duration > 0 else 0
        avg_query_time = (
            sum(self.query_times) / len(self.query_times) if self.query_times else 0
        )

        return {
            "qps": qps,
            "avg_query_time": avg_query_time * 1000,
            "total_queries": self.total_queries,
            "successful_queries": successful_queries,
            "failed_queries": self.failed_queries,
            "total_duration": total_duration,
        }

DEFAULT_HOST = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
DEFAULT_USER = os.getenv("COSDATA_USERNAME", "admin")
DEFAULT_PASS = os.getenv("COSDATA_PASSWORD", None)

COLLECTION_PREFIX = "hybrid_raw"
BATCH_UPSERT_SIZE = 1000
SEARCH_BATCH_SIZE = 100
TOP_K = 10
MAX_WORKERS = 32

# dense
DIMENSION = 768
EMBEDDING_BATCH_SIZE = 64

# sparse
K1 = 1.5
B = 0.75

STOPWORDS = {
    "a", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "s", "such", "t", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with", "www",
}
PUNCT = set(
    chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Normalize unicode and clean text
        text = unicodedata.normalize("NFD", text)
        text = re.sub(r"[^\w\s]", " ", text.lower())
        # Split and filter
        tokens = [
            token
            for token in text.split()
            if len(token) > 1 and token not in STOPWORDS
        ]
        return tokens


def compute_bm25_idf(total_docs: int, docs_with_term: int) -> float:
    return math.log1p((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5))


def compute_bm25_tf(
    raw_tf: int, doc_len: int, avg_doc_len: float, k: float = K1, b: float = B
) -> float:
    return raw_tf * (k + 1) / (raw_tf + k * (1 - b + b * doc_len / avg_doc_len))


def merge_cqa_dupstack(data_path: str, verbose: bool = False):
    """Merge CQADupStack subdatasets into a single dataset"""
    data_path = Path(data_path)
    dataset = data_path.name
    assert dataset == "cqadupstack", "Dataset must be CQADupStack"

    corpus_path = data_path / "corpus.jsonl"
    if not corpus_path.exists():
        corpus_files = list(data_path.glob("*/corpus.jsonl"))
        with open(corpus_path, "w") as f:
            for file in tqdm(corpus_files, desc="Merging Corpus", disable=not verbose):
                corpus_name = file.parent.name
                with open(file, "r") as f2:
                    for line in tqdm(
                        f2, desc=f"Merging {corpus_name} Corpus", leave=False, disable=not verbose
                    ):
                        line = json.loads(line)
                        line["_id"] = f"{corpus_name}_{line['_id']}"
                        f.write(json.dumps(line))
                        f.write("\n")

    queries_path = data_path / "queries.jsonl"
    if not queries_path.exists():
        queries_files = list(data_path.glob("*/queries.jsonl"))
        with open(queries_path, "w") as f:
            for file in tqdm(queries_files, desc="Merging Queries", disable=not verbose):
                corpus_name = file.parent.name
                with open(file, "r") as f2:
                    for line in tqdm(
                        f2, desc=f"Merging {corpus_name} Queries", leave=False, disable=not verbose
                    ):
                        line = json.loads(line)
                        line["_id"] = f"{corpus_name}_{line['_id']}"
                        f.write(json.dumps(line))
                        f.write("\n")

    qrels_path = data_path / "qrels" / "test.tsv"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    if not qrels_path.exists():
        qrels_files = list(data_path.glob("*/qrels/test.tsv"))
        with open(qrels_path, "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for file in tqdm(qrels_files, desc="Merging Qrels", disable=not verbose):
                corpus_name = file.parent.parent.name
                with open(file, "r") as f2:
                    next(f2)
                    for line in tqdm(
                        f2, desc=f"Merging {corpus_name} Qrels", leave=False, disable=not verbose
                    ):
                        qid, cid, score = line.strip().split("\t")
                        qid = f"{corpus_name}_{qid}"
                        cid = f"{corpus_name}_{cid}"
                        f.write(f"{qid}\t{cid}\t{score}\n")


client = None

def _password() -> str:
    return DEFAULT_PASS or getpass.getpass("Enter cosdata admin password: ")

def hybrid(
    self,
    *,
    dense_query: List[float],
    sparse_query: str,
    top_k: int = 10,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    return_raw_text: bool = False,
) -> Dict[str, Any]:
    """
    Hybrid search that combines dense and sparse (TF-IDF) signals.
    Uses raw text without client-side preprocessing - server handles all processing.
    """
    url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/hybrid"

    payload = {
        "query_vector": dense_query,
        "query_text": sparse_query,  # Raw text for server-side TF-IDF processing
        "top_k": top_k,
        "fusion_constant_k": 60.0,
        "return_raw_text": return_raw_text,
    }

    rsp = requests.post(
        url,
        headers=self.collection.client._get_headers(),
        json=payload,
        verify=self.collection.client.verify_ssl,
    )
    if rsp.status_code != 200:
        raise Exception(f"Hybrid search failed: {rsp.text}")

    return rsp.json()


def batch_hybrid(
    self,
    *,
    dense_queries: List[List[float]],
    sparse_queries: List[str],
    top_k: int = 10,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    return_raw_text: bool = False,
) -> List[Dict[str, Any]]:
    """
    Batch variant of `hybrid`.
    Uses raw text without client-side preprocessing - server handles all processing.

    Parameters
    ----------
    dense_queries  : list of dense vectors
    sparse_queries : list of raw text strings (same order / length)
    ...            : remaining arguments identical to `hybrid`

    Returns
    -------
    List[Dict] – one dict per query in the same order provided.
    """
    if len(dense_queries) != len(sparse_queries):
        raise ValueError("dense_queries and sparse_queries must have the same length")

    url = f"{self.collection.client.base_url}/collections/{self.collection.name}/search/batch-hybrid"

    queries = []
    for dv, sv in zip(dense_queries, sparse_queries):
        queries.append(
            {
                "query_vector": dv,
                "query_text": sv,  # Raw text for server-side processing
            }
        )

    payload = {
        "queries": queries,
        "top_k": top_k,
        "fusion_constant_k": 60.0,
        "return_raw_text": return_raw_text,
    }

    rsp = requests.post(
        url,
        headers=self.collection.client._get_headers(),
        json=payload,
        verify=self.collection.client.verify_ssl,
    )
    if rsp.status_code != 200:
        raise Exception(f"Batch hybrid search failed: {rsp.text}")

    data = rsp.json()
    return data["responses"]

def create_session():
    """Initialize the cosdata client"""
    global client
    client = Client(
        host=DEFAULT_HOST, 
        username=DEFAULT_USER, 
        password=_password(), 
        verify=False
    )
    return client

def setup_embedding_model():
    """Setup the embedding model for dense vectors"""
    try:
        embedding_model = TextEmbedding(
            model_name="thenlper/gte-base",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            model_kwargs={"torch_dtype": "float16"},
            max_length=512,
        )
        test_texts = ["test"]
        _ = list(embedding_model.embed(test_texts))
        print("Using GPU-accelerated embeddings")
        return embedding_model
    except Exception as e:
        print(f"GPU not available, falling back to CPU: {e}")
        return TextEmbedding(model_name="thenlper/gte-base", max_length=512)

def get_beir_dataset(dataset: str) -> Tuple[Dict, Dict, Dict]:
    """Download and load BEIR dataset"""
    save_dir = Path("datasets") / f"hybrid_raw_{dataset}"
    archive_dir = save_dir / "archive"
    save_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset and unzip the dataset
    base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
    data_path = beir.util.download_and_unzip(base_url.format(dataset), str(archive_dir))

    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path, verbose=True)

    loader = GenericDataLoader(data_folder=data_path)
    loader.check(fIn=loader.corpus_file, ext="jsonl")
    loader.check(fIn=loader.query_file, ext="jsonl")
    loader._load_corpus()
    loader._load_queries()
    corpus = loader.corpus
    queries = loader.queries
    
    loader.qrels_file = os.path.join(loader.qrels_folder, split + ".tsv")
    loader._load_qrels()
    qrels = loader.qrels
    
    return corpus, queries, qrels

def generate_dense_embeddings(corpus: Dict, queries: Dict, dataset: str, embedding_model) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Generate dense embeddings for corpus and queries"""
    cache_file = Path("datasets") / f"hybrid_{dataset}" / "dense_embeddings.npz"
    if cache_file.exists():
        print("Loading cached dense embeddings...")
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["corpus_ids"].tolist(),
            data["corpus_embeddings"],
            data["query_ids"].tolist(),
            data["query_embeddings"],
        )

    print("Generating dense embeddings...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        f"{corpus[doc_id].get('title', '')} {corpus[doc_id]['text']}"
        for doc_id in corpus_ids
    ]

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    corpus_embeddings = []
    for i in tqdm(
        range(0, len(corpus_texts), EMBEDDING_BATCH_SIZE), desc="Corpus embeddings"
    ):
        batch = corpus_texts[i : i + EMBEDDING_BATCH_SIZE]
        corpus_embeddings.extend(list(embedding_model.embed(batch)))
    corpus_embeddings = np.array(corpus_embeddings)

    query_embeddings = []
    for i in tqdm(range(0, len(query_texts), EMBEDDING_BATCH_SIZE), desc="Query embeddings"):
        batch = query_texts[i : i + EMBEDDING_BATCH_SIZE]
        query_embeddings.extend(list(embedding_model.embed(batch)))
    query_embeddings = np.array(query_embeddings)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        corpus_ids=corpus_ids,
        corpus_embeddings=corpus_embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
    )
    return corpus_ids, corpus_embeddings, query_ids, query_embeddings

def brute_force_dense(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    query_ids: List[str],
    top_k: int,
    dataset: str,
) -> List[Dict]:
    bf_file = Path("datasets") / f"hybrid_{dataset}" / "dense_brute_force.pkl"
    if bf_file.exists():
        return pickle.loads(bf_file.read_bytes())

    print("Computing dense brute-force...")
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    query_embeddings_norm = query_embeddings / (query_norms + 1e-8)
    corpus_embeddings_norm = corpus_embeddings / (corpus_norms + 1e-8)

    results = []
    for i, query_id in enumerate(tqdm(query_ids, desc="Dense brute-force")):
        scores = np.dot(corpus_embeddings_norm, query_embeddings_norm[i])
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_results = [
            {"id": corpus_ids[idx], "score": float(scores[idx])} for idx in top_indices
        ]
        results.append({"query_id": query_id, "top_results": top_results})

    bf_file.parent.mkdir(parents=True, exist_ok=True)
    bf_file.write_bytes(pickle.dumps(results))
    return results


def build_sparse_vectors_for_bf(corpus: Dict, dataset: str) -> List[Dict]:
    """Build sparse vectors for brute-force comparison"""
    cache_file = Path("datasets") / f"hybrid_{dataset}" / "sparse_vectors.pkl"
    if cache_file.exists():
        print("Loading cached sparse vectors...")
        return pickle.loads(cache_file.read_bytes())

    print("Tokenising corpus (sparse)...")
    stemmer = SnowballStemmer("english")

    def process_doc(doc_item):
        doc_id, doc = doc_item
        text = f"{doc.get('title', '')} {doc['text']}"
        tokens = SimpleTokenizer.tokenize(text)
        terms = [
            stemmer.stem_word(t.lower())
            for t in tokens
            if t.lower() not in STOPWORDS and t not in PUNCT and len(t) <= 40
        ]
        return {"id": doc_id, "tokens": terms, "length": len(terms)}

    docs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_doc, item) for item in corpus.items()]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Tokenising"):
            docs.append(f.result())

    total_docs = len(docs)
    term_doc_freq = defaultdict(int)
    total_len = sum(d["length"] for d in docs)
    avg_len = total_len / total_docs
    for d in docs:
        for t in set(d["tokens"]):
            term_doc_freq[t] += 1

    def build_vector(doc):
        tf = defaultdict(int)
        for tok in doc["tokens"]:
            tf[tok] += 1
        indices, values = [], []
        for tok, raw in tf.items():
            idf = compute_bm25_idf(total_docs, term_doc_freq[tok])
            tf_score = compute_bm25_tf(raw, doc["length"], avg_len)
            bm25_score = idf * tf_score
            if bm25_score > 0:
                indices.append(hash(tok) % (2**31))
                values.append(bm25_score)
        return {
            "id": doc["id"],
            "text": " ".join(doc["tokens"]),
            "indices": indices,
            "values": values,
        }

    vectors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(build_vector, d) for d in docs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Building sparse"):
            vectors.append(f.result())

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(vectors))
    return vectors


def build_sparse_queries_for_bf(queries: Dict, dataset: str, corpus_stats) -> List[Dict]:
    """Build sparse queries for brute-force comparison"""
    cache_file = Path("datasets") / f"hybrid_{dataset}" / "sparse_queries.pkl"
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())

    total_docs, term_doc_freq, avg_len = corpus_stats
    stemmer = SnowballStemmer("english")
    query_vecs = []
    for qid, qtext in tqdm(queries.items(), desc="Queries"):
        tokens = SimpleTokenizer.tokenize(qtext)
        terms = [
            stemmer.stem_word(t.lower())
            for t in tokens
            if t.lower() not in STOPWORDS and t not in PUNCT and len(t) <= 40
        ]
        tf = defaultdict(int)
        for tok in terms:
            tf[tok] += 1
        indices, values = [], []
        for tok, raw in tf.items():
            if tok in term_doc_freq:
                idf = compute_bm25_idf(total_docs, term_doc_freq[tok])
                tf_score = compute_bm25_tf(raw, len(terms), avg_len)
                bm25_score = idf * tf_score
                if bm25_score > 0:
                    indices.append(hash(tok) % (2**31))
                    values.append(bm25_score)
        query_vecs.append(
            {"id": qid, "text": " ".join(terms), "indices": indices, "values": values}
        )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(query_vecs))
    return query_vecs


def brute_force_sparse(
    queries: List[Dict], docs: List[Dict], top_k: int, dataset: str
) -> List[Dict]:
    bf_file = Path("datasets") / f"hybrid_{dataset}" / "sparse_brute_force.pkl"
    if bf_file.exists():
        return pickle.loads(bf_file.read_bytes())

    print("Computing sparse brute-force...")
    results = []
    for q in tqdm(queries, desc="Sparse brute-force"):
        scores = defaultdict(float)
        q_indices_set = set(q["indices"])
        for d in docs:
            # Compute dot product between sparse vectors
            for i, val in zip(d["indices"], d["values"]):
                if i in q_indices_set:
                    q_idx = q["indices"].index(i)
                    scores[d["id"]] += val * q["values"][q_idx]
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results.append(
            {
                "query_id": q["id"],
                "top_results": [{"id": i, "score": s} for i, s in top],
            }
        )

    bf_file.parent.mkdir(parents=True, exist_ok=True)
    bf_file.write_bytes(pickle.dumps(results))
    return results

def ensure_collection(name: str):
    """Create or recreate the collection with proper configuration"""
    try:
        print(f"Creating collection {name}...")
        collection = client.create_collection(
            name=name,
            dimension=DIMENSION,
            tf_idf_options={"enabled": True},
        )
        print("Collection created successfully")

        print("Creating dense HNSW index...")
        collection.create_index(distance_metric="cosine")

        print("Creating TF-IDF index...")
        collection.create_tf_idf_index(name=name, k1=K1, b=B)

        # Add hybrid search methods to collection via monkey patching
        collection.search.hybrid = hybrid.__get__(collection.search)
        collection.search.batch_hybrid = batch_hybrid.__get__(collection.search)

        return collection

    except Exception as e:
        print(
            f"Collection {name} may already exist, attempting to delete and recreate: {e}"
        )
        try:
            existing = client.get_collection(name)
            existing.delete()
            time.sleep(2)
        except Exception as delete_error:
            print(f"Error deleting collection: {delete_error}")

        collection = client.create_collection(
            name=name, dimension=DIMENSION, tf_idf_options={"enabled": True}
        )
        collection.create_index(distance_metric="cosine")
        collection.create_tf_idf_index(name=name, k1=K1, b=B)
        
        # Add hybrid search methods to collection via monkey patching
        collection.search.hybrid = hybrid.__get__(collection.search)
        collection.search.batch_hybrid = batch_hybrid.__get__(collection.search)
        
        return collection

def upsert_hybrid_vectors(collection, corpus_ids, corpus_embeddings, corpus):
    """Upsert both dense embeddings and raw text for hybrid search"""
    print("Upserting hybrid vectors (dense + raw text)...")
    
    txn_ = None
    with collection.transaction() as txn:
        for start in tqdm(
            range(0, len(corpus_ids), BATCH_UPSERT_SIZE), desc="Hybrid upsert"
        ):
            end = min(start + BATCH_UPSERT_SIZE, len(corpus_ids))
            batch = []
            for i in range(start, end):
                doc_id = corpus_ids[i]
                doc = corpus[doc_id]
                
                raw_text = f"{doc.get('title', '')} {doc['text']}".strip()
                
                batch.append(
                    {
                        "id": doc_id,
                        "dense_values": corpus_embeddings[i].tolist(),
                        "text": raw_text,
                        "metadata": {
                            "text": doc["text"],
                            "title": doc.get("title", ""),
                        },
                    }
                )

            txn.batch_upsert_vectors(batch)
        
        txn_ = txn

    if txn_:
        print(f"Transaction id for hybrid upsert: {txn_.transaction_id}")
        print(f"Polling transaction {txn_.transaction_id}...")
        final_status, success = txn_.poll_completion(
            target_status="complete", max_attempts=10, sleep_interval=30
        )

        if not success:
            print(
                f"Transaction {txn_.transaction_id} did not complete successfully. Final status: {final_status}"
            )
        else:
            print(f"Transaction {txn_.transaction_id} completed successfully")

    print("Hybrid upsert and indexing completed successfully")

def hybrid_search_comprehensive(
    collection,
    dense_query_vectors: np.ndarray,
    text_queries: List[str],
    query_ids: List[str],
    top_k: int,
) -> Tuple[List[List[Dict]], Dict, Dict]:
    """Perform comprehensive hybrid search with CPU monitoring and QPS tracking"""
    cpu_monitor = CPUMonitor()
    qps_tracker = QPSTracker()

    def _batch(batch_indices, batch_index):
        batch_start_time = time.time()
        failed_count = 0

        try:
            batch_dense = [dense_query_vectors[i].tolist() for i in batch_indices]
            batch_text = [text_queries[i] for i in batch_indices]
            
            batch_results = collection.search.batch_hybrid(
                dense_queries=batch_dense,
                sparse_queries=batch_text,
                top_k=top_k,
                dense_weight=0.5,
                sparse_weight=0.5,
            )
            
            formatted_results = []
            for result in batch_results:
                if isinstance(result, dict) and "results" in result:
                    formatted = [
                        {"id": r["id"], "score": r["score"]} 
                        for r in result["results"]
                    ]
                else:
                    formatted = [{"id": r["id"], "score": r["score"]} for r in result]
                formatted_results.append(formatted)
            
        except Exception as e:
            print(f"Batch hybrid search failed: {e}")
            failed_count = len(batch_indices)
            formatted_results = [[]] * len(batch_indices)
        
        batch_duration = time.time() - batch_start_time
        qps_tracker.record_query_batch(len(batch_indices), batch_duration, failed_count)
        return formatted_results

    batches = []
    for idx, start in enumerate(range(0, len(query_ids), SEARCH_BATCH_SIZE)):
        end = min(start + SEARCH_BATCH_SIZE, len(query_ids))
        batches.append((list(range(start, end)), idx))

    cpu_monitor.start_monitoring()
    qps_tracker.start_tracking()

    all_results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_batch, *b): b[1] for b in batches}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Hybrid search"):
            batch_idx = futures[f]
            all_results[batch_idx] = f.result()

    cpu_stats = cpu_monitor.stop_monitoring()
    qps_metrics = qps_tracker.stop_tracking()

    final_results = []
    for batch_result in all_results:
        if batch_result is not None:
            final_results.extend(batch_result)

    return final_results, cpu_stats, qps_metrics

def search_documents_hybrid(
    collection,
    queries: List[Tuple[str, str]], 
    dense_query_vectors: np.ndarray,
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    """Search using hybrid approach with raw text"""
    results = {}
    
    text_queries = [query[1] for query in queries]
    query_ids = [query[0] for query in queries]
    
    search_results, _, _ = hybrid_search_comprehensive(
        collection, dense_query_vectors, text_queries, query_ids, top_k
    )
    
    for qid, search_result in zip(query_ids, search_results):
        results[qid] = {doc["id"]: doc["score"] for doc in search_result}
    
    return results

def evaluate(
    hybrid_results: List[List[Dict]],
    bf_dense: List[Dict],
    bf_sparse: List[Dict],
    query_ids: List[str],
    qrels: Dict,
) -> Dict:
    hybrid_dict = {
        qid: {r["id"]: r["score"] for r in res}
        for qid, res in zip(query_ids, hybrid_results)
    }
    bf_dense_dict = {
        r["query_id"]: {rr["id"]: rr["score"] for rr in r["top_results"]}
        for r in bf_dense
    }
    recalls_dense = []
    for qid, bf_ids in bf_dense_dict.items():
        srv_ids = set(hybrid_dict.get(qid, {}))
        recalls_dense.append(len(srv_ids & set(bf_ids)) / len(bf_ids) if bf_ids else 0)
    bf_recall_dense = sum(recalls_dense) / len(recalls_dense) if recalls_dense else 0

    bf_sparse_dict = {
        r["query_id"]: {rr["id"]: rr["score"] for rr in r["top_results"]}
        for r in bf_sparse
    }
    recalls_sparse = []
    for qid, bf_ids in bf_sparse_dict.items():
        srv_ids = set(hybrid_dict.get(qid, {}))
        recalls_sparse.append(len(srv_ids & set(bf_ids)) / len(bf_ids) if bf_ids else 0)

    bf_recall_sparse = (
        sum(recalls_sparse) / len(recalls_sparse) if recalls_sparse else 0
    )

    try:
        evaluator = EvaluateRetrieval(k_values=[1, 5, 10])
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, hybrid_dict, [1, 5, 10])
        beir_metrics = {
            "ndcg@10": ndcg["NDCG@10"],
            "beir_recall@10": recall["Recall@10"],
            "map": _map["MAP@10"],
            "p@1": precision["P@1"],
            "p@5": precision["P@5"],
            "p@10": precision["P@10"],
        }
    except Exception as e:
        print(f"BEIR evaluation failed: {e}")
        beir_metrics = {
            "ndcg@10": 0.0, "beir_recall@10": 0.0, "map": 0.0,
            "p@1": 0.0, "p@5": 0.0, "p@10": 0.0,
        }

    return {
        "bf_recall_dense@10": bf_recall_dense,
        "bf_recall_sparse@10": bf_recall_sparse,
        **beir_metrics,
    }

def run_qps_latency(
    dense_query_vectors: np.ndarray,
    text_queries: List[str],
    collection,
    top_k: int,
):
    subset_size = min(1000, len(dense_query_vectors))
    dense_subset = dense_query_vectors[:subset_size]
    text_subset = text_queries[:subset_size]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        start = time.time()
        futures = [
            ex.submit(
                collection.search.hybrid,
                dense_query=dense_subset[i].tolist(),
                sparse_query=text_subset[i],
                top_k=top_k,
                dense_weight=0.5,
                sparse_weight=0.5,
            )
            for i in range(subset_size)
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Sequential QPS query failed: {e}")
    elapsed = time.time() - start
    qps = subset_size / elapsed

    times = []
    latency_subset_size = min(100, len(dense_query_vectors))
    for i in tqdm(range(latency_subset_size), desc="Latency"):
        start = time.time()
        try:
            collection.search.hybrid(
                dense_query=dense_query_vectors[i].tolist(),
                sparse_query=text_queries[i],
                top_k=top_k,
                dense_weight=0.5,
                sparse_weight=0.5,
            )
            times.append((time.time() - start) * 1000)
        except Exception as e:
            print(f"Latency query failed: {e}")
    times.sort()
    if not times:
        return qps, 0.0, 0.0
    return qps, times[len(times) // 2], times[int(len(times) * 0.95)]

def save_comprehensive_results(
    dataset: str,
    corpus_size: int,
    queries_size: int,
    qrels_size: int,
    avg_query_len: float,
    eval_metrics: Dict,
    qps_metrics: Dict,
    cpu_stats: Dict,
    latency_metrics: Tuple[float, float, float],
    search_duration: float,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset}_hybrid_raw_{timestamp}.csv"

    qps, p50, p95 = latency_metrics
    success_rate = (
        (qps_metrics["successful_queries"] / qps_metrics["total_queries"]) * 100
        if qps_metrics["total_queries"] > 0 else 0
    )

    results = {
        "dataset": dataset, "corpus_size": corpus_size, "queries_size": queries_size,
        "qrels_size": qrels_size, "avg_query_len": avg_query_len,
        "bf_recall_dense@10": eval_metrics["bf_recall_dense@10"],
        "bf_recall_sparse@10": eval_metrics["bf_recall_sparse@10"],
        "ndcg@10": eval_metrics["ndcg@10"], "beir_recall@10": eval_metrics["beir_recall@10"],
        "map": eval_metrics["map"], "p@1": eval_metrics["p@1"], "p@5": eval_metrics["p@5"],
        "p@10": eval_metrics["p@10"], "concurrent_qps": qps_metrics["qps"],
        "sequential_qps": qps, "avg_query_time_ms": qps_metrics["avg_query_time"],
        "p50_latency_ms": p50, "p95_latency_ms": p95, "total_queries": qps_metrics["total_queries"],
        "successful_queries": qps_metrics["successful_queries"], "failed_queries": qps_metrics["failed_queries"],
        "success_rate_pct": success_rate, "search_duration_s": search_duration,
        "total_test_duration_s": qps_metrics["total_duration"],
        "max_cpu_usage_pct": cpu_stats["max_cpu"], "avg_cpu_usage_pct": cpu_stats["avg_cpu"],
        "cpu_samples": cpu_stats["samples"],
    }

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)
    print(f"Comprehensive results saved to: {filename}")
    return filename

def main(
    dataset: str = "arguana",
    top_k: int = 10,
):
    """Main function to run the hybrid raw search benchmark"""
    print("=" * 60)
    print("Raw Hybrid Search Benchmark (dense + server-side sparse)")
    print("=" * 60)
    
    try:
        embedding_model = setup_embedding_model()
        create_session()
        
        corpus, queries, qrels = get_beir_dataset(dataset)
        print(f"Loaded {dataset}: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
        
        avg_query_len = sum(len(q.split()) for q in queries.values()) / len(queries)
        print(f"Average query length: {avg_query_len:.1f} tokens")
        
        corpus_ids, corpus_embeddings, query_ids, query_embeddings = generate_dense_embeddings(
            corpus, queries, dataset, embedding_model
        )
        
        print("\nBuilding brute-force baselines...")
        bf_dense = brute_force_dense(
            query_embeddings, corpus_embeddings, corpus_ids, query_ids, top_k, dataset
        )
        
        sparse_vectors = build_sparse_vectors_for_bf(corpus, dataset)
        
        total_docs = len(sparse_vectors)
        term_doc_freq = defaultdict(int)
        total_len = sum(len(v["text"].split()) for v in sparse_vectors)
        avg_len = total_len / total_docs
        for v in sparse_vectors:
            for idx in v["indices"]:
                term_doc_freq[idx] += 1
        
        sparse_queries = build_sparse_queries_for_bf(
            queries, dataset, (total_docs, term_doc_freq, avg_len)
        )
        bf_sparse = brute_force_sparse(sparse_queries, sparse_vectors, top_k, dataset)
        
        collection_name = f"{COLLECTION_PREFIX}_{dataset}"
        collection = ensure_collection(collection_name)
        
        start_time = time.time()
        upsert_hybrid_vectors(collection, corpus_ids, corpus_embeddings, corpus)
        indexing_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("HYBRID SEARCH WITH RAW TEXT")
        print("=" * 60)
        
        eval_queries = [(qid, queries[qid]) for qid in qrels.keys() if qid in query_ids]
        eval_query_ids = [q[0] for q in eval_queries]
        eval_dense_vectors = np.array([
            query_embeddings[query_ids.index(qid)] for qid in eval_query_ids
        ])
        eval_text_queries = [q[1] for q in eval_queries]
        
        print("Running comprehensive hybrid search...")
        search_start = time.time()
        hybrid_results, cpu_stats, qps_metrics = hybrid_search_comprehensive(
            collection, eval_dense_vectors, eval_text_queries, eval_query_ids, top_k
        )
        search_duration = time.time() - search_start
        
        print("Evaluating results...")
        eval_metrics = evaluate(
            hybrid_results,
            [r for r in bf_dense if r["query_id"] in eval_query_ids],
            [r for r in bf_sparse if r["query_id"] in eval_query_ids],
            eval_query_ids,
            qrels,
        )
        
        print("Running sequential QPS and latency tests...")
        seq_qps, p50, p95 = run_qps_latency(
            eval_dense_vectors, eval_text_queries, collection, top_k
        )
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE RESULTS")
        print("=" * 60)
        print(f"Dataset: {dataset}")
        print(f"Corpus Size: {len(corpus):,}")
        print(f"Queries Size: {len(queries):,}")
        print(f"Qrels Size: {len(qrels):,}")
        print(f"Avg Query Length: {avg_query_len:.1f} tokens")
        print()
        print("Brute-force Recall:")
        print(f"  vs Dense BF@10: {eval_metrics['bf_recall_dense@10']:.4f}")
        print(f"  vs Sparse BF@10: {eval_metrics['bf_recall_sparse@10']:.4f}")
        print()
        print("BEIR Metrics:")
        print(f"  NDCG@10: {eval_metrics['ndcg@10']:.4f}")
        print(f"  Recall@10: {eval_metrics['beir_recall@10']:.4f}")
        print(f"  MAP@10: {eval_metrics['map']:.4f}")
        print(f"  P@1: {eval_metrics['p@1']:.4f}")
        print(f"  P@5: {eval_metrics['p@5']:.4f}")
        print(f"  P@10: {eval_metrics['p@10']:.4f}")
        print()
        print("Performance:")
        print(f"  Concurrent QPS: {qps_metrics['qps']:.2f}")
        print(f"  Sequential QPS: {seq_qps:.2f}")
        print(f"  Avg Query Time: {qps_metrics['avg_query_time']:.2f}ms")
        print(f"  P50 Latency: {p50:.2f}ms")
        print(f"  P95 Latency: {p95:.2f}ms")
        print(f"  Success Rate: {(qps_metrics['successful_queries'] / qps_metrics['total_queries']) * 100:.1f}%")
        print()
        print("System:")
        print(f"  Max CPU: {cpu_stats['max_cpu']:.1f}%")
        print(f"  Avg CPU: {cpu_stats['avg_cpu']:.1f}%")
        print(f"  Search Duration: {search_duration:.2f}s")
        print("=" * 60)
        
        save_comprehensive_results(
            dataset, len(corpus), len(queries), len(qrels), avg_query_len,
            eval_metrics, qps_metrics, cpu_stats, (seq_qps, p50, p95), search_duration,
        )
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if "collection" in locals() and collection:
                print(f"\nCleaning up: deleting collection {collection.name}")
                collection.delete()
                print("Collection deleted successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw Hybrid Search Benchmark")
    parser.add_argument(
        "--dataset", type=str, default="arguana",
        help="BEIR dataset to benchmark (default: arguana)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top results to retrieve (default: 10)",
    )
    args = parser.parse_args()
    
    main(args.dataset, args.top_k)