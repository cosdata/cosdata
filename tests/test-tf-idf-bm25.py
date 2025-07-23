import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import getpass
import pickle
from tqdm import tqdm
import heapq
import math
import bm25s
import re
import sys
import unicodedata
import xxhash
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from py_rust_stemmers import SnowballStemmer
from dotenv import load_dotenv
from cosdata import Client

# Load environment variables from .env file
load_dotenv()

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
client = None
host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")


def create_session():
    """Initialize the cosdata client"""
    # Use environment variable from .env file if available, otherwise prompt
    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter admin password: ")

    username = os.getenv("COSDATA_USERNAME", "admin")

    global client
    client = Client(host=host, username=username, password=password, verify=False)
    return client


def create_db(name, description=None, dimension=20000):
    """Create collection using cosdata client"""
    collection = client.create_collection(
        name=name,
        dimension=dimension,
        description=description,
        tf_idf_options={"enabled": True},
    )
    return collection


def create_explicit_index(collection, k, b):
    """Create TF-IDF index using cosdata client"""
    # For TF-IDF, we create a specific index configuration
    index = collection.create_tf_idf_index(
        name=collection.name, sample_threshold=1000, k1=k, b=b
    )
    return index


def compute_bm25_idf(total_documents, docs_containing_term):
    """
    Compute IDF using the BM25 formula:
    ln(((total documents - documents contain the term + 0.5) / (documents containing the term + 0.5)) + 1)
    """
    return math.log1p(
        (total_documents - docs_containing_term + 0.5) / (docs_containing_term + 0.5)
    )


def compute_bm25_term_frequency(
    count, document_length, avg_document_length, k=1.5, b=0.75
):
    """
    Normalize term frequency for BM25 scoring:
    tf * (k+1) / (tf + k * (1 - b + b * (dl/avgdl)))
    """
    normalized_freq = (
        count
        * (k + 1)
        / (count + k * (1 - b + b * (document_length / avg_document_length)))
    )
    return normalized_freq


def get_all_punctuation() -> Set[str]:
    return set(
        chr(i)
        for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)).startswith("P")
    )


def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)


class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w]", " ", text.lower())
        text = re.sub(r"\s+", " ", text)
        return text.strip().split()


def raw_term_frequencies(tokens: List[str]) -> Dict[str, int]:
    counter: defaultdict[str, int] = defaultdict(int)
    for token in tokens:
        counter[token] += 1
    return counter


def hash_token(token: str) -> int:
    return xxhash.xxh32(token.encode("utf-8")).intdigest() & 0xFFFFFFFF


def construct_sparse_vector(tokens: List[str]) -> Tuple[List[Tuple[int, int]], int]:
    tf_dict = raw_term_frequencies(tokens)
    sparse_vector = [(hash_token(token), value) for token, value in tf_dict.items()]
    return sparse_vector, len(tokens)


def transform_sentence_to_vector(id, sentence, punctuations, stemmer, token_max_len=40):
    stopwords = {
        "a",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "s",
        "such",
        "t",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
        "www",
    }
    cleaned = remove_non_alphanumeric(sentence)
    tokens = SimpleTokenizer.tokenize(cleaned)
    processed_tokens = [
        stemmer.stem_word(token.lower())
        for token in tokens
        if token not in punctuations
        and token.lower() not in stopwords
        and len(token) <= token_max_len
    ]
    terms, length = construct_sparse_vector(processed_tokens)

    return {
        "id": id,
        "text": sentence,
        "indices": [term[0] for term in terms],
        "raw_term_frequencies": [term[1] for term in terms],
        "length": length,
    }


def get_dataset(dataset: str) -> tuple:
    """Generate raw dataset with just term frequencies"""
    dataset_folder = f"datasets/sparse_idf_dataset_{dataset}"
    raw_dataset_file = f"{dataset_folder}/raw_vectors.pkl"
    if os.path.exists(raw_dataset_file):
        print(f"Loading existing raw dataset from {raw_dataset_file}")
        with open(raw_dataset_file, "rb") as f:
            return pickle.load(f)

    save_dir = f"{dataset_folder}/dataset"

    print("Downloading the dataset...")
    bm25s.utils.beir.download_dataset(dataset, save_dir=save_dir)
    print("Loading the corpus...")
    corpus = bm25s.utils.beir.load_corpus(dataset, save_dir=save_dir)

    punctuations = get_all_punctuation()
    stemmer = SnowballStemmer("english")

    print("Converting to sparse vectors...")
    vectors = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for k, v in tqdm(corpus.items()):
            futures.append(
                executor.submit(
                    transform_sentence_to_vector,
                    k,
                    v["title"] + " " + v["text"],
                    punctuations,
                    stemmer,
                )
            )

        for future in tqdm(as_completed(futures)):
            vectors.append(future.result())

    # Save to disk
    with open(raw_dataset_file, "wb") as f:
        pickle.dump(vectors, f)

    print(f"Raw dataset generated and saved to {raw_dataset_file}")
    return vectors


def calculate_corpus_statistics(dataset, vectors):
    """Calculate corpus-wide statistics for BM25"""
    dataset_folder = f"datasets/sparse_idf_dataset_{dataset}"
    stats_file = f"{dataset_folder}/corpus_stats.pkl"
    if os.path.exists(stats_file):
        print(f"Loading existing corpus statistics from {stats_file}")
        with open(stats_file, "rb") as f:
            return pickle.load(f)

    print("Calculating corpus statistics...")

    # Calculate average document length
    total_length = sum(vec["length"] for vec in vectors)
    avg_document_length = total_length / len(vectors)

    # Count documents containing each term
    term_document_counts = {}
    for vec in tqdm(vectors):
        for term_index in vec["indices"]:
            term_document_counts[term_index] = (
                term_document_counts.get(term_index, 0) + 1
            )

    # Calculate IDF for each term
    total_documents = len(vectors)
    term_idf_values = {}
    for term, count in term_document_counts.items():
        term_idf_values[term] = compute_bm25_idf(total_documents, count)

    # Save statistics
    statistics = {
        "avg_document_length": avg_document_length,
        "term_document_counts": term_document_counts,
        "term_idf_values": term_idf_values,
        "total_documents": total_documents,
    }

    with open(stats_file, "wb") as f:
        pickle.dump(statistics, f)

    print(f"Corpus statistics calculated and saved to {stats_file}")
    return statistics


def apply_bm25_to_vectors(dataset, vectors, corpus_stats, k=1.5, b=0.75):
    """Apply BM25 scoring to all vectors using corpus statistics"""
    dataset_folder = f"datasets/sparse_idf_dataset_{dataset}"
    processed_dataset_file = f"{dataset_folder}/processed_vectors.pkl"
    if os.path.exists(processed_dataset_file):
        print(f"Loading existing processed vectors from {processed_dataset_file}")
        with open(processed_dataset_file, "rb") as f:
            return pickle.load(f)

    print("Applying BM25 scoring to vectors...")

    avg_document_length = corpus_stats["avg_document_length"]
    term_idf_values = corpus_stats["term_idf_values"]

    processed_vectors = []

    for vec in tqdm(vectors):
        # Calculate normalized term frequencies for BM25
        server_values = []
        brute_force_values = []

        for idx, (term_index, raw_freq) in enumerate(
            zip(vec["indices"], vec["raw_term_frequencies"])
        ):
            # Normalize term frequency
            normalized_freq = compute_bm25_term_frequency(
                raw_freq, vec["length"], avg_document_length, k, b
            )

            # Get IDF for this term
            idf = term_idf_values.get(term_index, 0)

            # Store normalized frequency for server
            server_values.append(normalized_freq)

            # Store TF-IDF for local brute force calculation
            brute_force_values.append(normalized_freq * idf)

        processed_vectors.append(
            {
                "id": vec["id"],
                "indices": vec["indices"],
                "text": vec["text"],
                "values": server_values,  # Normalized term frequency for server
                "brute_force_values": brute_force_values,  # TF-IDF for local calculation
                "length": vec["length"],
                "raw_term_frequencies": vec[
                    "raw_term_frequencies"
                ],  # Keep original frequencies
            }
        )

    # Save processed vectors
    with open(processed_dataset_file, "wb") as f:
        pickle.dump(processed_vectors, f)

    print(f"Processed vectors saved to {processed_dataset_file}")
    return processed_vectors


def get_query_vectors(dataset):
    dataset_folder = f"datasets/sparse_idf_dataset_{dataset}"
    query_vectors_file = f"{dataset_folder}/query.pkl"
    if os.path.exists(query_vectors_file):
        print(f"Loading existing query vectors from {query_vectors_file}")
        with open(query_vectors_file, "rb") as f:
            return pickle.load(f)
    save_dir = f"{dataset_folder}/dataset"
    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"
    queries = bm25s.utils.beir.load_queries(dataset, save_dir=save_dir)
    qrels = bm25s.utils.beir.load_qrels(dataset, split=split, save_dir=save_dir)
    queries_lst = [(k, v["text"]) for (k, v) in queries.items() if k in qrels]
    print(f"Processing {len(queries_lst)} queries...")
    queries = []

    punctuations = get_all_punctuation()
    stemmer = SnowballStemmer("english")

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for k, v in tqdm(queries_lst):
            futures.append(
                executor.submit(
                    transform_sentence_to_vector, k, v, punctuations, stemmer
                )
            )

        for future in tqdm(as_completed(futures)):
            queries.append(future.result())

    with open(query_vectors_file, "wb") as f:
        pickle.dump(queries, f)

    print(f"Query vectors saved to {query_vectors_file}")
    return queries


def compute_brute_force_results(dataset, vectors, query_vectors, top_k=10):
    """Compute dot product similarity between query vectors and all vectors"""
    dataset_folder = f"datasets/sparse_idf_dataset_{dataset}"
    brute_force_results_file = f"{dataset_folder}/brute_force_results.pkl"
    if os.path.exists(brute_force_results_file):
        print(f"Loading existing brute force results from {brute_force_results_file}")
        with open(brute_force_results_file, "rb") as f:
            return pickle.load(f)

    print(
        f"Computing brute force dot product similarity for {len(query_vectors)} queries..."
    )
    results = []

    for query in tqdm(query_vectors):
        dot_products = []

        for vec in vectors:
            # Find common indices
            common_indices = list(set(query["indices"]) & set(vec["indices"]))

            # Compute dot product only for common indices using brute force values
            dot_product = sum(
                vec["brute_force_values"][vec["indices"].index(idx)]
                for idx in common_indices
            )

            dot_products.append(dot_product)

        # Get top-k results
        top_indices = heapq.nlargest(
            top_k, range(len(dot_products)), key=lambda i: dot_products[i]
        )

        top_results = [
            {"id": vectors[idx]["id"], "score": float(dot_products[idx])}
            for idx in top_indices
        ][:top_k]

        results.append({"query_id": query["id"], "top_results": top_results})

    # Save to disk
    with open(brute_force_results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Brute force results computed and saved to {brute_force_results_file}")
    return results


def create_transaction(collection):
    """Create transaction using cosdata client"""
    return collection.transaction()


def upsert_in_transaction(collection, transaction, vectors):
    """
    Upsert vectors to the collection using transaction
    """
    # Create a new list of vectors with only needed fields
    filtered_vectors = [{"id": vec["id"], "text": vec["text"]} for vec in vectors]

    transaction.batch_upsert_vectors(filtered_vectors)


def commit_transaction(transaction):
    """Commit transaction using cosdata client"""
    transaction.commit()
    return transaction.transaction_id


def delete_collection(collection_name):
    """Delete a collection using cosdata client"""
    try:
        collection = client.get_collection(collection_name)
        collection.delete()
        print(f"Collection {collection_name} deleted successfully")
    except Exception as e:
        print(f"Warning: Failed to delete collection {collection_name}: {e}")


def create_db_and_upsert_vectors(vector_db_name, vectors, batch_size, k, b):
    # Create collection
    collection = None
    try:
        print(f"Creating collection: {vector_db_name}")
        collection = create_db(
            name=vector_db_name, dimension=1
        )  # dummy, not sure if we need it
        print("Collection created")

        # Create explicit index
        create_explicit_index(collection, k, b)
        print("Explicit index created")
    except Exception as e:
        print(f"Collection creation failed: {e}")
        # Try to get existing collection
        try:
            collection = client.get_collection(vector_db_name)
            print("Using existing collection")
        except Exception as get_error:
            print(f"Failed to get existing collection: {get_error}")
            raise Exception(f"Cannot create or access collection: {e}")

    # Insert vectors into server
    print("Creating transaction")
    txn_id = None

    print(f"Inserting {len(vectors)} vectors in batches of {batch_size}...")
    start = time.time()

    with collection.transaction() as txn:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for batch_start in range(0, len(vectors), batch_size):
                batch = vectors[batch_start : batch_start + batch_size]
                futures.append(
                    executor.submit(upsert_in_transaction, collection, txn, batch)
                )

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                try:
                    future.result()
                except Exception as e:
                    print(f"Batch {i + 1} failed: {e}")

        txn_id = txn.transaction_id

    # Poll for transaction completion
    print("Waiting for transaction to complete")
    final_status, success = txn_id.poll_completion(
        target_status="complete",
        max_attempts=10,
        sleep_interval=2,
    )

    if not success:
        print(
            f"Warning: Transaction may not have completed successfully. Final status: {final_status}"
        )

    end = time.time()
    insertion_time = end - start
    print(f"Insertion time: {insertion_time:.2f} seconds")
    return insertion_time, collection


def search_sparse_vector(collection, vector, top_k):
    """Search using TF-IDF with cosdata client"""
    results = collection.search.text(query_text=vector["text"], top_k=top_k)
    return results


def batch_ann_search(collection, batch, top_k):
    """Batch TF-IDF search using individual requests since batch endpoint is failing"""
    start_time = time.time()

    results = []
    for query_text in batch:
        try:
            result = collection.search.text(query_text=query_text, top_k=top_k)
            results.append(result)
        except Exception as e:
            print(f"Individual search failed for query: {e}")
            results.append({"results": []})

    elapsed_ms = (time.time() - start_time) * 1000.0
    return results, elapsed_ms


def calculate_recall(brute_force_results, server_results, top_k=10):
    """Calculate recall metrics"""
    recalls = []

    for bf_result, server_result in zip(brute_force_results, server_results):
        bf_ids = set(item["id"] for item in bf_result["top_results"])
        server_ids = set(item["id"] for item in server_result["results"])

        if not bf_ids:
            continue  # Skip if brute force found no results

        intersection = bf_ids.intersection(server_ids)
        recall = len(intersection) / len(bf_ids)
        print(
            "Search results for ID",
            bf_result["query_id"],
            f"[{len(intersection)}/{len(bf_ids)}]",
        )
        print("Brute force results:")
        for idx, result in enumerate(bf_result["top_results"]):
            print(f"{idx + 1}. {result['id']} ({result['score']})")
        print("Server results: ")
        for idx, result in enumerate(server_result["results"]):
            print(f"{idx + 1}. {result['id']} ({result['score']})")
        print()
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    return avg_recall, recalls


def run_qps_tests(qps_test_vectors, collection, batch_size=100, top_k=10):
    """Run QPS (Queries Per Second) tests with vectors"""
    print(f"Using {len(qps_test_vectors)} different test vectors for QPS testing")

    start_time_qps = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(0, len(qps_test_vectors), batch_size):
            batch = [vector["text"] for vector in qps_test_vectors[i : i + batch_size]]

            futures.append(executor.submit(batch_ann_search, collection, batch, top_k))

        for future in as_completed(futures):
            try:
                future.result()
                results.append(True)
            except Exception as e:
                print(f"Error in QPS test: {e}")
                results.append(False)

    end_time_qps = time.perf_counter()
    actual_duration = end_time_qps - start_time_qps

    successful_queries = sum(results) * batch_size
    failed_queries = (len(results) * batch_size) - successful_queries
    total_queries = len(results) * batch_size
    qps = successful_queries / actual_duration

    print("QPS Test Results:")
    print(f"Total Queries: {total_queries}")
    print(f"Successful Queries: {successful_queries}")
    print(f"Failed Queries: {failed_queries}")
    print(f"Test Duration: {actual_duration:.2f} seconds")
    print(f"Queries Per Second (QPS): {qps:.2f}")
    print(f"Success Rate: {(successful_queries / total_queries * 100):.2f}%")

    return (
        qps,
        actual_duration,
        successful_queries,
        failed_queries,
        total_queries,
    )


def run_latency_test(collection, query_vectors, top_k):
    print(f"Testing latency {len(query_vectors)} vectors and comparing results...")
    latencies = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []

        def timed_query(query):
            start = time.time()
            resp = search_sparse_vector(collection, query, top_k)
            end = time.time()
            latency = (end - start) * 1000.0
            return resp, latency

        for query in query_vectors:
            futures.append(executor.submit(timed_query, query))

        for future in as_completed(futures):
            try:
                _, latency = future.result()
                latencies.append(latency)
            except Exception as e:
                print(f"Latency test query failed: {e}")

    if not latencies:
        print("No successful latency measurements")
        return 0.0, 0.0

    latencies.sort()
    p50_latency = latencies[int(len(latencies) * 0.5)]
    p95_latency = latencies[int(len(latencies) * 0.95)]

    print("p50 latency:", p50_latency)
    print("p95 latency:", p95_latency)

    return p50_latency, p95_latency


def run_recall_and_qps_and_latency_test(
    collection,
    query_vectors,
    qps_test_vectors,
    top_k,
    brute_force_results,
    batch_size,
):
    # Search vectors and compare results
    print(f"Searching {len(query_vectors)} vectors and comparing results...")
    server_results = []

    start_search = time.time()
    for query in tqdm(query_vectors):
        try:
            result = search_sparse_vector(collection, query, top_k)
            server_results.append(result)
        except Exception as e:
            print(f"Search failed for query {query['id']}: {e}")
            server_results.append({"results": []})

    search_time = time.time() - start_search
    print(
        f"Average search time: {search_time / len(query_vectors):.4f} seconds per query"
    )

    # Calculate and display recall metrics
    avg_recall, recalls = calculate_recall(brute_force_results, server_results, top_k)

    print("\n=== Evaluation Results ===")
    print(f"Average Recall@{top_k}: {avg_recall * 100:.2f}%")
    print(f"Min Recall: {min(recalls) * 100:.2f}%")
    print(f"Max Recall: {max(recalls) * 100:.2f}%")

    # Count perfect recalls
    perfect_recalls = sum(1 for r in recalls if r == 1.0)
    print(
        f"Queries with perfect recall: {perfect_recalls} out of {len(recalls)} ({perfect_recalls / len(recalls) * 100:.2f}%)"
    )

    qps, qps_duration, _, _, _ = run_qps_tests(
        qps_test_vectors, collection, batch_size, top_k
    )

    p50_latency, p95_latency = run_latency_test(
        collection,
        qps_test_vectors[: max((len(qps_test_vectors) // 10), 1000)],
        top_k,
    )

    # Save detailed results
    detailed_results = {
        "avg_recall": avg_recall,
        "qps": qps,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
    }

    return detailed_results


def main(dataset, num_queries=100, batch_size=100, top_k=10, k=1.5, b=0.75):
    vector_db_name = f"sparse_idf_{dataset}_db"
    dataset_folder = f"datasets/sparse_idf_dataset_{dataset}"
    processed_dataset_file = f"{dataset_folder}/processed_vectors.pkl"

    Path(dataset_folder).mkdir(parents=True, exist_ok=True)

    # Check if processed vectors already exist
    if os.path.exists(processed_dataset_file):
        print(f"Loading pre-processed vectors from {processed_dataset_file}")
        with open(processed_dataset_file, "rb") as f:
            vectors = pickle.load(f)
    else:
        # Phase 1: Generate or load raw document vectors with term frequencies
        raw_vectors = get_dataset(dataset)

        # Phase 2: Calculate corpus-wide statistics
        corpus_stats = calculate_corpus_statistics(dataset, raw_vectors)
        avg_document_length = corpus_stats["avg_document_length"]
        print(f"Average document length: {avg_document_length:.2f}")

        # Phase 3: Apply BM25 scoring to vectors
        vectors = apply_bm25_to_vectors(dataset, raw_vectors, corpus_stats, k, b)

    all_query_vectors = get_query_vectors(dataset)

    # only use top 100 for recall test
    query_vectors = all_query_vectors[:num_queries]

    # Compute brute force results
    brute_force_results = compute_brute_force_results(
        dataset, vectors, query_vectors, top_k
    )

    range_min = min(
        min(vector["values"]) for vector in vectors if len(vector["values"]) > 0
    )
    range_max = max(
        max(vector["values"]) for vector in vectors if len(vector["values"]) > 0
    )

    total_queries_length = sum(len(query["indices"]) for query in all_query_vectors)
    total_vectors_length = sum(len(vector["indices"]) for vector in vectors)
    average_query_length = total_queries_length / len(all_query_vectors)
    average_vectors_length = total_vectors_length / len(vectors)

    print(f"Min: {range_min}")
    print(f"Max: {range_max}")
    print("avg query len:", average_query_length)
    print("avg vectors len:", average_vectors_length)

    # Login to get access token
    print("Logging in to server...")
    create_session()
    print("Session established")
    insert_vectors = input("Insert vectors? (Y/n): ").strip().lower() in ["y", ""]

    insertion_time = None
    collection = None

    try:
        if insert_vectors:
            insertion_time, collection = create_db_and_upsert_vectors(
                vector_db_name, vectors, batch_size, k, b
            )
        else:
            try:
                collection = client.get_collection(vector_db_name)
            except Exception as e:
                print(f"Collection not found, creating new one: {e}")
                insertion_time, collection = create_db_and_upsert_vectors(
                    vector_db_name, vectors, batch_size, k, b
                )

        result = run_recall_and_qps_and_latency_test(
            collection,
            query_vectors,
            all_query_vectors[:10_000],
            top_k,
            brute_force_results,
            batch_size,
        )

        if insertion_time is not None:
            result["insertion_time"] = insertion_time

        result["corpus_size"] = len(vectors)
        result["queries_size"] = len(all_query_vectors)

        return result

    finally:
        # Cleanup: delete the collection
        try:
            if collection:
                collection.delete()
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main_non_it(dataset, num_queries=100, batch_size=100, top_k=10, k=1.5, b=0.75):
    vector_db_name = f"sparse_idf_{dataset}_db"
    dataset_folder = f"datasets/sparse_idf_dataset_{dataset}"
    processed_dataset_file = f"{dataset_folder}/processed_vectors.pkl"

    Path(dataset_folder).mkdir(parents=True, exist_ok=True)

    # Check if processed vectors already exist
    if os.path.exists(processed_dataset_file):
        print(f"Loading pre-processed vectors from {processed_dataset_file}")
        with open(processed_dataset_file, "rb") as f:
            vectors = pickle.load(f)
    else:
        # Phase 1: Generate or load raw document vectors with term frequencies
        raw_vectors = get_dataset(dataset)

        # Phase 2: Calculate corpus-wide statistics
        corpus_stats = calculate_corpus_statistics(dataset, raw_vectors)
        avg_document_length = corpus_stats["avg_document_length"]
        print(f"Average document length: {avg_document_length:.2f}")

        # Phase 3: Apply BM25 scoring to vectors
        vectors = apply_bm25_to_vectors(dataset, raw_vectors, corpus_stats, k, b)

    all_query_vectors = get_query_vectors(dataset)

    # only use top 100 for recall test
    query_vectors = all_query_vectors[:num_queries]

    # Compute brute force results
    brute_force_results = compute_brute_force_results(
        dataset, vectors, query_vectors, top_k
    )

    range_min = min(
        min(vector["values"]) for vector in vectors if len(vector["values"]) > 0
    )
    range_max = max(
        max(vector["values"]) for vector in vectors if len(vector["values"]) > 0
    )

    total_queries_length = sum(len(query["indices"]) for query in all_query_vectors)
    total_vectors_length = sum(len(vector["indices"]) for vector in vectors)
    average_query_length = total_queries_length / len(all_query_vectors)
    average_vectors_length = total_vectors_length / len(vectors)

    print(f"Min: {range_min}")
    print(f"Max: {range_max}")
    print("avg query len:", average_query_length)
    print("avg vectors len:", average_vectors_length)

    # Login to get access token
    print("Logging in to server...")
    os.environ["COSDATA_PASSWORD"] = ""
    create_session()
    print("Session established")

    collection = None
    try:
        insertion_time, collection = create_db_and_upsert_vectors(
            vector_db_name, vectors, batch_size, k, b
        )

        result = run_recall_and_qps_and_latency_test(
            collection,
            query_vectors,
            all_query_vectors[:10_000],
            top_k,
            brute_force_results,
            batch_size,
        )

        result["insertion_time"] = insertion_time
        result["corpus_size"] = len(vectors)
        result["queries_size"] = len(all_query_vectors)

        return result

    finally:
        # Cleanup: delete the collection
        try:
            if collection:
                collection.delete()
        except Exception as e:
            print(f"Error during cleanup: {e}")


# def start_server():
#     # Run the shell command to clean, build, and run the server with taskset
#     cmd = 'rm -rf data && cargo b -r && taskset 0xFF ./target/release/cosdata --admin-key "" --confirmed'
#     return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


# def stop_server(process):
#     # Kill the entire process group
#     os.killpg(os.getpgid(process.pid), signal.SIGTERM)


if __name__ == "__main__":
    main(dataset="arguana")
    # datasets = ["trec-covid", "fiqa", "arguana", "webis-touche2020", "quora", "scidocs", "scifact", "nq", "msmarco", "fever", "climate-fever"]
    # results = []
    # for dataset in datasets:
    #     time.sleep(5)
    #     server_proc = start_server()
    #     time.sleep(5)
    #     try:
    #         result = main_non_it(dataset)
    #         result["dataset"] = dataset
    #     except Exception as e:
    #         print(f"Error with dataset {dataset}: {e}")
    #         stop_server(server_proc)
    #         continue
    #     results.append(result)
    #     with open("results.csv", "w", newline="") as csvfile:
    #         fieldnames = ["dataset", "corpus_size", "queries_size", "insertion_time", "avg_recall", "qps", "p50_latency", "p95_latency"]
    #         labels = ["Dataset", "Corpus Size", "Queries Size", "Insertion Time (seconds)", "Average Recall@10", "QPS", "p50 Latency (milliseconds)", "p95 Latency (milliseconds)"]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #         writer.writerow({ fieldname: label for fieldname, label in zip(fieldnames, labels) })
    #         writer.writerows(results)
    #     stop_server(server_proc)
