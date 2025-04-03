import requests
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import random
import getpass
import pickle
from tqdm import tqdm
import heapq
from pathlib import Path
import math
import numpy as np

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"

# Filenames for saving data
RAW_DATASET_FILE = "sparse_idf_dataset/raw_vectors.pkl"
PROCESSED_DATASET_FILE = "sparse_idf_dataset/processed_vectors.pkl"
QUERY_VECTORS_FILE = "sparse_idf_dataset/query.pkl"
BRUTE_FORCE_RESULTS_FILE = "sparse_idf_dataset/brute_force_results.pkl"
STATS_FILE = "sparse_idf_dataset/corpus_stats.pkl"

def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}

def create_session():
    url = f"{host}/auth/create-session"
    if "ADMIN_PASSWORD" in os.environ:
        password = os.environ["ADMIN_PASSWORD"]
    else:
        password = getpass.getpass("Enter admin password: ")

    data = {"username": "admin", "password": password}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    session = response.json()
    global token
    token = session["access_token"]
    return token

def create_db(name, description=None, dimension=20000):
    url = f"{base_url}/collections"
    data = {
        "name": name,
        "description": description,
        "dense_vector": {
            "enabled": False,
            "auto_create_index": False,
            "dimension": dimension,
        },
        "sparse_vector": {"enabled": True, "auto_create_index": False},
        "metadata_schema": None,
        "config": {"max_vectors": None, "replication_factor": None},
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()

def create_explicit_index(name):
    data = {
        "name": name,
        "quantization": 64,
        "early_terminate_threshold": 0.0,
        "isIDF": True,
    }
    response = requests.post(
        f"{base_url}/collections/{name}/indexes/sparse",
        headers=generate_headers(),
        data=json.dumps(data),
        verify=False,
    )
    return response.json()

def compute_bm25_idf(total_documents, docs_containing_term):
    """
    Compute IDF using the BM25 formula:
    ln(((total documents - documents contain the term + 0.5) / (documents containing the term + 0.5)) + 1)
    """
    print(total_documents, docs_containing_term)
    return math.log1p((total_documents - docs_containing_term + 0.5) / (docs_containing_term + 0.5))

def compute_bm25_term_frequency(count, document_length, avg_document_length, k=1.5, b=0.75):
    """
    Normalize term frequency for BM25 scoring:
    tf * (k+1) / (tf + k * (1 - b + b * (dl/avgdl)))
    """
    normalized_freq = (
        count * (k + 1) / 
        (count + k * (1 - b + b * (document_length / avg_document_length)))
    )
    return normalized_freq

def precompute_probabilities(min_val: int, max_val: int, step: int):
    num_bins = (max_val - min_val) // step
    probabilities = np.array([2**i for i in range(num_bins)])
    probabilities = probabilities / probabilities.sum()  # Normalize to sum to 1
    return probabilities, num_bins

def custom_rng(min_val: int, max_val: int, probs, num_bins, step) -> int:
        # Choose a range based on probabilities
    chosen_bin = np.random.choice(num_bins, p=probs)
    
    # Generate a number within the chosen range
    return np.random.randint(min_val + chosen_bin * step, min_val + (chosen_bin + 1) * step)

def generate_raw_document_vector(id, terms, probs, num_bins, step):
    """
    Generate a raw document vector with term frequencies
    """
    document_length = 0
    num_terms = random.randint(100, 150)

    indices = set()
    while len(indices) < num_terms:
        indices.add(terms[custom_rng(0, len(terms), probs, num_bins, step)])
    
    indices = sorted(indices)
    values = []
    
    for _ in indices:
        # 80% chance of frequency 1, 20% chance of higher frequencies
        freq = 1 if random.random() < 0.8 else random.randint(2, 5)
        document_length += freq
        values.append(freq)
    
    return {
        "id": id,
        "indices": indices,
        "raw_term_frequencies": values,
        "length": document_length
    }

def generate_raw_dataset(num_vectors):
    """Generate raw dataset with just term frequencies"""
    if os.path.exists(RAW_DATASET_FILE):
        print(f"Loading existing raw dataset from {RAW_DATASET_FILE}")
        with open(RAW_DATASET_FILE, 'rb') as f:
            return pickle.load(f)

    num_terms = 50_000
    steps = num_terms // 10

    terms = random.sample(range(0, 2**32), num_terms)
    probs, num_bins = precompute_probabilities(0, num_terms, steps)
    
    print(f"Generating {num_vectors} raw document vectors...")
    vectors = [
        generate_raw_document_vector(id, terms, probs, num_bins, steps)
        for id in tqdm(range(num_vectors))
    ]
    
    # Save to disk
    with open(RAW_DATASET_FILE, 'wb') as f:
        pickle.dump(vectors, f)
    
    print(f"Raw dataset generated and saved to {RAW_DATASET_FILE}")
    return vectors

def calculate_corpus_statistics(vectors):
    """Calculate corpus-wide statistics for BM25"""
    if os.path.exists(STATS_FILE):
        print(f"Loading existing corpus statistics from {STATS_FILE}")
        with open(STATS_FILE, 'rb') as f:
            return pickle.load(f)
    
    print("Calculating corpus statistics...")
    
    # Calculate average document length
    total_length = sum(vec["length"] for vec in vectors)
    avg_document_length = total_length / len(vectors)
    
    # Count documents containing each term
    term_document_counts = {}
    for vec in tqdm(vectors):
        for term_index in vec["indices"]:
            term_document_counts[term_index] = term_document_counts.get(term_index, 0) + 1
    
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
    
    with open(STATS_FILE, 'wb') as f:
        pickle.dump(statistics, f)
    
    print(f"Corpus statistics calculated and saved to {STATS_FILE}")
    return statistics

def apply_bm25_to_vectors(vectors, corpus_stats, k=1.5, b=0.75):
    """Apply BM25 scoring to all vectors using corpus statistics"""
    if os.path.exists(PROCESSED_DATASET_FILE):
        print(f"Loading existing processed vectors from {PROCESSED_DATASET_FILE}")
        with open(PROCESSED_DATASET_FILE, 'rb') as f:
            return pickle.load(f)
    
    print("Applying BM25 scoring to vectors...")
    
    avg_document_length = corpus_stats["avg_document_length"]
    term_idf_values = corpus_stats["term_idf_values"]
    
    processed_vectors = []
    
    for vec in tqdm(vectors):
        # Calculate normalized term frequencies for BM25
        server_values = []
        brute_force_values = []
        
        for idx, (term_index, raw_freq) in enumerate(zip(vec["indices"], vec["raw_term_frequencies"])):
            # Normalize term frequency
            normalized_freq = compute_bm25_term_frequency(
                raw_freq, 
                vec["length"], 
                avg_document_length,
                k,
                b
            )
            
            # Get IDF for this term
            idf = term_idf_values.get(term_index, 0)
            
            # Store normalized frequency for server
            server_values.append(normalized_freq)
            
            # Store TF-IDF for local brute force calculation
            brute_force_values.append(normalized_freq * idf)
        
        processed_vectors.append({
            "id": vec["id"],
            "indices": vec["indices"],
            "values": server_values,  # Normalized term frequency for server
            "brute_force_values": brute_force_values,  # TF-IDF for local calculation
            "length": vec["length"],
            "raw_term_frequencies": vec["raw_term_frequencies"]  # Keep original frequencies
        })
    
    # Save processed vectors
    with open(PROCESSED_DATASET_FILE, 'wb') as f:
        pickle.dump(processed_vectors, f)
    
    print(f"Processed vectors saved to {PROCESSED_DATASET_FILE}")
    return processed_vectors

def select_query_vectors(vectors, num_queries):
    """
    Select random query vectors from the dataset and create sparse queries with only
    the top 10 terms by term frequency
    """
    if os.path.exists(QUERY_VECTORS_FILE):
        print(f"Loading existing query vectors from {QUERY_VECTORS_FILE}")
        with open(QUERY_VECTORS_FILE, 'rb') as f:
            return pickle.load(f)
    
    print(f"Selecting {num_queries} random query vectors with top 10 terms...")
    # Select random vectors from the dataset
    query_indices = random.sample(range(len(vectors)), num_queries)
    query_vectors = []
    
    for idx in query_indices:
        original_vector = vectors[idx]
        
        # Create a list of (term_index, raw_freq, server_value, brute_force_value) tuples
        term_data = list(zip(
            original_vector["indices"], 
            original_vector["raw_term_frequencies"],
            original_vector["values"],
            original_vector["brute_force_values"]
        ))
        
        # Sort by raw term frequency in descending order
        term_data.sort(key=lambda x: x[1], reverse=True)
        
        # Take only the top 10 terms (or fewer if the vector has less than 10 terms)
        top_terms = term_data[:min(10, len(term_data))]
        
        # Extract the data back into separate lists
        indices, raw_term_frequencies, values, brute_force_values = zip(*top_terms) if top_terms else ([], [], [], [])
        
        # Create a new query vector with only the top terms
        query_vectors.append({
            "id": original_vector["id"],
            "indices": list(indices),
            "values": list(values),
            "brute_force_values": list(brute_force_values),
            "raw_term_frequencies": list(raw_term_frequencies),
            "length": sum(raw_term_frequencies)  # Recalculate the document length
        })
    
    # Save to disk
    with open(QUERY_VECTORS_FILE, 'wb') as f:
        pickle.dump(query_vectors, f)
    
    print(f"Query vectors with top 10 terms selected and saved to {QUERY_VECTORS_FILE}")
    return query_vectors

def compute_brute_force_results(vectors, query_vectors, top_k=10):
    """Compute dot product similarity between query vectors and all vectors"""
    if os.path.exists(BRUTE_FORCE_RESULTS_FILE):
        print(f"Loading existing brute force results from {BRUTE_FORCE_RESULTS_FILE}")
        with open(BRUTE_FORCE_RESULTS_FILE, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing brute force dot product similarity for {len(query_vectors)} queries...")
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
        top_indices = heapq.nlargest(top_k, range(len(dot_products)), key=lambda i: dot_products[i])
        
        top_results = [
            {"id": vectors[idx]["id"], "score": float(dot_products[idx])}
            for idx in top_indices
        ][:top_k]
        
        results.append({
            "query_id": query["id"],
            "top_results": top_results
        })
    
    # Save to disk
    with open(BRUTE_FORCE_RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Brute force results computed and saved to {BRUTE_FORCE_RESULTS_FILE}")
    return results

def format_for_server_query(vector):
    """Format a vector for server query"""
    return {
        "id": vector["id"],
        "values": [[ind, val] for ind, val in zip(vector["indices"], vector["values"])]
    }

def create_transaction(collection_name):
    url = f"{base_url}/collections/{collection_name}/transactions"
    data = {"index_type": "sparse"}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()

def upsert_in_transaction(vector_db_name, transaction_id, vectors):
    """
    Upsert vectors to the API, excluding unnecessary fields
    """
    # Create a new list of vectors with only needed fields
    filtered_vectors = [
        {
            "id": vec["id"],
            "indices": vec["indices"],
            "values": vec["values"]
        }
        for vec in vectors
    ]
    
    url = f"{base_url}/collections/{vector_db_name}/transactions/{transaction_id}/upsert"
    data = {"index_type": "sparse", "vectors": filtered_vectors}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to create vector: {response.status_code} - {response.text}")

def commit_transaction(collection_name, transaction_id):
    url = f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/commit"
    data = {"index_type": "sparse"}
    response = requests.post(
        url, data=json.dumps(data), headers=generate_headers(), verify=False
    )
    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to commit transaction: {response.status_code}")
    return response.json() if response.text else None

def search_sparse_vector(vector_db_name, vector, top_k=10):
    url = f"{base_url}/collections/{vector_db_name}/vectors/search"
    data = {
        "index_type": "sparse",
        "values": vector["values"],
        "top_k": top_k,
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()

def batch_ann_search(vector_db_name, batch, top_k=10):
    url = f"{base_url}/collections/{vector_db_name}/vectors/batch-search"
    data = {
        "index_type": "sparse",
        "vectors": batch,
        "top_k": top_k,
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code not in [200, 204]:
        print(f"Error response ({response.status_code}): {response.text}")
    return response.json()

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
        print("Search results for ID", bf_result["query_id"], f"[{len(intersection)}/{len(bf_ids)}]")
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

def run_rps_tests(rps_test_vectors, vector_db_name, batch_size=100):
    """Run RPS (Requests Per Second) tests with truncated vectors (top 10 terms only)"""
    print(f"Using {len(rps_test_vectors)} different test vectors for RPS testing (top 10 terms only)")

    # First, truncate the vectors to only include top 10 terms
    truncated_vectors = []
    for vector in rps_test_vectors:
        # Create a list of (term_index, value) tuples
        term_data = list(zip(vector["indices"], vector["raw_term_frequencies"], vector["values"]))
        
        # Sort by raw term frequency in descending order
        term_data.sort(key=lambda x: x[1], reverse=True)
        
        # Take only the top 10 terms (or fewer if the vector has less than 10 terms)
        top_terms = term_data[:min(10, len(term_data))]
        
        # Extract the data back into separate lists
        indices, _, values = zip(*top_terms) if top_terms else ([], [], [])
        
        # Create a truncated vector with only the top terms
        truncated_vectors.append({
            "id": vector["id"],
            "indices": list(indices),
            "values": list(values)
        })

    start_time_rps = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(0, len(truncated_vectors), batch_size):
            batch = [
                [[ind, val] for ind, val in zip(vector["indices"], vector["values"])] 
                for vector in truncated_vectors[i : i + batch_size]
            ]
            futures.append(executor.submit(batch_ann_search, vector_db_name, batch))

        for future in as_completed(futures):
            try:
                future.result()
                results.append(True)
            except Exception as e:
                print(f"Error in RPS test: {e}")
                results.append(False)

    end_time_rps = time.time()
    actual_duration = end_time_rps - start_time_rps

    successful_requests = sum(results) * batch_size
    failed_requests = (len(results) * batch_size) - successful_requests
    total_requests = len(results) * batch_size
    rps = successful_requests / actual_duration

    print("\nRPS Test Results (top 10 terms per vector):")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_requests}")
    print(f"Test Duration: {actual_duration:.2f} seconds")
    print(f"Requests Per Second (RPS): {rps:.2f}")
    print(f"Success Rate: {(successful_requests / total_requests * 100):.2f}%")

def main():
    vector_db_name = "sparse_vector_db"
    num_vectors = 1_000_000
    num_queries = 100
    batch_size = 100
    top_k = 10
    k_value = 1.5  # BM25 k parameter
    b_value = 0.75  # BM25 b parameter

    Path("sparse_idf_dataset").mkdir(parents=True, exist_ok=True)
    
    # Check if processed vectors already exist
    if os.path.exists(PROCESSED_DATASET_FILE):
        print(f"Loading pre-processed vectors from {PROCESSED_DATASET_FILE}")
        with open(PROCESSED_DATASET_FILE, 'rb') as f:
            vectors = pickle.load(f)
    else:
        # Phase 1: Generate or load raw document vectors with term frequencies
        raw_vectors = generate_raw_dataset(num_vectors)
        
        # Phase 2: Calculate corpus-wide statistics
        corpus_stats = calculate_corpus_statistics(raw_vectors)
        avg_document_length = corpus_stats["avg_document_length"]
        print(f"Average document length: {avg_document_length:.2f}")
        
        # Phase 3: Apply BM25 scoring to vectors
        vectors = apply_bm25_to_vectors(raw_vectors, corpus_stats, k_value, b_value)
    
    # Select query vectors
    query_vectors = select_query_vectors(vectors, num_queries)
    
    # Compute brute force results
    brute_force_results = compute_brute_force_results(vectors, query_vectors, top_k)

    range_min = min(min(vector["values"]) for vector in vectors)
    range_max = max(max(vector["values"]) for vector in vectors)

    print(f"Min: {range_min}")
    print(f"Max: {range_max}")
    
    # Login to get access token
    print("Logging in to server...")
    create_session()
    print("Session established")
    insert_vectors = input("Insert vectors? (Y/n): ").strip().lower() in ["y", ""]
    
    if insert_vectors:
        # Create collection
        try:
            print(f"Creating collection: {vector_db_name}")
            create_db(name=vector_db_name, dimension=1) # dummy, not sure if we need it
            print("Collection created")
        
            # Create explicit index
            create_explicit_index(vector_db_name)
            print("Explicit index created")
        except Exception as e:
            print(f"Collection may already exist: {e}")
    
        # Insert vectors into server
        print("Creating transaction")
        transaction_id = create_transaction(vector_db_name)["transaction_id"]
    
        print(f"Inserting {num_vectors} vectors in batches of {batch_size}...")
        start = time.time()
    
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for batch_start in range(0, num_vectors, batch_size):
                batch = vectors[batch_start: batch_start + batch_size]
                futures.append(
                    executor.submit(
                        upsert_in_transaction, vector_db_name, transaction_id, batch
                    )
                )
        
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                try:
                    future.result()
                except Exception as e:
                    print(f"Batch {i + 1} failed: {e}")
    
        print("Committing transaction")
        commit_transaction(vector_db_name, transaction_id)
    
        end = time.time()
        insertion_time = end - start
        print(f"Insertion time: {insertion_time:.2f} seconds")
    
    # Search vectors and compare results
    print(f"Searching {num_queries} vectors and comparing results...")
    server_results = []

    start_search = time.time()
    for query in tqdm(query_vectors):
        formatted_query = format_for_server_query(query)
        try:
            result = search_sparse_vector(vector_db_name, formatted_query, top_k)
            server_results.append(result)
        except Exception as e:
            print(f"Search failed for query {query['id']}: {e}")
            server_results.append({"Sparse": []})

    search_time = time.time() - start_search
    print(f"Average search time: {search_time / num_queries:.4f} seconds per query")

    # Calculate and display recall metrics
    avg_recall, recalls = calculate_recall(brute_force_results, server_results, top_k)

    print("\n=== Evaluation Results ===")
    print(f"Average Recall@{top_k}: {avg_recall * 100:.2f}%")
    print(f"Min Recall: {min(recalls) * 100:.2f}%")
    print(f"Max Recall: {max(recalls) * 100:.2f}%")

    # Count perfect recalls
    perfect_recalls = sum(1 for r in recalls if r == 1.0)
    print(f"Queries with perfect recall: {perfect_recalls} out of {len(recalls)} ({perfect_recalls / len(recalls) * 100:.2f}%)")

    # Save detailed results
    detailed_results = {
        "avg_recall": avg_recall,
        "recalls": recalls,
        "perfect_recalls": perfect_recalls,
        "search_time": search_time / num_queries,
        "bm25_parameters": {
            "k": k_value,
            "b": b_value,
            "avg_document_length": corpus_stats["avg_document_length"] if "corpus_stats" in locals() else None
        }
    }

    with open("sparse_idf_dataset/vector_evaluation_results.pkl", 'wb') as f:
        pickle.dump(detailed_results, f)

    print("Detailed results saved to sparse_idf_dataset/vector_evaluation_results.pkl")

    run_rps_tests(vectors[:10_000], vector_db_name, batch_size)
    
if __name__ == "__main__":
    main()
