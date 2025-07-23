import requests
import json
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import random
import pickle
from tqdm import tqdm
from pathlib import Path
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from dotenv import load_dotenv
from cosdata import Client
import getpass

# Load environment variables from .env file
load_dotenv()

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
client = None
host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")

# Filenames for saving data
DATASET_FILE = "datasets/sparse_dataset/vectors.pkl"
QUERY_VECTORS_FILE = "datasets/sparse_dataset/query.pkl"
BRUTE_FORCE_RESULTS_FILE = "datasets/sparse_dataset/brute_force_results.pkl"


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
        sparse_vector={"enabled": True},
        dense_vector={"enabled": False, "dimension": dimension},
        tf_idf_options={"enabled": False},
    )
    return collection


def create_explicit_index(collection):
    """Create sparse index using cosdata client"""
    index = collection.create_sparse_index(
        name=collection.name,
        quantization=64,
        sample_threshold=1000,
    )
    return index


def create_transaction(collection):
    """Create transaction using cosdata client"""
    return collection.transaction()


def upsert_in_transaction(collection, transaction, vectors):
    """
    Upsert vectors to the collection using transaction
    """
    # Convert to the format expected by the client
    formatted_vectors = [
        {
            "id": vector["id"],
            "sparse_values": vector["values"],
            "sparse_indices": vector["indices"],
        }
        for vector in vectors
    ]

    transaction.batch_upsert_vectors(formatted_vectors)


def commit_transaction(transaction):
    """Commit transaction using cosdata client"""
    transaction.commit()
    return transaction.transaction_id


def search_sparse_vector(collection, query_terms, top_k=10, early_terminate_threshold=0.0):
    """Search using sparse vector with cosdata client"""
    # query_terms should already be in the correct format from format_for_server_query
    results = collection.search.sparse(
        query_terms=query_terms,
        top_k=top_k,
        early_terminate_threshold=early_terminate_threshold,
    )
    return results


def batch_ann_search(collection, batch, top_k=10, early_terminate_threshold=0.0):
    """Batch sparse search using cosdata client"""
    try:
        results = collection.search.batch_sparse(
            query_terms_list=batch,
            top_k=top_k,
            early_terminate_threshold=early_terminate_threshold,
        )
        return results
    except Exception as e:
        print(f"Error in batch search: {e}")
        # Fallback to individual searches
        results = []
        for query_terms in batch:
            try:
                result = collection.search.sparse(
                    query_terms=query_terms,
                    top_k=top_k,
                    early_terminate_threshold=early_terminate_threshold,
                )
                results.append(result)
            except Exception as e2:
                print(f"Individual search failed: {e2}")
                results.append({"results": []})
        return results


def generate_random_sparse_vector(id, dimension, non_zero_dims):
    # Generate a random number of non-zero dimensions between 20 and 100
    actual_non_zero_dims = random.randint(20, non_zero_dims)

    # Generate unique indices
    indices = sorted(random.sample(range(dimension), actual_non_zero_dims))

    # Generate values between 0 and 2.0
    values = np.random.uniform(0.0, 2.0, actual_non_zero_dims).tolist()

    return {"id": str(id), "indices": indices, "values": values}


def generate_dataset(num_vectors, dimension, max_non_zero_dims):
    """Generate dataset if not already on disk"""
    if os.path.exists(DATASET_FILE):
        print(f"Loading existing dataset from {DATASET_FILE}")
        with open(DATASET_FILE, "rb") as f:
            vectors = pickle.load(f)
        return vectors

    print(f"Generating {num_vectors} random sparse vectors...")
    vectors = [
        generate_random_sparse_vector(id, dimension, max_non_zero_dims)
        for id in tqdm(range(num_vectors))
    ]

    # Save to disk
    with open(DATASET_FILE, "wb") as f:
        pickle.dump(vectors, f)

    print(f"Dataset generated and saved to {DATASET_FILE}")
    return vectors


def select_query_vectors(vectors, num_queries):
    """Select random query vectors from the dataset"""
    if os.path.exists(QUERY_VECTORS_FILE):
        print(f"Loading existing query vectors from {QUERY_VECTORS_FILE}")
        with open(QUERY_VECTORS_FILE, "rb") as f:
            query_indices = pickle.load(f)
        return [vectors[i] for i in query_indices]

    print(f"Selecting {num_queries} random query vectors...")
    query_indices = random.sample(range(len(vectors)), num_queries)
    query_vectors = [vectors[i] for i in query_indices]

    # Save indices to disk
    with open(QUERY_VECTORS_FILE, "wb") as f:
        pickle.dump(query_indices, f)

    print(f"Query vectors selected and indices saved to {QUERY_VECTORS_FILE}")
    return query_vectors


def compute_brute_force_results(vectors, query_vectors, dimension, top_k=10):
    """Compute dot product similarity using GPU acceleration with CuPy, fallback to CPU"""
    if os.path.exists(BRUTE_FORCE_RESULTS_FILE):
        print(f"Loading existing brute force results from {BRUTE_FORCE_RESULTS_FILE}")
        with open(BRUTE_FORCE_RESULTS_FILE, "rb") as f:
            return pickle.load(f)

    print(
        f"Computing brute force dot product similarity for {len(query_vectors)} queries..."
    )
    results = []

    try:
        # Try GPU computation first
        print("Attempting GPU computation...")
        results = _compute_brute_force_gpu(vectors, query_vectors, dimension, top_k)
    except Exception as e:
        print(f"GPU computation failed: {e}")
        print("Falling back to CPU computation...")
        results = _compute_brute_force_cpu(vectors, query_vectors, dimension, top_k)

    # Save to disk
    with open(BRUTE_FORCE_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"Brute force results computed and saved to {BRUTE_FORCE_RESULTS_FILE}")
    return results


def _compute_brute_force_cpu(vectors, query_vectors, dimension, top_k=10):
    """CPU-based brute force computation using scipy.sparse"""
    from scipy.sparse import csr_matrix as cpu_csr_matrix

    results = []
    n_vectors = len(vectors)
    print("Building dataset sparse matrix on CPU...")

    # Build dataset matrix
    data_list = []
    indices_list = []
    indptr = [0]

    for vec in tqdm(vectors):
        data_list.extend(vec["values"])
        indices_list.extend(vec["indices"])
        indptr.append(len(data_list))

    A = cpu_csr_matrix((data_list, indices_list, indptr), shape=(n_vectors, dimension))

    print("Computing similarities...")
    for i, query in enumerate(tqdm(query_vectors)):
        # Build query vector
        q_data = query["values"]
        q_indices = query["indices"]
        q_indptr = [0, len(q_data)]

        Q = cpu_csr_matrix((q_data, q_indices, q_indptr), shape=(1, dimension))

        # Compute dot products
        scores = A.dot(Q.T).toarray().flatten()

        # Get top-k
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        top_results = []
        for idx in top_indices:
            top_results.append({"id": vectors[idx]["id"], "score": float(scores[idx])})

        results.append({"query_id": query["id"], "top_results": top_results})

    return results


def _compute_brute_force_gpu(vectors, query_vectors, dimension, top_k=10):
    """GPU-based brute force computation with better memory management"""
    results = []
    n_vectors = len(vectors)

    # Check available GPU memory
    mempool = cp.get_default_memory_pool()
    available_memory = mempool.free_bytes() + mempool.total_bytes()
    print(f"Available GPU memory: {available_memory / 1e9:.2f} GB")

    # Process in smaller chunks if needed
    max_vectors_per_chunk = min(100000, n_vectors)  # Limit chunk size

    print("Building dataset sparse matrix on GPU in chunks...")

    for chunk_start in tqdm(range(0, n_vectors, max_vectors_per_chunk)):
        chunk_end = min(chunk_start + max_vectors_per_chunk, n_vectors)
        chunk_vectors = vectors[chunk_start:chunk_end]

        # Build chunk matrix
        data_list = []
        indices_list = []
        indptr = [0]

        for vec in chunk_vectors:
            data_list.append(vec["values"])
            indices_list.append(vec["indices"])
            indptr.append(indptr[-1] + len(vec["indices"]))

        # Use smaller data types to save memory
        data_gpu = cp.concatenate([cp.array(v, dtype=cp.float32) for v in data_list])
        indices_gpu = cp.concatenate(
            [cp.array(v, dtype=cp.int32) for v in indices_list]
        )
        indptr_gpu = cp.array(indptr, dtype=cp.int32)

        # Create CSR matrix for this chunk
        A_chunk = csr_matrix(
            (data_gpu, indices_gpu, indptr_gpu), shape=(len(chunk_vectors), dimension)
        )

        # Process queries against this chunk
        for query in query_vectors:
            # Build query vector
            q_data = cp.array(query["values"], dtype=cp.float32)
            q_indices = cp.array(query["indices"], dtype=cp.int32)
            q_indptr = cp.array([0, len(q_data)], dtype=cp.int32)

            Q = csr_matrix((q_data, q_indices, q_indptr), shape=(1, dimension))

            # Compute dot products for this chunk
            scores_chunk = A_chunk.dot(Q.T).toarray().flatten()

            # Update results for this query
            if chunk_start == 0:
                # Initialize results for this query
                query_idx = query_vectors.index(query)
                if query_idx >= len(results):
                    results.extend([None] * (query_idx + 1 - len(results)))
                results[query_idx] = {
                    "query_id": query["id"],
                    "scores": scores_chunk.copy(),
                    "vector_offset": chunk_start,
                }
            else:
                # Combine with previous chunks
                query_idx = query_vectors.index(query)
                if results[query_idx] is not None:
                    results[query_idx]["scores"] = cp.concatenate(
                        [results[query_idx]["scores"], scores_chunk]
                    )

        # Clean up GPU memory
        del A_chunk, data_gpu, indices_gpu, indptr_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Convert to final format with top-k
    final_results = []
    for result in results:
        if result is None:
            continue

        scores = result["scores"].get()  # Transfer to CPU

        # Get top-k
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        top_results = []
        for idx in top_indices:
            top_results.append({"id": vectors[idx]["id"], "score": float(scores[idx])})

        final_results.append(
            {"query_id": result["query_id"], "top_results": top_results}
        )

    return final_results


def format_for_server_query(vector):
    """Format a vector for server query"""
    # Return format expected by sparse search API: list of [index, value] tuples
    return [[ind, val] for ind, val in zip(vector["indices"], vector["values"])]

def calculate_recall(brute_force_results, server_results, top_k=10):
    """Calculate recall metrics"""
    recalls = []

    for i, (bf_result, server_result) in enumerate(
        zip(brute_force_results, server_results)
    ):
        # Debug: Print first few server results to understand structure
        if i < 3:
            print(f"\nDebug - Query {i}:")
            print(f"Brute force top 3: {bf_result['top_results'][:3]}")
            print(f"Server result keys: {list(server_result.keys())}")
            print(f"Server result: {server_result}")

        bf_ids = set(item["id"] for item in bf_result["top_results"])

        # Handle different possible server response structures
        if "results" in server_result:
            server_items = server_result["results"]
        elif "vectors" in server_result:
            server_items = server_result["vectors"]
        else:
            print(f"Warning: Unexpected server result structure: {server_result}")
            server_items = []

        # Extract IDs from server results - handle different possible structures
        server_ids = set()
        for item in server_items:
            if isinstance(item, dict):
                if "id" in item:
                    server_ids.add(item["id"])
                elif "vector_id" in item:
                    server_ids.add(item["vector_id"])
                else:
                    print(f"Warning: Unknown server item structure: {item}")
            else:
                print(f"Warning: Server item is not a dict: {item}")

        if not bf_ids:
            continue  # Skip if brute force found no results

        intersection = bf_ids.intersection(server_ids)
        recall = len(intersection) / len(bf_ids)
        recalls.append(recall)

        # Debug: Print recall for first few queries
        if i < 3:
            print(f"BF IDs (first 5): {list(bf_ids)[:5]}")
            print(f"Server IDs (first 5): {list(server_ids)[:5]}")
            print(f"Intersection: {len(intersection)}")
            print(f"Recall: {recall:.2%}")

    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    return avg_recall, recalls


def run_rps_tests(
    rps_test_vectors,
    collection,
    batch_size=100,
    top_k=10,
    early_terminate_threshold=0.0,
):
    """Run RPS (Requests Per Second) tests"""
    print(f"Using {len(rps_test_vectors)} different test vectors for RPS testing")

    start_time_rps = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(0, len(rps_test_vectors), batch_size):
            batch = [
                format_for_server_query(vector)
                for vector in rps_test_vectors[i : i + batch_size]
            ]
            futures.append(
                executor.submit(
                    batch_ann_search,
                    collection,
                    batch,
                    top_k,
                    early_terminate_threshold,
                )
            )

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

    print("\nRPS Test Results:")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_requests}")
    print(f"Test Duration: {actual_duration:.2f} seconds")
    print(f"Requests Per Second (RPS): {rps:.2f}")
    print(f"Success Rate: {(successful_requests / total_requests * 100):.2f}%")


def main():
    vector_db_name = "sparse_eval_db"
    num_vectors = 1_000_000
    dimension = 20_000
    max_non_zero_dims = 100
    num_queries = 100
    batch_size = 100
    top_k = 10
    early_terminate_threshold = 0.5

    Path("datasets/sparse_dataset").mkdir(parents=True, exist_ok=True)

    # Generate or load dataset
    vectors = generate_dataset(num_vectors, dimension, max_non_zero_dims)

    # Select query vectors
    query_vectors = select_query_vectors(vectors, num_queries)

    # Compute brute force results
    brute_force_results = compute_brute_force_results(
        vectors, query_vectors, dimension, top_k
    )

    # Login to get access token
    print("Logging in to server...")
    create_session()
    print("Session established")
    insert_vectors = input("Insert vectors? (Y/n): ").strip().lower() in ["y", ""]

    collection = None
    if insert_vectors:
        # Create collection
        try:
            print(f"Creating collection: {vector_db_name}")
            collection = create_db(name=vector_db_name, dimension=dimension)
            print("Collection created")

            # Create explicit index
            create_explicit_index(collection)
            print("Explicit index created")
        except Exception as e:
            print(f"Collection may already exist: {e}")
            try:
                collection = client.get_collection(vector_db_name)
                print("Using existing collection")
            except Exception as get_error:
                print(f"Failed to get existing collection: {get_error}")
                raise Exception(f"Cannot create or access collection: {e}")

        # Insert vectors into server using a single transaction
        print("Creating single transaction for all vectors")
        print(f"Inserting {num_vectors} vectors in batches of {batch_size}...")
        start = time.time()

        # Create single transaction and process all batches within it
        with collection.transaction() as txn:
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for batch_start in range(0, num_vectors, batch_size):
                    batch = vectors[batch_start : batch_start + batch_size]
                    futures.append(
                        executor.submit(upsert_in_transaction, collection, txn, batch)
                    )

                for i, future in enumerate(
                    tqdm(as_completed(futures), total=len(futures))
                ):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Batch {i + 1} failed: {e}")

            # Transaction is automatically committed when exiting the context manager
            transaction_id = txn.transaction_id

        end = time.time()
        insertion_time = end - start
        print(f"Insertion time: {insertion_time:.2f} seconds")
        print(f"Transaction committed with ID: {transaction_id}")

        # Wait for indexing to complete
        print("Waiting for indexing to complete...")
        final_status, success = transaction_id.poll_completion(
            target_status="complete",
            max_attempts=30,  # Increase attempts for large dataset
            sleep_interval=5,  # Longer sleep for indexing
        )

        if not success:
            print(
                f"Warning: Indexing may not have completed. Final status: {final_status}"
            )
        else:
            print("Indexing completed successfully")
    else:
        try:
            collection = client.get_collection(vector_db_name)
            print("Using existing collection")
        except Exception as e:
            print(f"Collection not found: {e}")
            return

    # Search vectors and compare results
    print(f"Searching {num_queries} vectors and comparing results...")
    server_results = []

    start_search = time.time()
    for i, query in enumerate(tqdm(query_vectors)):
        formatted_query = format_for_server_query(query)
        try:
            result = search_sparse_vector(
                collection, formatted_query, top_k, early_terminate_threshold
            )
            server_results.append(result)

            if i == 0:
                print("\nDebug - First search result structure:")
                print(f"Result keys: {list(result.keys())}")
                print(f"Result: {result}")
        except Exception as e:
            print(f"Search failed for query {query['id']}: {e}")
            server_results.append({"results": []})

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
    print(
        f"Queries with perfect recall: {perfect_recalls} out of {len(recalls)} ({perfect_recalls / len(recalls) * 100:.2f}%)"
    )

    # Save detailed results
    detailed_results = {
        "avg_recall": avg_recall,
        "recalls": recalls,
        "perfect_recalls": perfect_recalls,
        "search_time": search_time / num_queries,
    }

    with open("datasets/sparse_dataset/vector_evaluation_results.pkl", "wb") as f:
        pickle.dump(detailed_results, f)

    print(
        "Detailed results saved to datasets/sparse_dataset/vector_evaluation_results.pkl"
    )

    run_rps_tests(
        vectors[:10_000], collection, batch_size, top_k, early_terminate_threshold
    )

    # Cleanup
    try:
        collection.delete()
        print("Test collection deleted")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
