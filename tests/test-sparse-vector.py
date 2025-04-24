import requests
import json
import numpy as np
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

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"

# Filenames for saving data
DATASET_FILE = "sparse_dataset/vectors.pkl"
QUERY_VECTORS_FILE = "sparse_dataset/query.pkl"
BRUTE_FORCE_RESULTS_FILE = "sparse_dataset/brute_force_results.pkl"


def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}


# Function to login with credentials
def create_session():
    url = f"{host}/auth/create-session"
    # Use environment variable if available, otherwise prompt
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
            "dimension": dimension,
        },
        "sparse_vector": {"enabled": True},
        "tf_idf_options": {"enabled": False},
        "metadata_schema": None,
        "config": {"max_vectors": None, "replication_factor": None},
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def create_explicit_index(name):
    data = {
        "name": name,  # Name of the index
        "quantization": 64,
        "sample_threshold": 1000,
    }
    response = requests.post(
        f"{base_url}/collections/{name}/indexes/sparse",
        headers=generate_headers(),
        data=json.dumps(data),
        verify=False,
    )
    return response.json()


def create_transaction(collection_name):
    url = f"{base_url}/collections/{collection_name}/transactions"
    response = requests.post(url, headers=generate_headers(), verify=False)
    return response.json()


def upsert_in_transaction(vector_db_name, transaction_id, vectors):
    url = (
        f"{base_url}/collections/{vector_db_name}/transactions/{transaction_id}/upsert"
    )
    vectors = [
        {
            "id": vector["id"],
            "sparse_values": vector["values"],
            "sparse_indices": vector["indices"],
        }
        for vector in vectors
    ]
    data = {"vectors": vectors}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code not in [200, 204]:
        raise Exception(
            f"Failed to create vector: {response.status_code} - {response.text}"
        )


def commit_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/commit"
    )
    response = requests.post(url, headers=generate_headers(), verify=False)
    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to commit transaction: {response.status_code}")
    return response.json() if response.text else None


def search_sparse_vector(
    vector_db_name, vector, top_k=10, early_terminate_threshold=0.0
):
    url = f"{base_url}/collections/{vector_db_name}/search/sparse"
    data = {
        "query_terms": vector["values"],
        "top_k": top_k,
        "early_terminate_threshold": early_terminate_threshold,
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )

    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to search vector: {response.status_code}")

    return response.json()


def batch_ann_search(vector_db_name, batch, top_k=10, early_terminate_threshold=0.0):
    url = f"{base_url}/collections/{vector_db_name}/search/batch-sparse"
    data = {
        "query_terms_list": batch,
        "top_k": top_k,
        "early_terminate_threshold": early_terminate_threshold,
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code not in [200, 204]:
        print(f"Error response ({response.status_code}): {response.text}")
    return response.json()


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
    """Compute dot product similarity between query vectors and all vectors"""
    if os.path.exists(BRUTE_FORCE_RESULTS_FILE):
        print(f"Loading existing brute force results from {BRUTE_FORCE_RESULTS_FILE}")
        with open(BRUTE_FORCE_RESULTS_FILE, "rb") as f:
            return pickle.load(f)

    print(
        f"Computing brute force dot product similarity for {len(query_vectors)} queries..."
    )
    results = []

    for query in tqdm(query_vectors):
        # Create a sparse row vector for the query
        query_indices = query["indices"]
        query_values = query["values"]

        dot_products = []
        for vec in vectors:
            # Find common indices between query and vector
            common_indices = list(set(query_indices) & set(vec["indices"]))

            # Compute dot product only for common indices
            dot_product = sum(
                query_values[query_indices.index(idx)]
                * vec["values"][vec["indices"].index(idx)]
                for idx in common_indices
            )

            dot_products.append(dot_product)

        # Get top-k results
        top_indices = heapq.nlargest(
            top_k, range(len(dot_products)), key=lambda i: dot_products[i]
        )

        # Filter out the query vector itself (if it's in the top results)
        top_results = [
            {"id": vectors[idx]["id"], "score": float(dot_products[idx])}
            for idx in top_indices
        ][:top_k]

        results.append({"query_id": query["id"], "top_results": top_results})

    # Save to disk
    with open(BRUTE_FORCE_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)

    print(f"Brute force results computed and saved to {BRUTE_FORCE_RESULTS_FILE}")
    return results


def format_for_server_query(vector):
    """Format a vector for server query"""
    return {
        "id": vector["id"],
        "values": [[ind, val] for ind, val in zip(vector["indices"], vector["values"])],
    }


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
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    return avg_recall, recalls


def run_rps_tests(
    rps_test_vectors,
    vector_db_name,
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
                format_for_server_query(vector)["values"]
                for vector in rps_test_vectors[i : i + batch_size]
            ]
            futures.append(
                executor.submit(
                    batch_ann_search,
                    vector_db_name,
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

    Path("sparse_dataset").mkdir(parents=True, exist_ok=True)

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

    if insert_vectors:
        # Create collection
        try:
            print(f"Creating collection: {vector_db_name}")
            create_db(name=vector_db_name, dimension=dimension)
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
                batch = vectors[batch_start : batch_start + batch_size]
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
            result = search_sparse_vector(
                vector_db_name, formatted_query, top_k, early_terminate_threshold
            )
            server_results.append(result)
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

    with open("sparse_dataset/vector_evaluation_results.pkl", "wb") as f:
        pickle.dump(detailed_results, f)

    print("Detailed results saved to sparse_dataset/vector_evaluation_results.pkl")

    run_rps_tests(
        vectors[:10_000], vector_db_name, batch_size, top_k, early_terminate_threshold
    )


if __name__ == "__main__":
    main()
