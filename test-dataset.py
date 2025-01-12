import requests
import json
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import os
import math
import random

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"


def load_brute_force_results(dataset_name):
    csv_path = f"{dataset_name}-brute_force_results.csv"

    try:
        print("Attempting to load pre-computed brute force results...")
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded results from {csv_path}")
        return df.to_dict("records")
    except FileNotFoundError:
        return False


def generate_brute_force_results(dataset_name, vector_list):
    """Load brute force results from CSV, or generate them if file doesn't exist"""
    csv_path = f"{dataset_name}-brute_force_results.csv"

    vectors_corrected = vector_list
    total_vectors = len(vectors_corrected)
    print(f"Total vectors for brute force computation: {total_vectors}")

    # Randomly select 100 fixed test vectors
    np.random.seed(42)  # Fix seed for reproducibility
    test_indices = np.random.choice(total_vectors, 100, replace=False)
    test_vectors = [vectors_corrected[i] for i in test_indices]

    print("Computing brute force similarities...")
    results = []

    for i, query in enumerate(test_vectors):
        if i % 10 == 0:
            print(f"Processing query vector {i+1}/100, ID: {query['id']}")

        similarities = []
        for vector in vectors_corrected:
            sim = cosine_similarity(query["values"], vector["values"])
            similarities.append((vector["id"], sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top5 = similarities[:5]

        results.append(
            {
                "query_id": query["id"],
                "top1_id": top5[0][0],
                "top1_sim": top5[0][1],
                "top2_id": top5[1][0],
                "top2_sim": top5[1][1],
                "top3_id": top5[2][0],
                "top3_sim": top5[2][1],
                "top4_id": top5[3][0],
                "top4_sim": top5[3][1],
                "top5_id": top5[4][0],
                "top5_sim": top5[4][1],
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nGenerated and saved results to {csv_path}")
    return results


def load_or_generate_brute_force_results(dataset_name):
    results = load_brute_force_results(dataset_name)
    if results:
        return results
    vectors = read_dataset_from_parquet(dataset_name)
    results = generate_brute_force_results(dataset_name, vectors)
    del vectors
    return results


def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}


def create_session():
    url = f"{host}/auth/create-session"
    data = {"username": "admin", "password": "admin"}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    session = response.json()
    global token
    token = session["access_token"]
    return token


def create_db(name, description=None, dimension=1024):
    url = f"{base_url}/collections"
    data = {
        "name": name,
        "description": description,
        "dense_vector": {
            "enabled": True,
            "auto_create_index": False,
            "dimension": dimension,
        },
        "sparse_vector": {"enabled": False, "auto_create_index": False},
        "metadata_schema": None,
        "config": {"max_vectors": None, "replication_factor": None},
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def create_explicit_index(name):
    data = {
        "collection_name": name,  # Name of the collection
        "name": name,  # Name of the index
        "distance_metric_type": "cosine",  # Type of distance metric (e.g., cosine, euclidean)
        "quantization": {
            # "type": "scalar",
            # "properties": {
            #     "data_type": "binary",
            #     "range": {
            #         "min": -1.0,
            #         "max": 1.0,
            #     },
            # },
            "type": "auto",
            "properties": {
                "sample_threshold": 100
            }
        },
        "index": {
            "type": "hnsw",
            "properties": {
                "num_layers": 7,
                "max_cache_size": 1000,
                "ef_construction": 512,
                "ef_search": 256,
                "neighbors_count": 32,
                "layer_0_neighbors_count": 64,
            },
        },
    }
    response = requests.post(
        f"{base_url}/indexes",
        headers=generate_headers(),
        data=json.dumps(data),
        verify=False,
    )

    return response.json()


def create_transaction(collection_name):
    url = f"{base_url}/collections/{collection_name}/transactions"
    response = requests.post(url, headers=generate_headers(), verify=False)
    return response.json()


def upsert_in_transaction(collection_name, transaction_id, vectors):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/upsert"
    )
    data = {"vectors": vectors}
    print(f"Request URL: {url}")
    print(f"Request Vectors Count: {len(vectors)}")
    response = requests.post(
        url,
        headers=generate_headers(),
        data=json.dumps(data),
        verify=False,
        timeout=10000,
    )
    print(f"Response Status: {response.status_code}")
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to create vector: {response.status_code}")


def commit_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/commit"
    )
    response = requests.post(url, headers=generate_headers(), verify=False)
    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to commit transaction: {response.status_code}")
    return response.json() if response.text else None


def abort_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/abort"
    )
    response = requests.post(url, headers=generate_headers(), verify=False)
    return response.json()


def ann_vector(idd, vector_db_name, vector):
    url = f"{base_url}/search"
    data = {"vector_db_name": vector_db_name, "vector": vector, "nn_count": 5}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to search vector: {response.status_code}")
    result = response.json()

    # Handle empty results gracefully
    if not result.get("RespVectorKNN", {}).get("knn"):
        return (idd, {"RespVectorKNN": {"knn": []}})
    return (idd, result)


def dot_product(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))


def magnitude(vec):
    return np.sqrt(sum(v**2 for v in vec))


def cosine_similarity(vec1, vec2):
    dot_prod = dot_product(vec1, vec2)
    magnitude_vec1 = magnitude(vec1)
    magnitude_vec2 = magnitude(vec2)

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0  # Handle the case where one or both vectors are zero vectors

    return dot_prod / (magnitude_vec1 * magnitude_vec2)


def cosine_similarity(vec1, vec2):
    # Convert inputs to numpy arrays
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    # Check if vectors have the same length
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same length")

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Check for zero vectors
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Cannot compute cosine similarity for zero vectors")

    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)

    return cosine_sim


def bruteforce_search(vectors, query, k=5):
    """Debug version of bruteforce search"""
    similarities = []
    print(f"\nSearching for query vector ID: {query['id']}")
    query_array = np.array(query["values"])
    print(f"Query vector first 5 values: {query_array[:5]}")

    # First check similarity with itself
    self_similarity = cosine_similarity(query["values"], query["values"])
    print(f"Self similarity check for ID {query['id']}: {self_similarity}")

    for vector in vectors:
        similarity = cosine_similarity(query["values"], vector["values"])
        if similarity > 0.999:  # Check for very high similarities
            print(f"High similarity found:")
            print(f"Vector ID: {vector['id']}")
            print(f"Similarity: {similarity}")
            print(f"First 5 values of this vector: {vector['values'][:5]}")
            print(f"First 5 values of query vector: {query['values'][:5]}")

        similarities.append((vector["id"], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]
    print(f"\nTop {k} matches for query {query['id']}:")
    for id, sim in top_k:
        print(f"ID: {id}, Similarity: {sim}")

    return top_k


def batch_ann_search(vector_db_name, vectors):
    url = f"{base_url}/batch-search"
    data = {"vector_db_name": vector_db_name, "vectors": vectors, "nn_count": 5}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to search vector: {response.status_code}")
    result = response.json()
    return result


datasets = {
    "cohere-wiki-embedding-100k": {
        "id": "id",
        "embeddings": "emb",
        "size": 100_000,
    },
    "million-text-embeddings": {
        "id": None,
        "embeddings": "embedding",
        "size": 1_000_000,
        "dimension": 768,
    },
    "job_listing_embeddings": {
        "id": None,
        "embeddings": "embedding",
        "size": 1_000_000,
    },
    "arxiv-embeddings-ada-002": {
        "id": None,
        "embeddings": "embeddings",
        "size": 1_000_000,
    },
    "dbpedia-entities-openai-1M": {
        "id": None,
        "embeddings": "openai",
        "size": 1_000_000,
        "dimension": 1536,
    },
}


def prompt_and_get_dataset_metadata():
    print("\nChoose a dataset to test with:")
    dataset_names = list(datasets.keys())
    for i, key in enumerate(dataset_names):
        print(f"{i + 1}) {key}")
    print()
    dataset_idx = int(input("Select: ")) - 1
    dataset_name = dataset_names[dataset_idx]
    print(f"Reading {dataset_name} ...")
    dataset = datasets[dataset_name]
    return (dataset_name, dataset)


def pre_process_vector(id, values):
    corrected_values = [float(v) for v in values]
    vector = {"id": int(id), "values": corrected_values}
    return vector


def read_dataset_from_parquet(dataset_name):
    metadata = datasets[dataset_name]
    dfs = []

    path = f"{dataset_name}/test0.parquet"

    while os.path.exists(path):
        dfs.append(pd.read_parquet(path))
        count = len(dfs)
        path = f"{dataset_name}/test{count}.parquet"

    df = pd.concat(dfs, ignore_index=True)

    print("Pre-processing ...")
    size = metadata["size"]
    dataset = (
        df[[metadata["id"], metadata["embeddings"]]].values.tolist()
        if metadata["id"] != None
        else list(enumerate(row[0] for row in df[[metadata["embeddings"]]].values))
    )

    vectors = []
    print("Dimension:", len(dataset[0][1]))
    print("Size: ", len(dataset))

    for row in dataset:
        vector = pre_process_vector(row[0], row[1])
        vectors.append(vector)

    return vectors


def compare_vectors_in_detail(vectors_corrected, id1, id2):
    """Compare two vectors in detail"""
    vec1 = next(v for v in vectors_corrected if v["id"] == id1)
    vec2 = next(v for v in vectors_corrected if v["id"] == id2)

    values1 = np.array(vec1["values"])
    values2 = np.array(vec2["values"])

    # Compare raw values
    is_identical = np.allclose(values1, values2, rtol=1e-15)
    cosine_sim = np.dot(values1, values2) / (
        np.linalg.norm(values1) * np.linalg.norm(values2)
    )

    # Value comparisons
    diff = values1 - values2
    max_diff = np.max(np.abs(diff))
    avg_diff = np.mean(np.abs(diff))

    print(f"\nDetailed Vector Comparison for IDs {id1} and {id2}:")
    print(f"Identical vectors: {is_identical}")
    print(f"Cosine similarity: {cosine_sim}")
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Average absolute difference: {avg_diff}")
    print("First 5 values of each vector:")
    print(f"Vector {id1}: {values1[:5]}")
    print(f"Vector {id2}: {values2[:5]}")
    return is_identical


def read_single_parquet_file(path, dataset_name, file_index, base_id):
    """Read and process a single parquet file"""
    try:
        print(f"Reading file {file_index}: {path}")
        df = pd.read_parquet(path)

        dataset_config = datasets[dataset_name]
        id_col = dataset_config["id"]
        emb_col = dataset_config["embeddings"]

        if id_col is not None:
            dataset = df[[id_col, emb_col]].values.tolist()
        else:
            dataset = list(
                enumerate((row[0] for row in df[[emb_col]].values), start=base_id)
            )

        vectors = []
        for row in dataset:
            vector = pre_process_vector(row[0], row[1])
            vectors.append(vector)

        print(f"Processed {len(vectors)} vectors from file {file_index}")
        return vectors

    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return None


def process_parquet_files(
    dataset_name,
    vector_db_name,
    brute_force_results,
    batch_size=100,
    max_vectors_per_transaction=250000,
    matches_sample_size=100,
    rps_sample_size=100000,
):
    """
    Process parquet files asynchronously and upsert vectors to the server.
    Collects random samples of vectors for testing purposes.
    """
    file_count = 0
    total_vectors_inserted = 0
    total_insertion_time = 0
    id_counter = 0
    matches_test_vectors = []
    rps_test_vectors = []
    matches_test_vector_ids_set = {result["query_id"] for result in brute_force_results}

    # Transaction state
    current_transaction_id = None
    vectors_in_current_transaction = 0

    def get_next_file_path(count):
        return f"{dataset_name}/test{count}.parquet"

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=3) as executor:
        current_path = get_next_file_path(file_count)
        if not os.path.exists(current_path):
            return matches_test_vectors, rps_test_vectors

        future = executor.submit(
            read_single_parquet_file,
            current_path,
            dataset_name,
            file_count,
            id_counter,
        )

        try:
            while True:
                try:
                    vectors = future.result()
                    id_counter += len(vectors)
                except Exception as e:
                    print(f"Error reading file {current_path}: {e}")
                    break

                if not vectors:
                    break

                file_count += 1
                next_path = get_next_file_path(file_count)
                if os.path.exists(next_path):
                    future = executor.submit(
                        read_single_parquet_file,
                        next_path,
                        dataset_name,
                        file_count,
                        id_counter,
                    )

                if len(matches_test_vectors) < matches_sample_size:
                    for vector in vectors:
                        if vector["id"] in matches_test_vector_ids_set:
                            matches_test_vectors.append(vector)

                if len(rps_test_vectors) < rps_sample_size:
                    sample_size = min(
                        rps_sample_size - len(rps_test_vectors), len(vectors)
                    )
                    rps_test_vectors.extend(random.sample(vectors, sample_size))

                insertion_start = time.time()
                (
                    vectors_inserted,
                    current_transaction_id,
                    vectors_in_current_transaction,
                ) = process_vectors_batch(
                    vectors,
                    vector_db_name,
                    batch_size,
                    max_vectors_per_transaction,
                    current_transaction_id,
                    vectors_in_current_transaction,
                )
                insertion_end = time.time()
                insertion_time = insertion_end - insertion_start
                total_insertion_time += insertion_time
                total_vectors_inserted += vectors_inserted

                print(f"\nProcessing file: {current_path}")
                print(f"File {file_count-1} statistics:")
                print(f"Vectors inserted: {vectors_inserted}")
                print(
                    f"Matches test vectors collected: {len(matches_test_vectors)}/{matches_sample_size}"
                )
                print(
                    f"RPS test vectors collected: {len(rps_test_vectors)}/{rps_sample_size}"
                )

                del vectors
                current_path = next_path

                if not os.path.exists(next_path):
                    break

        finally:
            # Commit final transaction if any vectors remain
            if current_transaction_id is not None:
                try:
                    commit_response = commit_transaction(
                        vector_db_name, current_transaction_id
                    )
                    print(
                        f"Committed final transaction {current_transaction_id}: {commit_response}"
                    )
                except Exception as e:
                    print(f"Error committing final transaction: {e}")
                    try:
                        abort_transaction(vector_db_name, current_transaction_id)
                    except Exception as abort_error:
                        print(f"Error aborting transaction: {abort_error}")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nProcessing complete!")
    print(f"Total files processed: {file_count}")
    print(f"Total vectors inserted: {total_vectors_inserted}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total insertion time: {total_insertion_time:.2f} seconds")
    print(
        f"Average insertion time per vector: {(total_insertion_time/total_vectors_inserted)*1000:.2f} ms"
    )
    print(f"Final matches test vectors collected: {len(matches_test_vectors)}")
    print(f"Final RPS test vectors collected: {len(rps_test_vectors)}")

    return matches_test_vectors, rps_test_vectors


def process_vectors_batch(
    vectors,
    vector_db_name,
    batch_size,
    max_vectors_per_transaction,
    current_transaction_id,
    vectors_in_current_transaction,
):
    """Process a batch of vectors and insert them into the database"""
    total_vectors = len(vectors)
    vectors_inserted = 0

    try:
        total_batches = (total_vectors + batch_size - 1) // batch_size

        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []

            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_vectors)
                current_batch = vectors[batch_start:batch_end]

                # Create new transaction if needed
                if current_transaction_id is None:
                    transaction_response = create_transaction(vector_db_name)
                    current_transaction_id = transaction_response["transaction_id"]
                    vectors_in_current_transaction = 0

                vectors_inserted += len(current_batch)
                vectors_in_current_transaction += len(current_batch)

                futures.append(
                    executor.submit(
                        upsert_in_transaction,
                        vector_db_name,
                        current_transaction_id,
                        current_batch,
                    )
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in future: {e}")
                    raise e

            # Commit transaction if we've reached the limit
            if vectors_in_current_transaction >= max_vectors_per_transaction:
                commit_response = commit_transaction(
                    vector_db_name, current_transaction_id
                )
                print(
                    f"Committed transaction {current_transaction_id}: {commit_response}"
                )
                current_transaction_id = None
                vectors_in_current_transaction = 0

    except Exception as e:
        print(f"Error in batch processing: {e}")
        if current_transaction_id:
            try:
                abort_transaction(vector_db_name, current_transaction_id)
                print(f"Aborted transaction {current_transaction_id}")
            except Exception as abort_error:
                print(f"Error aborting transaction: {abort_error}")
            current_transaction_id = None
            vectors_in_current_transaction = 0
        raise e

    return vectors_inserted, current_transaction_id, vectors_in_current_transaction


def run_matching_tests(test_vectors, vector_db_name, brute_force_results):
    """Run matching accuracy tests"""
    print("\nStarting similarity search tests...")

    total_recall = 0
    total_queries = 0

    for i, test_vec in enumerate(brute_force_results):
        try:
            query_vec = next(v for v in test_vectors if v["id"] == test_vec["query_id"])
            idr, ann_response = ann_vector(
                query_vec["id"], vector_db_name, query_vec["values"]
            )

            if (
                "RespVectorKNN" in ann_response
                and "knn" in ann_response["RespVectorKNN"]
            ):
                query_id = test_vec["query_id"]
                print(f"\nQuery {i + 1} (Vector ID: {query_id}):")

                server_top5 = [
                    match[0] for match in ann_response["RespVectorKNN"]["knn"][:5]
                ]
                print("Server top 5:", server_top5)

                brute_force_top5 = [test_vec[f"top{j}_id"] for j in range(1, 6)]
                print("Brute force top 5:", brute_force_top5)

                matches = sum(1 for id in server_top5 if id in brute_force_top5)
                recall = (matches / 5) * 100
                total_recall += recall
                total_queries += 1

                print(f"Recall@5 for this query: {recall}% ({matches}/5 matches)")

                time.sleep(0.1)

        except Exception as e:
            print(f"Error in query {i + 1}: {e}")

    if total_queries > 0:
        average_recall = total_recall / total_queries
        print(f"\nFinal Matching Results:")
        print(f"Average Recall@5: {average_recall:.2f}%")
    else:
        print("No valid queries completed")


def run_rps_tests(rps_test_vectors, vector_db_name, batch_size=100):
    """Run RPS (Requests Per Second) tests"""
    print(f"Using {len(rps_test_vectors)} different test vectors for RPS testing")

    start_time_rps = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(rps_test_vectors), batch_size):
            batch = [
                vector["values"] for vector in rps_test_vectors[i : i + batch_size]
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

    print("\nRPS Test Results:")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_requests}")
    print(f"Test Duration: {actual_duration:.2f} seconds")
    print(f"Requests Per Second (RPS): {rps:.2f}")
    print(f"Success Rate: {(successful_requests/total_requests*100):.2f}%")


if __name__ == "__main__":
    # Create database
    vector_db_name = "testdb"
    batch_size = 100
    max_vectors_per_transaction = 250_000
    num_match_test_vectors = 100  # Number of vectors to test
    num_rps_test_vectors = 100_000
    rps_batch_size = 100

    dataset_name, dataset_metadata = prompt_and_get_dataset_metadata()

    # Load or generate brute force results
    brute_force_results = load_or_generate_brute_force_results(dataset_name)
    print(f"Loaded {len(brute_force_results)} pre-computed brute force results")

    session_response = create_session()
    print("Session Response:", session_response)

    create_collection_response = create_db(
        name=vector_db_name,
        description="Test collection for vector database",
        dimension=dataset_metadata["dimension"],
    )
    print("Create Collection(DB) Response:", create_collection_response)
    create_explicit_index(vector_db_name)

    matches_test_vectors, rps_test_vectors = process_parquet_files(
        dataset_name,
        vector_db_name,
        brute_force_results,
        batch_size,
        max_vectors_per_transaction,
        num_match_test_vectors,
        num_rps_test_vectors,
    )

    run_matching_tests(matches_test_vectors, vector_db_name, brute_force_results)

    run_rps_tests(rps_test_vectors, vector_db_name, rps_batch_size)
