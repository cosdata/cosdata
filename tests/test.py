#!/usr/bin/env python

import requests
import json
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import random
import getpass
import os

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"


def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}


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


def create_db(name, description=None, dimension=1024):
    url = f"{base_url}/collections"
    data = {
        "name": name,
        "description": description,
        "dense_vector": {
            "enabled": True,
            "dimension": dimension,
        },
        "sparse_vector": {"enabled": False},
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
        "distance_metric_type": "cosine",  # Type of distance metric (e.g., cosine, euclidean)
        "quantization": {"type": "auto", "properties": {"sample_threshold": 100}},
        "index": {
            "type": "hnsw",
            "properties": {
                "num_layers": 7,
                "max_cache_size": 1000,
                "ef_construction": 512,
                "ef_search": 256,
                "neighbors_count": 32,
                "level_0_neighbors_count": 64,
            },
        },
    }
    response = requests.post(
        f"{base_url}/collections/{name}/indexes/dense",
        headers=generate_headers(),
        data=json.dumps(data),
        verify=False,
    )

    if response.status_code not in [200, 201, 204]:
        raise Exception(
            f"Failed to create index: {response.status_code} ({response.text})"
        )
    return response.json() if response.status_code == 200 and response.text else {}


# Function to find a database (collection) by Id
def find_collection(id):
    url = f"{base_url}/collections/{id}"

    response = requests.get(url, headers=generate_headers(), verify=False)
    return response.json()


def create_transaction(collection_name):
    url = f"{base_url}/collections/{collection_name}/transactions"
    response = requests.post(url, headers=generate_headers(), verify=False)
    if response.status_code != 200:
        print(f"Error creating transaction: {response.status_code} ({response.text})")
    return response.json()


def upsert_in_transaction(collection_name, transaction_id, vectors):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/upsert"
    )
    vectors = [
        {"id": vector["id"], "dense_values": vector["values"]} for vector in vectors
    ]
    data = {"vectors": vectors}
    print(f"Request URL: {url}")
    print(f"Request Vectors Count: {len(vectors)}")
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    print(f"Response Status: {response.status_code}")
    if response.status_code not in [200, 204]:
        raise Exception(
            f"Failed to upsert in transaction: {response.status_code} ({response.text})"
        )


def commit_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/commit"
    )
    response = requests.post(url, headers=generate_headers(), verify=False)
    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to commit transaction: {response.status_code}")
    return None


def abort_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/abort"
    )
    response = requests.post(url, headers=generate_headers(), verify=False)
    if response.status_code not in [200, 204]:
        print(f"Error aborting transaction: {response.status_code} ({response.text})")
    return None


def ann_vector(idd, vector_db_name, vector):
    url = f"{base_url}/collections/{vector_db_name}/search/dense"
    data = {"query_vector": vector, "top_k": 5}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to search vector: {response.status_code}")
    result = response.json()
    return (idd, result)


# Function to fetch vector
def fetch_vector(vector_db_name, vector_id):
    url = f"{base_url}/collections/{vector_db_name}/vectors/{vector_id}/neighbors"
    response = requests.get(url, headers=generate_headers(), verify=False)
    if response.status_code != 200:
        print(f"Error fetching neighbors: {response.status_code} ({response.text})")
        return []
    return response.json()


# Function to generate a random vector with given constraints
def generate_random_vector(rows, dimensions, min_val, max_val):
    return np.random.uniform(min_val, max_val, (rows, dimensions)).tolist()


def generate_random_vector_with_id(id, length):
    values = np.random.uniform(-1, 1, length).tolist()
    return {"id": str(id), "values": values}


def perturb_vector(vector, perturbation_degree):
    # Generate the perturbation
    perturbation = np.random.uniform(
        -perturbation_degree, perturbation_degree, len(vector["values"])
    )
    # Apply the perturbation and clamp the values within the range of -1 to 1
    perturbed_values = np.array(vector["values"]) + perturbation
    clamped_values = np.clip(perturbed_values, -1, 1)
    vector["values"] = clamped_values.tolist()
    return vector


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


def generate_perturbation(base_vector, idd, perturbation_degree, dimensions):
    # Generate the perturbation
    perturbation = np.random.uniform(
        -perturbation_degree, perturbation_degree, dimensions
    )

    # Apply the perturbation and clamp the values within the range of -1 to 1
    # perturbed_values = base_vector["values"] + perturbation
    perturbed_values = np.array(base_vector["values"]) + perturbation
    clamped_values = np.clip(perturbed_values, -1, 1)

    perturbed_vector = {"id": idd, "values": clamped_values.tolist()}
    # print(base_vector["values"][:10])
    # print( perturbed_vector["values"][:10] )
    # cs = cosine_similarity(base_vector["values"], perturbed_vector["values"] )
    # print ("cosine similarity of perturbed vec: ", row_ct, cs)
    return perturbed_vector
    # if np.random.rand() < 0.01:  # 1 in 100 probability
    #     shortlisted_vectors.append(perturbed_vector)


def process_base_vector_batch(
    req_ct, base_idx, vector_db_name, transaction_id, dimensions, perturbation_degree
):
    try:
        # Generate one base vector
        base_vector = generate_random_vector_with_id(
            req_ct * 10000 + base_idx * 100, dimensions
        )

        # Create batch containing base vector and its perturbations
        batch_vectors = [base_vector]  # Start with base vector

        # Generate 99 perturbations for this base vector
        for i in range(99):
            perturbed_vector = generate_perturbation(
                base_vector,
                req_ct * 10000
                + base_idx * 100
                + i
                + 1,  # Unique ID for each perturbation
                perturbation_degree,
                dimensions,
            )
            batch_vectors.append(perturbed_vector)

        # Submit this base vector and its perturbations as one batch
        upsert_in_transaction(vector_db_name, transaction_id, batch_vectors)
        print(
            f"Upsert complete for base vector {base_idx} and its {len(batch_vectors) - 1} perturbations"
        )

        return (
            base_idx,
            generate_perturbation(
                base_vector,
                req_ct * 10000
                + base_idx * 100
                + i
                + 1,  # Unique ID for each perturbation
                perturbation_degree,
                dimensions,
            ),
            batch_vectors,
        )
    except Exception as e:
        print(f"Error processing base vector {base_idx}: {e}")
        raise


def bruteforce_search(vectors, query, k=5):
    similarities = []
    for vector in vectors:
        similarity = cosine_similarity(query["values"], vector["values"])
        similarities.append((vector["id"], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


def generate_vectors(req_ct, batch_count, batch_size, dimensions, perturbation_degree):
    vectors = []

    for base_idx in range(batch_count):
        base_vector = generate_random_vector_with_id(
            (req_ct * batch_count * batch_size) + (base_idx * batch_size), dimensions
        )

        vectors.append(base_vector)

        for i in range(batch_size - 1):
            perturbed_vector = generate_perturbation(
                base_vector,
                (req_ct * batch_count * batch_size)
                + (base_idx * batch_size + i + 1),  # Unique ID for each perturbation
                perturbation_degree,
                dimensions,
            )
            vectors.append(perturbed_vector)

    # Shuffle the vectors
    np.random.shuffle(vectors)
    return vectors


def search(vectors, vector_db_name, query):
    ann_response = ann_vector(query["id"], vector_db_name, query["values"])
    bruteforce_result = bruteforce_search(vectors, query, 5)
    return (ann_response, bruteforce_result)


if __name__ == "__main__":
    # Create database
    vector_db_name = "testdb"
    dimensions = 1024
    max_val = 1.0
    min_val = -1.0
    perturbation_degree = 0.95  # Degree of perturbation
    batch_size = 100
    batch_count = 1000

    session_response = create_session()
    print("Session Response:", session_response)

    create_collection_response = create_db(
        name=vector_db_name,
        description="Test collection for vector database",
        dimension=dimensions,
    )
    print("Create Collection(DB) Response:", create_collection_response)
    create_explicit_index(vector_db_name)

    start_time = time.time()

    shortlisted_vectors = []

    for req_ct in range(1):
        transaction_id = None
        try:
            # Create a new transaction
            transaction_response = create_transaction(vector_db_name)
            transaction_id = transaction_response["transaction_id"]
            print(f"Created transaction: {transaction_id}")

            vectors = generate_vectors(
                req_ct, batch_count, batch_size, dimensions, perturbation_degree
            )

            # Process vectors concurrently
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for base_idx in range(batch_count):
                    futures.append(
                        executor.submit(
                            upsert_in_transaction,
                            vector_db_name,
                            transaction_id,
                            vectors[
                                base_idx * batch_size : (base_idx * batch_size)
                                + batch_size
                            ],
                        )
                    )
                    if random.random() < 0.9:
                        continue

                    shortlisted_vectors.append(
                        generate_perturbation(
                            vectors[base_idx * batch_size],
                            base_idx,
                            perturbation_degree,
                            dimensions,
                        )
                    )

                # Collect results
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in future: {e}")

            # Commit the transaction after all vectors are inserted
            commit_response = commit_transaction(vector_db_name, transaction_id)
            print(f"Committed transaction {transaction_id}: {commit_response}")
            transaction_id = None

        except Exception as e:
            print(f"Error in transaction: {e}")
            if transaction_id:
                try:
                    abort_transaction(vector_db_name, transaction_id)
                    print(f"Aborted transaction {transaction_id} due to error")
                except Exception as abort_error:
                    print(f"Error aborting transaction: {abort_error}")

        # End time
    end_time = time.time()
    # Search vector concurrently using perturbed vectors
    best_matches_server = []
    best_matches_bruteforce = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for query in shortlisted_vectors:
            futures.append(executor.submit(search, vectors, vector_db_name, query))

        for i, future in enumerate(as_completed(futures)):
            try:
                ((idr, ann_response), (bruteforce_results)) = future.result()
                # Check if the new 'results' key exists and is a list
                if "results" in ann_response and isinstance(
                    ann_response["results"], list
                ):
                    print("  Server:")
                    for j, match in enumerate(ann_response["results"][:5]):
                        id = match["id"]
                        score = match["score"]
                        print(f"    {j + 1}: {id} ({score})")
                    if ann_response["results"]:
                        best_matches_server.append(
                            ann_response["results"][0].get("score", 0.0)
                        )

                    print("  Brute force:")
                    for j, result in enumerate(bruteforce_results):
                        cs = result[1]
                        id = result[0]
                        print(f"    {j + 1}: {id} ({cs})")
                    best_matches_bruteforce.append(bruteforce_results[0][1])

            except Exception as e:
                print(f"Error in ANN vector {i + 1}: {e}")

    if best_matches_server:
        best_match_server_average = sum(best_matches_server) / len(best_matches_server)
        best_match_bruteforce_average = sum(best_matches_bruteforce) / len(
            best_matches_bruteforce
        )
        print(f"\n\nBest Match Server Average: {best_match_server_average}")
        print(f"Best Match Brute force Average: {best_match_bruteforce_average}")
    else:
        print("No valid matches found.")

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
