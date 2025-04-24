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
        "name": name,
        "distance_metric_type": "cosine",
        "quantization": {"type": "auto", "properties": {"sample_threshold": 100}},
        "index": {
            "type": "hnsw",
            "properties": {
                "num_layers": 5,
                "max_cache_size": 1000,
                "ef_construction": 64,
                "ef_search": 32,
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

    if response.status_code not in [200, 201, 204]:  # Allow 201
        raise Exception(
            f"Failed to create index: {response.status_code} ({response.text})"
        )
    return response.json() if response.status_code != 204 and response.text else {}


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


def create_vector_in_transaction(collection_name, transaction_id, vector):
    url = f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/vectors"
    data = {"id": vector["id"], "dense_values": vector["values"], "metadata": {}}
    print(f"Request URL: {url}")
    print(f"Request Data: {json.dumps(data)}")
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    print(f"Response Status: {response.status_code}")
    print(f"Response Text: {response.text}")
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to create vector: {response.status_code}")
    return response.json() if response.text else None


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
        raise Exception(f"Failed to create vector: {response.status_code}")


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
    return response.json() if response.status_code == 200 and response.text else None


def batch_ann_search(vector_db_name, vectors):
    url = f"{base_url}/collections/{vector_db_name}/search/batch-dense"
    data = {"queries": [{"vector": vector} for vector in vectors], "top_k": 5}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to batch search vector: {response.status_code}")
    result = response.json()
    return result


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


def bruteforce_search(vectors, query, k=5):
    similarities = []
    for vector in vectors:
        similarity = cosine_similarity(query["values"], vector["values"])
        similarities.append((vector["id"], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


def generate_vectors(
    txn_count, batch_count, batch_size, dimensions, perturbation_degree, id_offset=0
):
    vectors = [
        generate_random_vector_with_id(id + id_offset, dimensions)
        for id in range(txn_count * batch_count * batch_size)
    ]

    # Shuffle the vectors
    np.random.shuffle(vectors)
    return vectors


if __name__ == "__main__":
    # Create database
    vector_db_name = "testdb"
    dimensions = 1024
    max_val = 1.0
    min_val = -1.0
    perturbation_degree = 0.25  # Degree of perturbation
    # ~500k vectors
    batch_size = 256
    batch_count = 977
    txn_count = 2
    # 100k query vectors
    query_batch_count = 500
    query_batch_size = 200

    session_response = create_session()
    print("Session Response:", session_response)

    create_collection_response = create_db(
        name=vector_db_name,
        description="Test collection for vector database",
        dimension=dimensions,
    )
    print("Create Collection(DB) Response:", create_collection_response)
    create_explicit_index(vector_db_name)

    vectors = generate_vectors(
        txn_count, batch_count, batch_size, dimensions, perturbation_degree
    )

    start_time = time.time()

    for req_ct in range(txn_count):
        transaction_id = None
        try:
            # Create a new transaction
            transaction_response = create_transaction(vector_db_name)
            transaction_id = transaction_response["transaction_id"]
            print(f"Created transaction: {transaction_id}")

            # Process vectors concurrently
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = []
                for base_idx in range(batch_count):
                    req_start = req_ct * batch_count * batch_size
                    batch_start = req_start + base_idx * batch_size
                    batch = vectors[batch_start : batch_start + batch_size]
                    futures.append(
                        executor.submit(
                            upsert_in_transaction, vector_db_name, transaction_id, batch
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
            # time.sleep(10)

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

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

    query_vectors = random.sample(vectors, query_batch_count * query_batch_size)

    for i in range(len(query_vectors)):
        query_vectors[i] = perturb_vector(query_vectors[i], perturbation_degree)[
            "values"
        ]

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for batch_idx in range(query_batch_count):
            batch = query_vectors[
                batch_idx * query_batch_size : batch_idx * query_batch_size
                + query_batch_size
            ]
            futures.append(executor.submit(batch_ann_search, vector_db_name, batch))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in search: {e}")

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("RPS:", len(query_vectors) / elapsed_time)
    print(f"Elapsed time: {elapsed_time} seconds")
