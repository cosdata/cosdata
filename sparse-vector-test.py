"""
Sparse Vector Database Test Script

This script demonstrates and tests the functionality of a sparse vector database system.
It includes operations for creating a sparse vector index, inserting sparse vectors,
and querying the index for similar vectors.

Key Features:
- Generation of random sparse vectors with varying degrees of sparsity
- Creation of a sparse vector index using a RESTful API
- Perturbation of sparse vectors to simulate real-world query scenarios
- Querying the sparse vector index and retrieving similar vectors

The script uses a sparse vector representation where each vector is defined by its
non-zero indices and corresponding values. This approach is memory-efficient for
high-dimensional data where most elements are zero. The perturbation logic
simulates realistic query scenarios by slightly modifying existing vectors,
allowing for robust testing of the similarity search capabilities.

Usage:
- Ensure the vector database API is running and accessible
- Set the appropriate base_url and authentication token
- Run the script to perform a series of operations including index creation,
  vector insertion, and similarity querying
- The script outputs the responses from these operations for verification and testing

Note: This script is intended for testing and demonstration purposes. In a production
environment, additional error handling, security measures, and optimizations would be necessary.
"""

import requests
import json
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import random

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"


def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}


# Function to login with credentials
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
        "name": name,  # Name of the index
        "quantization": 16,
    }
    response = requests.post(
        f"{base_url}/collections/{name}/indexes/sparse",
        headers=generate_headers(),
        data=json.dumps(data),
        verify=False,
    )

    return response.json()


# Function to create sparse vectors in an inverted index
def create_sparse_vector(vector_db_name, vector):
    url = f"{base_url}/collections/{vector_db_name}/vectors"
    data = {"sparse": vector}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def upsert_in_transaction(vector_db_name, transaction_id, vectors):
    url = (
        f"{base_url}/collections/{vector_db_name}/transactions/{transaction_id}/upsert"
    )
    data = {"index_type": "sparse", "vectors": vectors}
    print(f"Request URL: {url}")
    print(f"Request Vectors Count: {len(vectors)}")
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    print(f"Response Status: {response.status_code}")
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to create vector: {response.status_code}")


def search_sparse_vector(vector_db_name, vector):
    url = f"{base_url}/collections/{vector_db_name}/vectors/search"
    data = {"sparse": vector}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def generate_random_sparse_vector(id, dimension, non_zero_dims):
    return {
        "id": id,
        "indices": random.sample(range(dimension), non_zero_dims),
        "values": np.random.uniform(0.1, 1.0, non_zero_dims).tolist(),
    }


def create_transaction(collection_name):
    url = f"{base_url}/collections/{collection_name}/transactions"
    data = {"index_type": "sparse"}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def commit_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/commit"
    )
    data = {"index_type": "sparse"}
    response = requests.post(
        url, data=json.dumps(data), headers=generate_headers(), verify=False
    )
    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to commit transaction: {response.status_code}")
    return response.json() if response.text else None


def abort_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/abort"
    )
    data = {"index_type": "sparse"}
    response = requests.post(
        url, data=json.dumps(data), headers=generate_headers(), verify=False
    )
    return response.json()


if __name__ == "__main__":
    vector_db_name = "sparse_testdb"
    max_index = 1000
    num_vectors = 100_000
    perturbation_degree = 0.25
    non_zero_dims = 100
    dimension = 20_000
    batch_size = 100

    # first login to get the access token
    session_response = create_session()
    print("Session Response:", session_response)
    create_collection_response = create_db(
        name=vector_db_name,
    )
    print("Create Collection(DB) Response:", create_collection_response)
    create_explicit_index(vector_db_name)

    print("Generating vectors")
    vectors = [
        generate_random_sparse_vector(id, dimension, non_zero_dims)
        for id in range(num_vectors)
    ]
    print("Creating transaction")
    transaction_id = create_transaction(vector_db_name)["transaction_id"]
    print("Inserting vectors")
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
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
                print(f"Request {i + 1} completed successfully")
            except Exception as e:
                print(f"Request {i + 1} failed: {e}")
    print("Committing transaction")
    commit_transaction(vector_db_name, transaction_id)
    end = time.time()
    insertion_time = end - start

    print(f"Insertion time: {insertion_time} seconds")
    search_vector = random.choice(vectors)
    query = {
        "id": num_vectors,
        "values": [
            [ind, search_vector["values"][i]]
            for i, ind in enumerate(search_vector["indices"])
        ],
    }
    print("Searching vector:", search_vector["id"])
    start = time.time()
    search_res = search_sparse_vector(vector_db_name, query)
    end = time.time()
    search_time = end - start
    print(f"Search time: {search_time} seconds")
    print(f"Search response:")
    for i, result in enumerate(search_res["Sparse"]):
        id = result["vector_id"]
        sim = result["similarity"]
        print(f"{i + 1}. {id} ({sim})")
