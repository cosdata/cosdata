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

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = "https://127.0.0.1:8443"
base_url = f"{host}/vectordb"


def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}


# Function to login with credentials
def login():
    url = f"{host}/auth/login"
    data = {"username": "admin", "password": "admin", "pretty_print": False}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    global token
    token = response.text
    return token


# Function to create sparse vectors in an inverted index
def create_sparse_vector(vector_db_name, records):
    url = f"{base_url}/collections/{vector_db_name}/vectors"
    data = {
        "dense": records,
    }
    response = requests.post(url, headers=generate_headers(), data=json.dumps(data), verify=False)
    return response.json()

# Function to query sparse vectors inverted index
# def query_sparse_index(vector_db_name, sparse_query, limit):
#     url = f"{base_url}/query_sparse_index"
#     data = {
#         "vector_db_name": vector_db_name,
#         "sparse_query": sparse_query,
#         "limit": limit
#     }
#     response = requests.post(url, headers=generate_headers(), data=json.dumps(data), verify=False)
#     return response.json()

# Updated function to generate multiple random sparse vectors
def generate_random_sparse_vectors(num_records, max_index, min_nonzero, max_nonzero):
    records = []
    for i in range(num_records):
        num_nonzero = np.random.randint(min_nonzero, max_nonzero + 1)
        indices = sorted(np.random.choice(max_index, num_nonzero, replace=False))
        values = np.random.uniform(0, 1, num_nonzero).tolist()
        record = {
            'id': f'vec{i}',
                'indices': list(map(int, indices)),
                'values': values
        }
        records.append(record)
    return records

# Function to perturb a sparse vector
def perturb_sparse_vector(vector, perturbation_degree, max_index):
    indices = set(vector['indices'])
    values = vector['values']
    
    # Perturb existing values
    perturbed_values = [max(0, min(1, v + np.random.uniform(-perturbation_degree, perturbation_degree))) for v in values]
    
    # Add or remove some indices
    num_changes = np.random.randint(0, 3)
    for _ in range(num_changes):
        if np.random.random() < 0.5 and len(indices) > 1:
            # Remove an index
            remove_idx = np.random.choice(len(indices))
            indices.remove(list(indices)[remove_idx])
            del perturbed_values[remove_idx]
        else:
            # Add a new index
            new_index = np.random.randint(0, max_index)
            if new_index not in indices:
                indices.add(new_index)
                perturbed_values.append(np.random.uniform(0, 1))
    
    return {
        'id': vector['id'],
            'indices': sorted(list(indices)),
            'values': perturbed_values
    }

if __name__ == "__main__":
    vector_db_name = "sparse_testdb"
    max_index = 1000
    num_vectors = 100
    min_nonzero = 5
    max_nonzero = 15
    perturbation_degree = 0.25

    # first login to get the auth jwt token
    login_response = login()
    print("Login Response:", login_response)

    # Generate random sparse vectors
    records = generate_random_sparse_vectors(num_vectors, max_index, min_nonzero, max_nonzero)

    # Create sparse index
    # create_response = create_sparse_index(vector_db_name, records)
    create_response = create_sparse_vector(vector_db_name, records[0])
    print("Create Sparse Vector Response:", create_response)

    # Generate a query vector (perturbed version of a random vector)
    query_base = records[np.random.randint(0, num_vectors)]
    query_vector = perturb_sparse_vector(query_base, perturbation_degree, max_index)

    # # Query sparse index
    # query_response = query_sparse_index(
    #     vector_db_name,
    #     query_vector['sparse_values'],
    #     limit=10
    # )
    # print("Query Sparse Index Response:", query_response)

    # You can add more testing, concurrent operations, or performance measurements here

# Example of how the records being inserted would look
sample_records = generate_random_sparse_vectors(3, max_index, min_nonzero, max_nonzero)
print("\nExample of records being inserted:")
print(json.dumps(sample_records, indent=2))
