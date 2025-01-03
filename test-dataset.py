import requests
import json
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import os
import math

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"

# Add these checks after loading vectors:
def verify_vector_uniqueness(vectors):
    vector_map = {}
    for vec in vectors:
        vec_tuple = tuple(vec['values'])
        if vec_tuple in vector_map:
            print(f"Found duplicate vector: IDs {vec['id']} and {vector_map[vec_tuple]}")
        vector_map[vec_tuple] = vec['id']

# Also verify the specific vectors in question:
def verify_specific_vectors(vectors, id1, id2):
    vec1 = next((v for v in vectors if v['id'] == id1), None)
    vec2 = next((v for v in vectors if v['id'] == id2), None)
    if vec1 and vec2:
        sim = cosine_similarity(vec1['values'], vec2['values'])
        print(f"Similarity between {id1} and {id2}: {sim}")

def login():
    url = f"{host}/auth/login"
    data = {"username": "admin", "password": "admin", "pretty_print": False}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    global token
    token = response.text
    return token

def load_or_generate_brute_force_results():
    """Load brute force results from CSV, or generate them if file doesn't exist"""
    csv_path = 'brute_force_results.csv'
    
    try:
        print("Attempting to load pre-computed brute force results...")
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded results from {csv_path}")
        return df.to_dict('records')
    except FileNotFoundError:
        print(f"{csv_path} not found, generating new brute force results...")
        
        # Load and correct vectors
        # vectors_original = read_dataset_from_parquet()
        # vectors_corrected = []
        # for vec in vectors_original:
        #     corrected_values = [max(-0.9878, min(float(v), 0.987890)) for v in vec['values']]
        #     vectors_corrected.append({
        #         "id": vec['id'],
        #         "values": corrected_values
        #     })
        vectors_corrected = read_dataset_from_parquet()
        
        total_vectors = len(vectors_corrected)
        print(f"Total vectors read from parquet files: {total_vectors}")

        # Randomly select 100 fixed test vectors
        np.random.seed(42)  # Fix seed for reproducibility
        test_indices = np.random.choice(total_vectors, 100, replace=False)
        test_vectors = [vectors_corrected[i] for i in test_indices]
        
        print("Computing brute force similarities...")
        results = []
        
        # Process one vector at a time
        for i, query in enumerate(test_vectors):
            if i % 10 == 0:
                print(f"Processing query vector {i+1}/100, ID: {query['id']}")
            
            similarities = []
            for vector in vectors_corrected:
                sim = cosine_similarity(query['values'], vector['values'])
                similarities.append((vector['id'], sim))
            
            # Get top 5
            similarities.sort(key=lambda x: x[1], reverse=True)
            top5 = similarities[:5]
            
            # Only warn if the similarity is significantly different from 1.0
            query_similarity = next((sim for id, sim in similarities if id == query['id']), None)
            if top5[0][0] != query['id'] and abs(top5[0][1] - query_similarity) < 1e-6:
                print(f"\nNote: Query {query['id']} has an equally similar vector:")
                print(f"Self similarity: {query_similarity}")
                print(f"Top match: ID {top5[0][0]} with similarity {top5[0][1]}")
            
            # Store results
            results.append({
                'query_id': query['id'],
                'top1_id': top5[0][0],
                'top1_sim': top5[0][1],
                'top2_id': top5[1][0],
                'top2_sim': top5[1][1],
                'top3_id': top5[2][0],
                'top3_sim': top5[2][1],
                'top4_id': top5[3][0],
                'top4_sim': top5[3][1],
                'top5_id': top5[4][0],
                'top5_sim': top5[4][1],
            })
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"\nGenerated and saved results to {csv_path}")
        return results


def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}

def login():
    url = f"{host}/auth/login"
    data = {"username": "admin", "password": "admin", "pretty_print": False}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    global token
    token = response.text
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
        "collection_name": name,          # Name of the collection
        "name": name,                     # Name of the index
        "distance_metric_type": "cosine", # Type of distance metric (e.g., cosine, euclidean)
        "quantization": {
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
                "ef_construction": 128,      
                "ef_search": 128,            
                "neighbors_count": 16,
                "layer_0_neighbors_count": 32
            }
        }
    }
    response = requests.post(f"{base_url}/indexes", headers=generate_headers(), data=json.dumps(data), verify=False)

    return response.json()


# Function to create database (collection)
def create_db_old(vector_db_name, dimensions, max_val, min_val):
    url = f"{base_url}/collections"
    data = {
        "vector_db_name": vector_db_name,
        "dimensions": dimensions,
        "max_val": max_val,
        "min_val": min_val,
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


# Function to find a database (collection) by Id
def find_collection(id):
    url = f"{base_url}/collections/{id}"

    response = requests.get(url, headers=generate_headers(), verify=False)
    return response.json()


def create_transaction(collection_name):
    url = f"{base_url}/collections/{collection_name}/transactions"
    response = requests.post(url, headers=generate_headers(), verify=False)
    return response.json()


def create_vector_in_transaction(collection_name, transaction_id, vector):
    url = f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/vectors"
    data = {"id": vector["id"], "values": vector["values"], "metadata": {}}
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
    data = {"vectors": vectors}
    print(f"Request URL: {url}")
    print(f"Request Vectors Count: {len(vectors)}")
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False, timeout=10000
    )
    print(f"Response Status: {response.status_code}")
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to create vector: {response.status_code}")


def upsert_vectors_in_transaction(collection_name, transaction_id, vectors):
    url = f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/vectors"
    data = {"vectors": vectors}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


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


# Function to upsert vectors
def upsert_vector(vector_db_name, vectors):
    url = f"{base_url}/upsert"
    data = {"vector_db_name": vector_db_name, "vectors": vectors}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


# Function to search vector
def ann_vector_old(idd, vector_db_name, vector):
    url = f"{base_url}/search"
    data = {"vector_db_name": vector_db_name, "vector": vector}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return (idd, response.json())


def ann_vector(idd, vector_db_name, vector):
    url = f"{base_url}/search"
    data = {"vector_db_name": vector_db_name, "vector": vector}
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


# Function to fetch vector
def fetch_vector(vector_db_name, vector_id):
    url = f"{base_url}/fetch"
    data = {"vector_db_name": vector_db_name, "vector_id": vector_id}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


# Function to generate a random vector with given constraints
def generate_random_vector(rows, dimensions, min_val, max_val):
    return np.random.uniform(min_val, max_val, (rows, dimensions)).tolist()


def generate_random_vector_with_id(id, length):
    values = np.random.uniform(-1, 1, length).tolist()
    return {"id": id, "values": values}


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
    return np.sqrt(sum(v ** 2 for v in vec))


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
    query_array = np.array(query['values'])
    print(f"Query vector first 5 values: {query_array[:5]}")
    
    # First check similarity with itself
    self_similarity = cosine_similarity(query['values'], query['values'])
    print(f"Self similarity check for ID {query['id']}: {self_similarity}")
    
    for vector in vectors:
        similarity = cosine_similarity(query['values'], vector['values'])
        if similarity > 0.999:  # Check for very high similarities
            print(f"High similarity found:")
            print(f"Vector ID: {vector['id']}")
            print(f"Similarity: {similarity}")
            print(f"First 5 values of this vector: {vector['values'][:5]}")
            print(f"First 5 values of query vector: {query['values'][:5]}")
            
        similarities.append((vector['id'], similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]
    print(f"\nTop {k} matches for query {query['id']}:")
    for id, sim in top_k:
        print(f"ID: {id}, Similarity: {sim}")
        
    return top_k


def generate_vectors(txn_count, batch_count, batch_size, dimensions, perturbation_degree):
    vectors = [generate_random_vector_with_id(id, dimensions) for id in range(txn_count * batch_count * batch_size)]

    # Shuffle the vectors
    np.random.shuffle(vectors)
    return vectors

def batch_ann_search(vector_db_name, vectors):
    url = f"{base_url}/batch-search"
    data = {"vector_db_name": vector_db_name, "vectors": vectors}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to search vector: {response.status_code}")
    result = response.json()
    return result

def search(vectors, vector_db_name, query):
    ann_response = ann_vector(query["id"], vector_db_name, query["values"])
    bruteforce_result = bruteforce_search(vectors, query, 5)
    return (ann_response, bruteforce_result)

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

datasets = {
    "cohere-wiki-embedding-100k": {
        "id": "id",
        "embeddings": "emb",
        "size": 100_000,
        "range": (-1.0, 1.0)
    },
    "million-text-embeddings": {
        "id": None,
        "embeddings": "embedding",
        "size": 1_000_000,
        "range": (-0.1, 0.1)
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
}

def pre_process_vector(id, values):
    corrected_values = [float(v) for v in values]
    vector = {
        "id": int(id),
        "values": corrected_values
    }
    return vector

def read_dataset_from_parquet():
    print("\nChoose a dataset to test with:")
    dataset_names = list(datasets.keys())
    for i, key in enumerate(dataset_names):
        print(f"{i + 1}) {key}")
    print()
    dataset_idx = int(input("Select: ")) - 1
    dataset_name = dataset_names[dataset_idx]
    print(f"Reading {dataset_name} ...")
    dataset = datasets[dataset_name]

    dfs = []

    path = f"{dataset_name}/test0.parquet"

    while os.path.exists(path):
        dfs.append(pd.read_parquet(path))
        count = len(dfs)
        path = f"{dataset_name}/test{count}.parquet"

    df = pd.concat(dfs, ignore_index=True)

    print("Pre-processing ...")
    size = dataset["size"]
    dataset = df[[dataset["id"], dataset["embeddings"]]].values.tolist() if dataset["id"] != None else list(enumerate(row[0] for row in df[[dataset["embeddings"]]].values))

    vectors = []
    print("Dimension:", len(dataset[0][1]))

    for row in dataset:
        vector = pre_process_vector(row[0], row[1])
        vectors.append(vector)

    # # Flatten the list of lists
    # flattened_data = [value for row in dataset for value in row[1]]

    # # Calculate min and max
    # min_value = min(flattened_data)
    # max_value = max(flattened_data)

    # # Calculate percentages
    # total_count = len(flattened_data)
    # above_0_1 = sum(1 for x in flattened_data if x > 0.1) / total_count * 100
    # above_0_2 = sum(1 for x in flattened_data if x > 0.2) / total_count * 100
    # below_minus_0_1 = sum(1 for x in flattened_data if x < -0.1) / total_count * 100
    # below_minus_0_2 = sum(1 for x in flattened_data if x < -0.2) / total_count * 100

    # # Output results
    # print(f"Min value: {min_value}")
    # print(f"Max value: {max_value}")
    # print(f"Percentage above 0.1: {above_0_1}%")
    # print(f"Percentage above 0.2: {above_0_2}%")
    # print(f"Percentage below -0.1: {below_minus_0_1}%")
    # print(f"Percentage below -0.2: {below_minus_0_2}%")

    return vectors

def compare_vectors_in_detail(vectors_corrected, id1, id2):
    """Compare two vectors in detail"""
    vec1 = next(v for v in vectors_corrected if v['id'] == id1)
    vec2 = next(v for v in vectors_corrected if v['id'] == id2)
    
    values1 = np.array(vec1['values'])
    values2 = np.array(vec2['values'])
    
    # Compare raw values
    is_identical = np.allclose(values1, values2, rtol=1e-15)
    cosine_sim = np.dot(values1, values2) / (np.linalg.norm(values1) * np.linalg.norm(values2))
    
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


if __name__ == "__main__":
   # Create database
   vector_db_name = "testdb"
   dimensions = 768
   batch_size = 100
   max_vectors_per_transaction = 250000
   num_test_vectors = 100  # Number of vectors to test
   
   # Load or generate brute force results
   brute_force_results = load_or_generate_brute_force_results()
   print(f"Loaded {len(brute_force_results)} pre-computed brute force results")
   
   # Randomly select 100 vectors from the 1000 for this test run
   np.random.seed(int(time.time()))  # Random seed for each run
   selected_indices = np.random.choice(len(brute_force_results), num_test_vectors, replace=False)
   test_vectors = [brute_force_results[i] for i in selected_indices]
   print(f"Selected {len(test_vectors)} random vectors for testing")
   
   login_response = login()
   print("Login Response:", login_response)

   create_collection_response = create_db(
       name=vector_db_name,
       description="Test collection for vector database",
       dimension=dimensions,
   )
   print("Create Collection(DB) Response:", create_collection_response)
   create_explicit_index(vector_db_name)

   # Load vectors for server insertion
   # vectors_original = read_dataset_from_parquet()
   vectors_corrected = read_dataset_from_parquet()
   # for vec in vectors_original:
   #     corrected_values = [max(-0.9878, min(float(v), 0.9878)) for v in vec['values']]
   #     vectors_corrected.append({
   #         "id": vec['id'],
   #         "values": corrected_values
   #     })
   # Add shuffling here
   # print(f"Shuffling {len(vectors_corrected)} vectors...")
   # np.random.seed(42)  # Optional: for reproducibility
   # np.random.shuffle(vectors_corrected)
   # print("Shuffling complete")  

   # Add these verification calls:
   print("\nChecking for duplicate vectors...")
   verify_vector_uniqueness(vectors_corrected)

   print("\nVerifying specific vectors that showed inconsistency...")
   verify_specific_vectors(vectors_corrected, 45948, 84610)  # The two IDs that showed ~1.0 similarity

   total_vectors = len(vectors_corrected)
   print(f"Total vectors read from parquet files: {total_vectors}")

   # Select 2000 vectors for RPS testing (excluding match test vectors)
   match_test_ids = {test_vec['query_id'] for test_vec in test_vectors}
   available_vectors = [vec for vec in vectors_corrected if vec['id'] not in match_test_ids]
   rps_test_vectors = np.random.choice(available_vectors, 100_000, replace=False)
   print(f"Selected {len(rps_test_vectors)} separate vectors for RPS testing")

   # Initialize tracking set for vector IDs
   processed_vector_ids = set()
   total_vectors_inserted = 0
   start_time = time.time()

   # Calculate number of transactions needed
   num_transactions = (total_vectors + max_vectors_per_transaction - 1) // max_vectors_per_transaction
   print(f"\nWill process vectors in {num_transactions} transactions")
   
   for txn_idx in range(num_transactions):
       vectors_inserted_in_txn = 0
       transaction_id = None
       
       try:
           # Calculate transaction boundaries
           txn_start = txn_idx * max_vectors_per_transaction
           txn_end = min(txn_start + max_vectors_per_transaction, total_vectors)
           vectors_in_txn = vectors_corrected[txn_start:txn_end]
           
           print(f"\nTransaction {txn_idx + 1}: Processing vectors {txn_start} to {txn_end-1}")
           print(f"First vector ID: {vectors_in_txn[0]['id']}, Last vector ID: {vectors_in_txn[-1]['id']}")
           
           total_batches = (len(vectors_in_txn) + batch_size - 1) // batch_size

           transaction_response = create_transaction(vector_db_name)
           transaction_id = transaction_response["transaction_id"]
           print(f"Created transaction {transaction_id} ({txn_idx + 1}/{num_transactions})")

           with ThreadPoolExecutor(max_workers=32) as executor:
               futures = []
               for batch_idx in range(total_batches):
                   batch_start = batch_idx * batch_size
                   batch_end = min(batch_start + batch_size, len(vectors_in_txn))
                   current_batch = vectors_in_txn[batch_start:batch_end]
                   
                   # Track vector IDs in this batch
                   # batch_ids = {vec['id'] for vec in current_batch}
                   # if batch_ids & processed_vector_ids:
                   #     print(f"WARNING: Duplicate vectors found in batch {batch_idx}")
                   #     print(f"Duplicate IDs: {batch_ids & processed_vector_ids}")
                   # processed_vector_ids.update(batch_ids)
                   
                   # Verify batch boundaries
                   # print(f"Batch {batch_idx}: Processing vectors {batch_start} to {batch_end-1}")
                   # print(f"Batch {batch_idx}: First vector ID: {current_batch[0]['id']}, Last vector ID: {current_batch[-1]['id']}")
                   
                   vectors_inserted_in_txn += len(current_batch)
                   
                   futures.append(
                       executor.submit(
                           upsert_in_transaction,
                           vector_db_name,
                           transaction_id,
                           current_batch
                       )
                   )

               for future in as_completed(futures):
                   try:
                       future.result()
                   except Exception as e:
                       print(f"Error in future: {e}")

           print(f"Vectors inserted in transaction {transaction_id}: {vectors_inserted_in_txn}")
           commit_response = commit_transaction(vector_db_name, transaction_id)
           print(f"Committed transaction {transaction_id}: {commit_response}")
           total_vectors_inserted += vectors_inserted_in_txn
           transaction_id = None

       except Exception as e:
           print(f"Error in transaction: {e}")
           if transaction_id:
               try:
                   abort_transaction(vector_db_name, transaction_id)
                   print(f"Aborted transaction {transaction_id} due to error")
               except Exception as abort_error:
                   print(f"Error aborting transaction: {abort_error}")

   # Verification after all insertions
   total_unique_vectors = len(processed_vector_ids)
   print(f"\nVerification Results:")
   print(f"Total vectors in dataset: {total_vectors}")
   print(f"Total vectors inserted (including duplicates): {total_vectors_inserted}")
   print(f"Total unique vectors processed: {total_unique_vectors}")
   
   if total_unique_vectors != total_vectors:
       print(f"WARNING: Mismatch in vector counts! Missing {total_vectors - total_unique_vectors} vectors")
       # Find missing vector IDs
       all_vector_ids = {vec['id'] for vec in vectors_corrected}
       missing_ids = all_vector_ids - processed_vector_ids
       print(f"Missing vector IDs: {sorted(list(missing_ids))}")
   else:
       print("All vectors were processed successfully!")

   # End time for insertion
   insertion_end_time = time.time()
   print(f"\nInsertion time: {insertion_end_time - start_time} seconds")


   print("\nStarting similarity search tests...")
   # Modified comparison section - now sequential
   total_recall = 0
   total_queries = 0
   
   # Sequential processing of queries
   for i, test_vec in enumerate(test_vectors):
       try:
           query_vec = next(v for v in vectors_corrected if v['id'] == test_vec['query_id'])
           idr, ann_response = ann_vector(query_vec['id'], vector_db_name, query_vec['values'])
           
           if "RespVectorKNN" in ann_response and "knn" in ann_response["RespVectorKNN"]:
               query_id = test_vec['query_id']
               print(f"\nQuery {i + 1} (Vector ID: {query_id}):")
               
               # Get top 5 IDs from server
               server_top5 = [match[0] for match in ann_response["RespVectorKNN"]["knn"][:5]]
               print("Server top 5:", server_top5)
               
               # Get brute force top 5 from pre-computed results
               brute_force_top5 = [
                   test_vec[f'top{j}_id'] 
                   for j in range(1, 6)
               ]
               print("Brute force top 5:", brute_force_top5)
               
               # Calculate recall@5
               matches = sum(1 for id in server_top5 if id in brute_force_top5)
               recall = (matches / 5) * 100
               total_recall += recall
               total_queries += 1
               
               print(f"Recall@5 for this query: {recall}% ({matches}/5 matches)")
               
               # Print detailed comparison
               print("\nDetailed comparison:")
               print("Server Results:")
               for j, match in enumerate(ann_response["RespVectorKNN"]["knn"][:5]):
                   id = match[0]
                   cs = match[1]["CosineSimilarity"]
                   print(f"    {j + 1}: ID {id} (Cosine Similarity: {cs})")
               
               print("Brute Force Results:")
               for j in range(5):
                   id = test_vec[f'top{j+1}_id']
                   sim = test_vec[f'top{j+1}_sim']
                   print(f"    {j + 1}: ID {id} (Cosine Similarity: {sim})")

               # If top results differ, investigate
               if server_top5[0] != brute_force_top5[0]:
                   compare_vectors_in_detail(vectors_corrected, server_top5[0], brute_force_top5[0])
               
               # Add a small delay between queries
               time.sleep(0.1)  # 100ms delay between queries

       except Exception as e:
           print(f"Error in query {i + 1}: {e}")

   # Final timings and results
   end_time = time.time()
   total_time = end_time - start_time
   search_time = end_time - insertion_end_time

   if total_queries > 0:
       average_recall = total_recall / total_queries
       print(f"\nFinal Results:")
       print(f"Average Recall@5: {average_recall:.2f}%")
       print(f"Total time: {total_time:.2f} seconds")
       print(f"Insertion time: {insertion_end_time - start_time:.2f} seconds")
       print(f"Search time: {search_time:.2f} seconds")
       print(f"Average search time per query: {search_time/total_queries:.2f} seconds")
   else:
       print("No valid queries completed")
   
   # RPS Testing with separate test set
   print("\nStarting RPS (Requests Per Second) testing...")
   
   # Start RPS test
   print(f"Using {len(rps_test_vectors)} different test vectors for RPS testing")
   start_time_rps = time.time()
   
   results = []
   with ThreadPoolExecutor(max_workers=32) as executor:
       futures = []
       for i in range(0, len(rps_test_vectors), 100):
           batch = [vector["values"] for vector in rps_test_vectors[i:i+100]]
           futures.append(executor.submit(batch_ann_search, vector_db_name, batch))
   
       # Collect all results
       for future in as_completed(futures):
           try:
               future.result()
               results.append(True)
           except Exception as e:
               print(f"Error in RPS test: {e}")
               results.append(False)
   
   end_time_rps = time.time()
   actual_duration = end_time_rps - start_time_rps
   
   # Count successes and failures from results
   successful_requests = sum(results)
   failed_requests = len(results) - successful_requests
   total_requests = len(results)
   rps = successful_requests / actual_duration
   
   print("\nRPS Test Results:")
   print(f"Total Requests: {total_requests}")
   print(f"Successful Requests: {successful_requests}")
   print(f"Failed Requests: {failed_requests}")
   print(f"Test Duration: {actual_duration:.2f} seconds")
   print(f"Requests Per Second (RPS): {rps:.2f}")
   print(f"Success Rate: {(successful_requests/total_requests*100):.2f}%")

