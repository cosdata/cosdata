import requests
import json
import pandas as pd 
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = "http://127.0.0.1:8443"
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

def bruteforce_search(vectors, query, k=5):
    similarities = []
    for vector in vectors:
        similarity = cosine_similarity(query["values"], vector["values"])
        similarities.append((vector["id"], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


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

def write_to_csv(bruteforce_results, csv_file):

    df = pd.DataFrame(bruteforce_results, columns=["ID", "CosineSimilarity"])
    df.to_csv(csv_file, mode='a', header=False, index=False)
    print(f"Results appended to {csv_file}")

# Function to read from the parquet file
def read_dataset_from_parquet():

    df = pd.read_parquet("test.parquet", engine='pyarrow')

    dataset = df[['id', 'emb']]

    columns = list(dataset.columns)

    vectors = []

    for index, row in dataset.iterrows():
        vector = {
            "id": row['id'],
            "values": row['emb'].tolist()
        }
        vectors.append(vector)

    return vectors


if __name__ == "__main__":

    vector_db_name = "testdb"
    dimensions = 768
    max_val = 1.0
    min_val = -1.0
    batch_size = 1
    batch_count = 1

    login_response = login()
    print("Login Response:", login_response)
    
    create_collection_response = create_db(
        name=vector_db_name,
        description="Test collection for vector database",
        dimension=dimensions,
    )
    print("Create Collection(DB) Response:", create_collection_response)



    vectors = read_dataset_from_parquet()

    best_matches_bruteforce = []
    best_matches_server = []


    shortlisted_vectors = vectors[:100]

    start_time = time.time()

    for query in shortlisted_vectors:
        bruteforce_results = bruteforce_search(vectors, query, 5)
        best_matches_bruteforce.extend(result[1] for result in bruteforce_results)
        print("Brute force search:")
        for j, result in enumerate(bruteforce_results):
            id = result[0]
            cs = result[1]
            print(f"    {j + 1}: {id} ({cs})")
            
        write_to_csv(bruteforce_results, 'bruteforce.csv')

    if best_matches_bruteforce:
        best_match_bruteforce_average = sum(best_matches_bruteforce) / len(best_matches_bruteforce)
        print(f"\nBest Match Brute force Average: {best_match_bruteforce_average}")
    else:
        print("No valid matches found.")

    
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for query in shortlisted_vectors:
            futures.append(executor.submit(bruteforce_search, vectors, query, 5))
            best_matches_bruteforce.extend(result[1] for result in bruteforce_results)

        for future in as_completed(futures):
            try:
                server_results = future.result()
                best_matches_server.extend(result[1] for result in server_results)
            except Exception as e:
                print(f"Error in server search: {e}")

    if best_matches_server:
        best_match_server_average = sum(best_matches_server) / len(best_matches_server)
        print(f"\nBest Match Server Average: {best_match_server_average}")
    else:
        print("No valid server matches found.")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")