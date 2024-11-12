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


def create_db(name, description=None, dimension=1024):
    url = f"{base_url}/collections"
    data = {
        "name": name,
        "description": description,
        "dense_vector": {
            "enabled": True,
            "auto_create_index": True,
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
        url, headers=generate_headers(), data=json.dumps(data), verify=False
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
            f"Upsert complete for base vector {base_idx} and its {len(batch_vectors)-1} perturbations"
        )

        return (base_idx, base_vector)
    except Exception as e:
        print(f"Error processing base vector {base_idx}: {e}")
        raise


if __name__ == "__main__":
    # Create database
    vector_db_name = "testdb"
    dimensions = 1024
    max_val = 1.0
    min_val = -1.0
    perturbation_degree = 0.25  # Degree of perturbation

    # first login to get the auth jwt token
    login_response = login()
    print("Login Response:", login_response)

    create_collection_response = create_db(
        name=vector_db_name,
        description="Test collection for vector database",
        dimension=dimensions,
    )
    print("Create Collection(DB) Response:", create_collection_response)

    shortlisted_vectors = []
    start_time = time.time()

    for req_ct in range(1):
        transaction_id = None
        try:
            # Create a new transaction
            transaction_response = create_transaction(vector_db_name)
            transaction_id = transaction_response["transaction_id"]
            print(f"Created transaction: {transaction_id}")

            # Process vectors concurrently
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for base_idx in range(100):
                    futures.append(
                        executor.submit(
                            process_base_vector_batch,
                            req_ct,
                            base_idx,
                            vector_db_name,
                            transaction_id,
                            dimensions,
                            perturbation_degree,
                        )
                    )

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        shortlisted_vectors.append(result)
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
    best_matches = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for idd, vector in shortlisted_vectors:
            futures.append(
                executor.submit(ann_vector, idd, vector_db_name, vector["values"])
            )

        for i, future in enumerate(as_completed(futures)):
            try:
                (idr, ann_response) = future.result()
                print(f"ANN Vector Response <<< {idr} >>>:", ann_response)
                if (
                    "RespVectorKNN" in ann_response
                    and "knn" in ann_response["RespVectorKNN"]
                ):
                    best_matches.append(
                        ann_response["RespVectorKNN"]["knn"][0][1]
                    )  # Collect the second item in the knn list
            except Exception as e:
                print(f"Error in ANN vector {i + 1}: {e}")

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for idd, vector in shortlisted_vectors:
            futures.append(executor.submit(fetch_vector, vector_db_name, vector["id"]))

        for i, future in enumerate(as_completed(futures)):
            try:
                fetch_response = future.result()
                print(f"Fetch Vector Response {i + 1}:", fetch_response)

            except Exception as e:
                print(f"Error in Fetch vector {i + 1}: {e}")

    if best_matches:
        best_match_average = sum(m["CosineSimilarity"] for m in best_matches) / len(
            best_matches
        )
        print(f"\n\nBest Match Average: {best_match_average}")
    else:
        print("No valid matches found.")

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
