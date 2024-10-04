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
    data = {
    "username": "admin",
    "password": "admin",
    "pretty_print": False
    }
    response = requests.post(url, headers=generate_headers(), data=json.dumps(data), verify=False)
    global token
    token = response.text
    return token
    


# Function to create database (collection)
def create_db(vector_db_name, dimensions, max_val, min_val):
    url = f"{base_url}/collections"
    data = {
        "vector_db_name": vector_db_name,
        "dimensions": dimensions,
        "max_val": max_val,
        "min_val": min_val,
    }
    response = requests.post(url, headers=generate_headers(), data=json.dumps(data), verify=False)
    return response.json()

# Function to find a database (collection) by Id 
def find_collection(id):
    url = f"{base_url}/collections/{id}"

    response = requests.get(url,headers=generate_headers(), verify=False)
    return response.json()


# Function to upsert vectors
def upsert_vector(vector_db_name, vectors):
    url = f"{base_url}/upsert"
    data = {"vector_db_name": vector_db_name, "vectors": vectors}
    response = requests.post(url, headers=generate_headers(), data=json.dumps(data), verify=False)
    return response.json()


# Function to search vector
def ann_vector(idd, vector_db_name, vector):
    url = f"{base_url}/search"
    data = {"vector_db_name": vector_db_name, "vector": vector}
    response = requests.post(url, headers=generate_headers(), data=json.dumps(data), verify=False)
    return (idd, response.json())


# Function to fetch vector
def fetch_vector(vector_db_name, vector_id):
    url = f"{base_url}/fetch"
    data = {"vector_db_name": vector_db_name, "vector_id": vector_id}
    response = requests.post(url, headers=generate_headers(), data=json.dumps(data), verify=False)
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


# Example usage
if __name__ == "__main__":
    # Create database
    vector_db_name = "testdb"
    dimensions = 1024
    max_val = 1.0
    min_val = -1.0
    rows = 100
    perturbation_degree = 0.25  # Degree of perturbation

    # first login to get the auth jwt token
    login_response = login()
    print("Login Response:", login_response)

    
    create_Collection_response = create_db(vector_db_name, dimensions, max_val, min_val)
    print("Create Collection(DB) Response:", create_Collection_response)

    find_collection_response = find_collection(create_Collection_response["id"])
    print("Find Collection(DB) Response:", find_collection_response)

    shortlisted_vectors = []

    # Start time
    start_time = time.time()

    # Upsert vectors concurrently
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        #
        # number of upsert calls
        #
        for req_ct in range(1):
            base_vector = generate_random_vector_with_id(req_ct * rows, dimensions)

            # Generate a single random vector
            final_list = [base_vector]
            for row_ct in range(1, rows):
                idd = (req_ct * rows) + row_ct
                # Generate the perturbation
                perturbation = np.random.uniform( -perturbation_degree, perturbation_degree, dimensions)
                
                # Apply the perturbation and clamp the values within the range of -1 to 1
                perturbed_values = base_vector["values"] + perturbation
                clamped_values = np.clip(perturbed_values, -1, 1)
                perturbed_vector = {}
                perturbed_vector["values"] = clamped_values.tolist()
                perturbed_vector["id"] = idd

                # print(base_vector["values"][:10])
                # print( perturbed_vector["values"][:10] )
                cs = cosine_similarity(base_vector["values"], perturbed_vector["values"] )
                # print ("cosine similarity of perturbed vec: ", row_ct, cs)
                final_list.append(perturbed_vector)
                # if np.random.rand() < 0.01:  # 1 in 100 probability
                #     shortlisted_vectors.append(perturbed_vector)
            shortlisted_vectors.append((idd - (rows - 1), base_vector))

            futures.append(executor.submit(upsert_vector, vector_db_name, final_list))

        for i, future in enumerate(as_completed(futures)):
            try:
                upsert_response = future.result()
                print(f"Upsert Vector Response {i + 1}: ", upsert_response)
            except Exception as e:
                print(f"Error in upsert vector {i + 1}: {e}")


    # End time
    end_time = time.time()

    # Apply perturbations to shortlisted vectors
    # for i in range(len(shortlisted_vectors)):
    #     shortlisted_vectors[i] = perturb_vector(
    #         shortlisted_vectors[i], perturbation_degree
    #     )

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
        best_match_average = sum(m["CosineSimilarity"] for m in best_matches) / len(best_matches)
        print(f"\n\nBest Match Average: {best_match_average}")
    else:
        print("No valid matches found.")

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
