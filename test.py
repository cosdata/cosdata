import requests
import json
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = "8cf11a8cb97b0e002b31197c5808d13e3b18e488234a61946690668db5c5fece"
base_url = "https://127.0.0.1:8443/vectordb"
headers = {"Authorization": f"Bearer {token}", "Content-type": "application/json"}


# Function to create database
def create_db(vector_db_name, dimensions, max_val, min_val):
    url = f"{base_url}/createdb"
    data = {
        "vector_db_name": vector_db_name,
        "dimensions": dimensions,
        "max_val": max_val,
        "min_val": min_val,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    return response.json()


# Function to upsert vectors
def upsert_vector(vector_db_name, vectors):
    url = f"{base_url}/upsert"
    data = {"vector_db_name": vector_db_name, "vectors": vectors}
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    return response.json()


# Function to search vector
def ann_vector(vector_db_name, vector):
    url = f"{base_url}/search"
    data = {"vector_db_name": vector_db_name, "vector": vector}
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    return response.json()


# Function to generate a random vector with given constraints
def generate_random_vector(rows, dimensions, min_val, max_val):
    return np.random.uniform(min_val, max_val, (rows, dimensions)).tolist()


def generate_random_vector_with_id(id, length):
    values = np.random.uniform(-1, 1, length).tolist()
    return {"id": id, "values": values}

def perturb_vector(vector, perturbation_degree):
    # Generate the perturbation
    perturbation = np.random.uniform(-perturbation_degree, perturbation_degree, len(vector["values"]))
    # Apply the perturbation and clamp the values within the range of -1 to 1
    perturbed_values = np.array(vector["values"]) + perturbation
    clamped_values = np.clip(perturbed_values, -1, 1)
    vector["values"] = clamped_values.tolist()
    return vector

# Example usage
if __name__ == "__main__":
    # Create database
    vector_db_name = "testdb"
    dimensions = 1024
    max_val = 1.0
    min_val = 0.0
    rows = 100
    perturbation_degree = 0.1  # Degree of perturbation

    create_response = create_db(vector_db_name, dimensions, max_val, min_val)
    print("Create DB Response:", create_response)

    shortlisted_vectors = []

    # Start time
    start_time = time.time()

    # Upsert vectors concurrently
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        #
        # number of upsert calls
        #
        for req_ct in range(200):
            base_vector = generate_random_vector_with_id(req_ct * rows, dimensions)
            # Generate a single random vector
            final_list = [base_vector]
            for row_ct in range(1, rows):
                perturbed_vector = perturb_vector(
                    generate_random_vector_with_id(
                        (req_ct * rows) + row_ct, dimensions
                    ),
                    perturbation_degree,
                )
                final_list.append(perturbed_vector)
                if np.random.rand() < 0.01:  # 1 in 100 probability
                    shortlisted_vectors.append(perturbed_vector)
            futures.append(executor.submit(upsert_vector, vector_db_name, final_list))

        for i, future in enumerate(as_completed(futures)):
            try:
                upsert_response = future.result()
                print(f"Upsert Vector Response {i + 1}:", upsert_response)
            except Exception as e:
                print(f"Error in upsert vector {i + 1}: {e}")

    # Apply perturbations to shortlisted vectors
    # for i in range(len(shortlisted_vectors)):
    #     shortlisted_vectors[i] = perturb_vector(
    #         shortlisted_vectors[i], perturbation_degree
    #     )

    # Search vector concurrently using perturbed vectors
    best_matches = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for vector in shortlisted_vectors:
            futures.append(
                executor.submit(ann_vector, vector_db_name, vector["values"])
            )

        for i, future in enumerate(as_completed(futures)):
            try:
                ann_response = future.result()
                print(f"ANN Vector Response {i + 1}:", ann_response)
                if (
                    "RespVectorKNN" in ann_response
                    and "knn" in ann_response["RespVectorKNN"]
                ):
                    best_matches.append(
                        ann_response["RespVectorKNN"]["knn"][0][1]
                    )  # Collect the second item in the knn list
            except Exception as e:
                print(f"Error in ANN vector {i + 1}: {e}")

    # End time
    end_time = time.time()

    if best_matches:
        best_match_average = sum(best_matches) / len(best_matches)
        print(f"\n\nBest Match Average: {best_match_average}")
    else:
        print("No valid matches found.")

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
