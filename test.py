import requests
import json
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define your dynamic variables
token = "8cf11a8cb97b0e002b31197c5808d13e3b18e488234a61946690668db5c5fece"
base_url = "https://127.0.0.1:8443/vectordb"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-type": "application/json"
}

# Function to create database
def create_db(vector_db_name, dimensions, max_val, min_val):
    url = f"{base_url}/createdb"
    data = {
        "vector_db_name": vector_db_name,
        "dimensions": dimensions,
        "max_val": max_val,
        "min_val": min_val
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    return response.json()

# Function to upsert vectors
def upsert_vector(vector_db_name, vector):
    url = f"{base_url}/upsert"
    data = {
        "vector_db_name": vector_db_name,
        "vector": vector
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    return response.json()

# Function to generate a random vector with given constraints
def generate_random_vector(rows, dimensions, min_val, max_val):
    return np.random.uniform(min_val, max_val, (rows, dimensions)).tolist()

# Example usage
if __name__ == "__main__":
    # Create database
    vector_db_name = "testdb"
    dimensions = 1024
    max_val = 1.0
    min_val = 0.0
    rows = 100

    create_response = create_db(vector_db_name, dimensions, max_val, min_val)
    print("Create DB Response:", create_response)

    # Upsert vectors concurrently
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(100):
            vector = generate_random_vector(rows, dimensions, min_val, max_val)
            futures.append(executor.submit(upsert_vector, vector_db_name, vector))

        for i, future in enumerate(as_completed(futures)):
            try:
                upsert_response = future.result()
                print(f"Upsert Vector Response {i+1}:", upsert_response)
            except Exception as e:
                print(f"Error in upsert vector {i+1}: {e}")
