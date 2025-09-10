import os
import getpass
import random
import time
from cosdata import Client
import requests
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global configuration
CARDINALITY = 4
NUM_LABELS = 20
NUM_SEQUENCES = 50_000_000  # Number of sequences to generate
BATCH_SIZE = 10000       # Number of sequences per upsert batch
NUM_LATENCY_QUERIES = 100  # Number of test queries for latency measurement
QUERY_MIN_LENGTH = 2     # Minimum query length
QUERY_MAX_LENGTH = 5     # Maximum query length

class Collection:
    """
    Represents a collection in the vector database.
    """

    def __init__(self, client, name: str):
        """
        Initialize a collection.

        Args:
            client: Client instance
            name: Name of the collection
        """
        self.client = client
        self.name = name

def generate_distinct_u32s(count):
    return random.sample(range(0, 2**32), count)

def create_collection(client: Client, collection_name: str) -> Collection:
    client._ensure_session()

    url = f"{client.base_url}/collections"
    data = {
        "name": collection_name,
        "description": None,
        "dense_vector": {"enabled": False, "dimension": 1},
        "sparse_vector": {"enabled": False},
        "tf_idf_options": {"enabled": False},
        "om_options": {"enabled": True},
        "config": {"max_vectors": None, "replication_factor": None},
        "store_raw_text": False,
    }

    response = requests.post(
        url,
        headers=client._get_headers(),
        data=json.dumps(data),
        verify=client.verify_ssl,
    )

    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to create collection: {response.text}")

    return Collection(client, collection_name)

def create_index(collection: Collection):
    url = f"{collection.client.base_url}/collections/{collection.name}/indexes/om"

    response = requests.post(
        url,
        headers=collection.client._get_headers(),
        verify=collection.client.verify_ssl,
    )

    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to create index: {response.text}")

def upsert_batch(collection: Collection, vectors_batch):
    url = f"{collection.client.base_url}/collections/{collection.name}/streaming/om_upsert"
    data = {"vectors": vectors_batch}

    response = requests.post(
        url,
        headers=collection.client._get_headers(),
        data=json.dumps(data),
        verify=collection.client.verify_ssl,
    )

    if response.status_code not in [200, 201, 204]:
        raise Exception(f"Failed to streaming upsert vectors: {response.text}")

def sum_query(collection: Collection, labels) -> float:
    url = f"{collection.client.base_url}/collections/{collection.name}/search/om-sum"
    data = {
        "labels": labels
    }

    response = requests.post(
        url,
        headers=collection.client._get_headers(),
        data=json.dumps(data),
        verify=collection.client.verify_ssl,
    )

    if response.status_code != 200:
        raise Exception(f"Failed to search sparse vector: {response.text}")

    return response.json()["sum"]

def generate_queries(num_queries, min_length, max_length, labels_hashes):
    """Generate random queries for testing"""
    queries = []
    for _ in range(num_queries):
        query_length = random.randint(min_length, max_length)
        query = []
        for _ in range(query_length):
            label_idx = random.randint(0, NUM_LABELS - 1)
            hash_idx = random.randint(0, CARDINALITY - 1)
            query.append(labels_hashes[label_idx][hash_idx])
        queries.append(query)
    return queries

def run_latency_test(collection, queries):
    """Run latency test and measure individual query performance"""
    print(f"Running latency test with {len(queries)} queries...")
    
    latencies = []

    print("Response for first 10 queries")
    
    # Warm-up: run a few queries first
    for i in range(min(10, len(queries))):
        result = sum_query(collection, queries[i])
        print(f"{i + 1}. {result}")
    
    # Measure latency for each query
    for i, query in enumerate(queries):
        start_time = time.time()
        result = sum_query(collection, query)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency_ms)
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1} queries...")
    
    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = statistics.quantiles(latencies, n=100)[94]  # 95th percentile
    
    print("Latency Results:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Minimum latency: {min_latency:.2f} ms")
    print(f"  Maximum latency: {max_latency:.2f} ms")
    print(f"  95th percentile: {p95_latency:.2f} ms")
    
    return latencies, avg_latency, min_latency, max_latency, p95_latency

def generate_sequence_batch(batch_size, labels_hashes):
    """Generate a batch of sequences without storing the entire corpus in memory"""
    batch = []
    for _ in range(batch_size):
        sequence = []
        for label_hashes in labels_hashes:
            chosen_hash = random.choice(label_hashes)
            sequence.append(chosen_hash)
        # Assign a random float value to the sequence
        value = random.uniform(1.0, 100.0)
        batch.append({"key": sequence, "value": value})
    return batch

def generate_and_upsert_batch(collection, batch_size, labels_hashes):
    batch = generate_sequence_batch(batch_size, labels_hashes)
    upsert_batch(collection, batch)

def main():
    # Generate 80 distinct u32 values
    total_hashes = CARDINALITY * NUM_LABELS
    distinct_hashes = generate_distinct_u32s(total_hashes)
    
    # Group hashes by label: each label has CARDINALITY hashes
    labels_hashes = []
    for i in range(NUM_LABELS):
        start_index = i * CARDINALITY
        end_index = start_index + CARDINALITY
        labels_hashes.append(distinct_hashes[start_index:end_index])
    
    # Generate random queries for latency tests
    latency_queries = generate_queries(NUM_LATENCY_QUERIES, QUERY_MIN_LENGTH, QUERY_MAX_LENGTH, labels_hashes)
    
    # Connect to server
    DB_NAME = "om_test_collection"
    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter your database password: ")

    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    username = os.getenv("COSDATA_USERNAME", "admin")
    client = Client(host=host, username=username, password=password, verify=False)
    
    # Create collection and index
    collection = create_collection(client, DB_NAME)
    create_index(collection)
    
    # Upsert vectors in batches without storing the entire corpus
    print(f"Generating and upserting {NUM_SEQUENCES} vectors in batches of {BATCH_SIZE}...")
    start = time.time()
    futures = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(0, NUM_SEQUENCES, BATCH_SIZE):
            current_batch_size = min(BATCH_SIZE, NUM_SEQUENCES - i)
            future = executor.submit(generate_and_upsert_batch, collection, current_batch_size, labels_hashes)
            futures.append(future)

        for i, future in enumerate(as_completed(futures)):
            print(f"Upserted batch {i + 1}/{(NUM_SEQUENCES-1)//BATCH_SIZE + 1}")

    end = time.time()
    print(f"Inserted {NUM_SEQUENCES} in {end - start} seconds")
    
    # Test latency
    latencies, avg_latency, min_latency, max_latency, p95_latency = run_latency_test(collection, latency_queries)

if __name__ == "__main__":
    main()
