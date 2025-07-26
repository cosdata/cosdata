import time
import random
import numpy as np
from cosdata import Client
import requests

def generate_metadata_fields(field_defs):
    """Generate a random metadata dict based on field definitions."""
    return {field['name']: random.choice(field['values']) for field in field_defs}

def main():
    # Configurable parameters
    NUM_FIELDS = 5
    FIELD_CARDINALITY = 32
    NUM_VECTORS = 1000
    VECTOR_DIM = 128
    NUM_QUERIES = 50
    TOP_K = 5
    HOST = "http://127.0.0.1:8443"
    USERNAME = "admin"
    PASSWORD = "admin"

    # 1. Create SDK client (handles authentication)
    client = Client(
        host=HOST,
        username=USERNAME,
        password=PASSWORD,
        verify=False
    )

    # 2. Create a high-cardinality metadata schema
    field_defs = [
        {
            "name": f"field_{i}",
            "values": [f"val_{j}" for j in range(FIELD_CARDINALITY)]
        }
        for i in range(NUM_FIELDS)
    ]
    metadata_schema = {
        "fields": field_defs,
        "supported_conditions": [
            {"op": "and", "field_names": [f["name"] for f in field_defs]}
        ]
    }

    # 3. Try to create collection with metadata_schema using SDK
    collection_name = f"dp_benchmark_{random.randint(1000,9999)}"
    print(f"Creating collection: {collection_name}")
    try:
        collection = client.create_collection(
            name=collection_name,
            dimension=VECTOR_DIM,
            description="DP benchmark collection",
            metadata_schema=metadata_schema  # Will work if SDK supports it
        )
    except TypeError as e:
        print("SDK does not support metadata_schema directly, falling back to raw HTTP for collection creation...")
        # Use raw HTTP for collection creation
        url = f"{HOST}/vectordb/collections"
        token = getattr(client, 'token', None) or getattr(client, '_token', None)
        headers = {"Authorization": f"Bearer {token}", "Content-type": "application/json"}
        data = {
            "name": collection_name,
            "description": "DP benchmark collection",
            "dense_vector": {"enabled": True, "auto_create_index": False, "dimension": VECTOR_DIM},
            "sparse_vector": {"enabled": False, "auto_create_index": False},
            "tf_idf_options": {"enabled": False},
            "metadata_schema": metadata_schema,
            "config": {"max_vectors": None, "replication_factor": None},
            "store_raw_text": False,
        }
        resp = requests.post(url, headers=headers, json=data, verify=False)
        resp.raise_for_status()
        collection = client.get_collection(collection_name)

    # 4. Create index
    print("Creating index...")
    collection.create_index(
        distance_metric="cosine",
        num_layers=7,
        max_cache_size=1000,
        ef_construction=256,
        ef_search=128,
        neighbors_count=32,
        level_0_neighbors_count=64
    )

    # 5. Generate and insert vectors with random metadata
    print(f"Inserting {NUM_VECTORS} vectors...")
    vectors = []
    for i in range(NUM_VECTORS):
        vectors.append({
            "id": f"vec_{i}",
            "dense_values": np.random.uniform(-1, 1, VECTOR_DIM).tolist(),
            "metadata": generate_metadata_fields(field_defs)
        })

    with collection.transaction() as txn:
        txn.batch_upsert_vectors(vectors, max_workers=8, max_retries=3)

    print("Waiting for collection to finish indexing...")
    time.sleep(10)  # Adjust as needed for your environment

    # 6. Prepare complex metadata filters for queries
    queries = []
    for _ in range(NUM_QUERIES):
        # Randomly select a vector and use its metadata as a filter (AND on all fields)
        vec = random.choice(vectors)
        filter_predicates = [
            {"field_name": k, "field_value": v, "operator": "Equal"}
            for k, v in vec["metadata"].items()
        ]
        filter_json = {"And": filter_predicates}
        queries.append((vec["dense_values"], filter_json))

    # 7. Benchmark search latency with metadata filters
    print(f"Running {NUM_QUERIES} dense search queries with complex metadata filters...")
    latencies = []
    for query_vector, metadata_filter in queries:
        start = time.time()
        _ = collection.search.dense(
            query_vector=query_vector,
            top_k=TOP_K,
            filter=metadata_filter,
            return_raw_text=False
        )
        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage search latency with complex metadata filters: {avg_latency:.2f} ms")
    print(f"Min: {min(latencies):.2f} ms, Max: {max(latencies):.2f} ms")

    print("\nThis benchmark directly exercises the DP-optimized metadata path. "
          "Lower latency here means the DP optimization is working!")

if __name__ == "__main__":
    main() 