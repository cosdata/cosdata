#!/usr/bin/env python

import numpy as np
from cosdata import Client
import getpass
import json
import requests
import time

def format_vector(vector):
    """Format a vector for better readability"""
    # Handle Vector class instances
    if hasattr(vector, 'dense_values'):
        dense_values = vector.dense_values
        if dense_values:
            # Show first 5 dimensions and total count
            preview = dense_values[:5]
            formatted = {
                "id": vector.id,
                "dense_values": f"{preview}... (total: {len(dense_values)} dimensions)"
            }
        else:
            formatted = {
                "id": vector.id,
                "dense_values": "[]"
            }
            return json.dumps(formatted, indent=2)
    # Handle dictionary format
    elif isinstance(vector, dict):
        dense_values = vector.get('dense_values', [])
        if dense_values:
            # Show first 5 dimensions and total count
            preview = dense_values[:5]
            formatted = {
                "id": vector.get("id", "N/A"),
                "dense_values": f"{preview}... (total: {len(dense_values)} dimensions)"
            }
        else:
            formatted = {
                "id": vector.get("id", "N/A"),
                "dense_values": "[]"
            }
        return json.dumps(formatted, indent=2)
    return str(vector)

def get_transaction_status(client, coll_name, txn_id):
    host = client.host
    url = f"{host}/vectordb/collections/{coll_name}/transactions/{txn_id}/status"
    resp = requests.get(url, headers=client._get_headers(), verify=False)
    result = resp.json()
    return result['status']

def test_basic_functionality():
    # Get password securely
    password = getpass.getpass("Enter your database password: ")

    # Initialize client
    client = Client(
        host="http://127.0.0.1:8443",
        username="admin",
        password=password,
        verify=False
    )

    # Test collection name
    collection_name = "test_collection"

    try:
        # Create a collection
        print("Creating collection...")
        collection = client.create_collection(
            name=collection_name,
            dimension=768,
            description="Test collection for basic functionality"
        )
        print("Collection created successfully")

        # Create an index
        print("Creating index...")
        index = collection.create_index(
            distance_metric="cosine",
            num_layers=7,
            max_cache_size=1000,
            ef_construction=512,
            ef_search=256,
            neighbors_count=32,
            level_0_neighbors_count=64
        )
        print("Index created successfully")

        # Generate test vectors
        print("Generating test vectors...")
        test_vectors = []
        for i in range(10):
            values = np.random.uniform(-1, 1, 768).tolist()
            test_vectors.append({
                "id": f"vec_{i}",
                "dense_values": values,
            })

        # Insert vectors using transaction
        print("Inserting vectors...")
        txn_id = None
        with collection.transaction() as txn:
            txn.batch_upsert_vectors(test_vectors)
            txn_id = txn.transaction_id

        print("Vectors inserted successfully")

        print("Waiting for transaction to complete")
        txn_status = 'indexing_in_progress'
        remaining_attempts = 3
        while txn_status != 'complete':
            txn_status = get_transaction_status(client, collection_name, txn_id)
            time.sleep(2)
            remaining_attempts -= 1
            if remaining_attempts == 0:
                print("Max attempts waiting for transaction to complete exceeded")
                break

        # Test search
        print("\nTesting search...")
        query_vector = test_vectors[0]["dense_values"]
        results = collection.search.dense(
            query_vector=query_vector,
            top_k=5,
            return_raw_text=True
        )

        print("\nSearch results:")
        for i, result in enumerate(results.get("results", []), 1):
            # Fetch the full vector for each result
            vector = collection.vectors.get(result["id"])
            print(f"\nResult {i} (score: {result['score']:.4f}):")
            print(format_vector(vector))

        # Test vector retrieval
        print("\nTesting vector retrieval...")
        vector = collection.vectors.get("vec_0")
        print("\nRetrieved vector:")
        print(format_vector(vector))

        print("\nAll basic functionality tests passed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        # Cleanup
        try:
            collection.delete()
            print("Test collection deleted")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    test_basic_functionality()
