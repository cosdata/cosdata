#!/usr/bin/env python

import numpy as np
from cosdata import Client
import getpass
import json
import time
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

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

def test_basic_functionality():
    # Get password from .env file or prompt securely
    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter your database password: ")

    # Initialize client
    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    username = os.getenv("COSDATA_USERNAME", "admin")
    client = Client(
        host=host,
        username=username,
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
        final_status, success = txn_id.poll_completion( 
            target_status='complete', 
            max_attempts=3, 
            sleep_interval=2
        )
        
        if not success:
            print(f"Transaction did not complete successfully. Final status: {final_status}")

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
