#!/usr/bin/env python

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from cosdata import Client
import json
import getpass
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def format_vector(vector):
    """Format a vector for better readability"""
    # Handle Vector class instances
    if hasattr(vector, "dense_values"):
        dense_values = vector.dense_values
        if dense_values:
            # Show first 5 dimensions and total count
            preview = dense_values[:5]
            formatted = {
                "id": vector.id,
                "dense_values": f"{preview}... (total: {len(dense_values)} dimensions)",
            }
        else:
            formatted = {"id": vector.id, "dense_values": "[]"}
        return json.dumps(formatted, indent=2)
    # Handle dictionary format
    elif isinstance(vector, dict):
        dense_values = vector.get("dense_values", [])
        if dense_values:
            # Show first 5 dimensions and total count
            preview = dense_values[:5]
            formatted = {
                "id": vector.get("id", "N/A"),
                "dense_values": f"{preview}... (total: {len(dense_values)} dimensions)",
            }
        else:
            formatted = {"id": vector.get("id", "N/A"), "dense_values": "[]"}
        return json.dumps(formatted, indent=2)
    return str(vector)


def generate_random_vector(id: int, dimension: int) -> dict:
    values = np.random.uniform(-1, 1, dimension).tolist()
    return {
        "id": f"vec_{id}",
        "dense_values": values,
        "document_id": f"doc_{id // 10}",  # Group vectors into documents
    }


def generate_perturbation(
    base_vector: dict, id: int, perturbation_degree: float
) -> dict:
    perturbation = np.random.uniform(
        -perturbation_degree, perturbation_degree, len(base_vector["dense_values"])
    )
    perturbed_values = np.array(base_vector["dense_values"]) + perturbation
    clamped_values = np.clip(perturbed_values, -1, 1)
    return {
        "id": f"vec_{id}",
        "dense_values": clamped_values.tolist(),
        "document_id": f"doc_{id // 10}",  # Group vectors into documents
    }


def generate_batch(
    base_idx: int, batch_size: int, dimensions: int, perturbation_degree: float
) -> list:
    """Generate a batch of vectors"""
    batch_vectors = []
    base_vector = generate_random_vector(base_idx * batch_size, dimensions)
    batch_vectors.append(base_vector)

    for i in range(batch_size - 1):
        perturbed_vector = generate_perturbation(
            base_vector, base_idx * batch_size + i + 1, perturbation_degree
        )
        batch_vectors.append(perturbed_vector)

    return batch_vectors


def test_performance():
    # Get password from .env file or prompt securely
    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter your database password: ")

    # Initialize client
    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    username = os.getenv("COSDATA_USERNAME", "admin")
    client = Client(host=host, username=username, password=password, verify=False)

    # Test parameters
    collection_name = "performance_test"
    dimensions = 1024
    perturbation_degree = 0.95
    batch_size = 100
    batch_count = 1000
    num_workers = 32

    try:
        # Create collection
        print("Creating collection...")
        collection = client.create_collection(
            name=collection_name,
            dimension=dimensions,
            description="Performance test collection",
        )

        # Create index
        print("Creating index...")
        collection.create_index(
            distance_metric="cosine",
            num_layers=7,
            max_cache_size=1000,
            ef_construction=512,
            ef_search=256,
            neighbors_count=32,
            level_0_neighbors_count=64,
        )

        start_time = time.time()

        # Generate vectors in parallel
        print("Generating test vectors...")
        test_vectors = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for base_idx in range(batch_count):
                futures.append(
                    executor.submit(
                        generate_batch,
                        base_idx,
                        batch_size,
                        dimensions,
                        perturbation_degree,
                    )
                )

            # Collect all generated vectors
            for future in as_completed(futures):
                batch = future.result()
                test_vectors.extend(batch)

        # Insert vectors in a single transaction
        print("Inserting vectors...")
        txn_id = None
        with collection.transaction() as txn:
            txn.batch_upsert_vectors(test_vectors)
            txn_id = txn.transaction_id
        print("Vectors inserted successfully")

        print("Waiting for transaction to complete")
        final_status, success = txn_id.poll_completion(
            client,
            collection_name,
            txn_id,
            target_status="complete",
            max_attempts=10,
            sleep_interval=2,
        )

        if not success:
            print(
                f"Transaction did not complete successfully. Final status: {final_status}"
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")

        # Test search performance
        print("\nTesting search performance...")
        search_start_time = time.time()
        query_vector = test_vectors[0]["dense_values"]
        results = collection.search.dense(
            query_vector=query_vector, top_k=5, return_raw_text=True
        )
        search_end_time = time.time()
        print(f"Search time: {search_end_time - search_start_time:.4f} seconds")

        print("\nSearch results:")
        for i, result in enumerate(results.get("results", []), 1):
            # Fetch the full vector for each result
            vector = collection.vectors.get(result["id"])
            print(f"\nResult {i} (score: {result['score']:.4f}):")
            print(format_vector(vector))

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
    test_performance()
