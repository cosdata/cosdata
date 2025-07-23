#!/usr/bin/env python

import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import getpass
from tqdm.auto import tqdm
from cosdata import Client
import requests
import json
import urllib3
import gzip
import shutil
from pathlib import Path
import sys
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = None
host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
base_url = f"{host}/vectordb"

# Dataset configurations with correct column names
datasets = {
    "cohere-wiki-embedding-100k": {
        "id": "id",
        "embeddings": "emb",
        "size": 100_000,
        "dimension": 768,
        "dataset_id": "ashraq/cohere-wiki-embedding-100k",
        "description": "Cohere Wikipedia embeddings dataset (100k vectors)",
    },
    "million-text-embeddings": {
        "id": None,
        "embeddings": "embedding",
        "size": 1_000_000,
        "dimension": 768,
        "dataset_id": "Sreenath/million-text-embeddings",
        "description": "Million text embeddings dataset",
    },
    "arxiv-embeddings-ada-002": {
        "id": "id",
        "embeddings": "embeddings",
        "size": 1_000_000,
        "dataset_id": "Tychos/arxiv-embeddings-ada-002",
        "description": "ArXiv paper embeddings using Ada-002",
    },
    "dbpedia-entities-openai-1M": {
        "id": "id",
        "embeddings": "openai",
        "size": 1_000_000,
        "dimension": 1536,
        "dataset_id": "KShivendu/dbpedia-entities-openai-1M",
        "description": "DBpedia entity embeddings using OpenAI",
    },
    "glove-100": {
        "id": "id",
        "embeddings": "emb",
        "size": 1_200_000,
        "dimension": 100,
        "dataset_id": "open-vdb/glove-100-angular",
        "description": "GloVe 100-dimensional word embeddings",
    },
}


def download_huggingface_dataset(dataset_id, destination):
    """Download a dataset from Hugging Face using their API"""
    # First, get the parquet file URL
    parquet_url = (
        f"https://huggingface.co/api/datasets/{dataset_id}/parquet/default/train"
    )
    response = requests.get(parquet_url)
    response.raise_for_status()
    parquet_info = response.json()

    if not parquet_info or not isinstance(parquet_info, list) or not parquet_info:
        raise Exception(f"No parquet files found for dataset {dataset_id}")

    # Get the first parquet file URL
    parquet_file_url = parquet_info[0]["url"]

    # Download the parquet file
    print(f"Downloading parquet file from {parquet_file_url}")
    response = requests.get(parquet_file_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(destination, "wb") as f,
        tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def download_file(url, destination):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(destination, "wb") as f,
            tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        raise


def prepare_glove_dataset(dataset_dir):
    """Prepare GloVe dataset by converting it to parquet format"""
    zip_path = os.path.join(dataset_dir, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print("Downloading GloVe dataset...")
        download_file(datasets["glove-100"]["url"], zip_path)

    print("Extracting GloVe dataset...")
    with gzip.open(zip_path, "rb") as f_in:
        with open(os.path.join(dataset_dir, "glove.6B.100d.txt"), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("Converting GloVe to parquet format...")
    # Read the GloVe text file
    data = []
    with open(
        os.path.join(dataset_dir, "glove.6B.100d.txt"), "r", encoding="utf-8"
    ) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = [float(x) for x in values[1:]]
            data.append({"id": word, "embeddings": vector})

    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(dataset_dir, "test0.parquet"))

    # Clean up temporary files
    os.remove(os.path.join(dataset_dir, "glove.6B.100d.txt"))
    os.remove(zip_path)


def prepare_dataset(dataset_name):
    """Download and prepare a dataset"""
    dataset_dir = os.path.join("datasets", dataset_name)
    dataset_config = datasets[dataset_name]
    parquet_path = os.path.join(dataset_dir, "test0.parquet")

    print(f"Downloading {dataset_name}...")
    prepare_huggingface_dataset(dataset_config["dataset_id"], parquet_path)

    print(f"Dataset {dataset_name} is ready at {dataset_dir}")


def prepare_huggingface_dataset(dataset_id, destination):
    """Download and prepare a dataset from Hugging Face"""
    print(f"Loading dataset {dataset_id} from Hugging Face...")

    try:
        # Try loading with default config first
        dataset = load_dataset(dataset_id, split="train")
    except ValueError as e:
        if "Config name is missing" in str(e):
            # If config is required, try with 'train' config
            print("Config name required, using 'train' config...")
            dataset = load_dataset(dataset_id, "train", split="train")
        else:
            raise

    # Convert to pandas DataFrame
    df = dataset.to_pandas()

    # Print dataset structure
    print("\nDataset Structure:")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First row sample: {df.iloc[0].to_dict()}")
    print(f"Number of rows: {len(df)}")

    # Ensure the required columns exist
    if "id" not in df.columns:
        df["id"] = range(len(df))

    if "dense_values" not in df.columns:
        # Find the embedding column
        embedding_cols = [
            col
            for col in df.columns
            if "emb" in col.lower()
            or "embedding" in col.lower()
            or "vector" in col.lower()
        ]
        if not embedding_cols:
            raise ValueError(
                f"No embedding column found in dataset {dataset_id}. Available columns: {df.columns.tolist()}"
            )
        df["dense_values"] = df[embedding_cols[0]]

    # Save as parquet
    print(f"\nSaving dataset to {destination}")
    df.to_parquet(destination)

    print(f"Dataset saved successfully with {len(df)} rows")


def ensure_dataset_available(dataset_name):
    """Ensure a dataset is downloaded and ready to use"""
    # Create datasets directory if it doesn't exist
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        print(f"Created datasets directory at {os.path.abspath(datasets_dir)}")

    # Create dataset-specific directory if it doesn't exist
    dataset_dir = os.path.join(datasets_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created dataset directory at {os.path.abspath(dataset_dir)}")

    # Check if dataset files exist
    if not any(f.endswith(".parquet") for f in os.listdir(dataset_dir)):
        print(f"Dataset {dataset_name} not found. Downloading...")
        prepare_dataset(dataset_name)
    else:
        print(f"Dataset {dataset_name} is already available at {dataset_dir}")


def prompt_and_get_dataset_metadata():
    print("\nChoose a dataset to test with:")
    dataset_names = list(datasets.keys())
    for i, key in enumerate(dataset_names):
        print(f"{i + 1}) {key} - {datasets[key]['description']}")
    print()
    dataset_idx = int(input("Select: ")) - 1
    dataset_name = dataset_names[dataset_idx]
    print(f"Reading {dataset_name} ...")

    # Ask for test mode
    print("\nChoose test mode:")
    print("1) Quick test (smaller dataset, faster)")
    print("2) Full test (complete dataset, slower)")
    test_mode = int(input("Select (1 or 2): "))

    # Ensure dataset is available
    ensure_dataset_available(dataset_name)

    dataset = datasets[dataset_name]
    return (dataset_name, dataset, test_mode == 1)


def cosine_sim_matrix(A, B):
    """Compute cosine similarity between each row in A and each row in B"""
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    A[np.isnan(A)] = 0
    B[np.isnan(B)] = 0
    return np.dot(A, B.T)


def generate_brute_force_results(dataset_name, vector_list, quick_test=False):
    """Generate brute force results for similarity comparison"""
    total_vectors = len(vector_list)
    print(f"Total vectors for brute force computation: {total_vectors}")

    # Pre-process vectors into NumPy arrays
    ids = [v["id"] for v in vector_list]
    vectors = np.array([v["dense_values"] for v in vector_list], dtype=np.float64)

    # Use smaller sample for quick test
    num_test_vectors = 10 if quick_test else 100
    np.random.seed(42)
    test_indices = np.random.choice(total_vectors, num_test_vectors, replace=False)
    test_ids = [ids[i] for i in test_indices]
    test_vectors = vectors[test_indices]

    print("Computing brute force similarities...")
    sim_matrix = cosine_sim_matrix(test_vectors, vectors)

    results = []
    for i, sims in enumerate(sim_matrix):
        if i % 10 == 0:
            print(
                f"Processing query vector {i + 1}/{num_test_vectors}, ID: {test_ids[i]}"
            )

        top5_idx = np.argpartition(-sims, 5)[:5]
        top5_sorted = top5_idx[np.argsort(-sims[top5_idx])]

        top_ids = [ids[j] for j in top5_sorted]
        top_sims = [sims[j] for j in top5_sorted]

        results.append(
            {
                "query_id": test_ids[i],
                **{f"top{j + 1}_id": top_ids[j] for j in range(5)},
                **{f"top{j + 1}_sim": top_sims[j] for j in range(5)},
            }
        )

    print(f"\nGenerated brute force results for {len(results)} queries")
    return results


def load_or_generate_brute_force_results(dataset_name, quick_test=False):
    """Always generate brute force results"""
    print("Generating brute force results...")
    vectors = read_dataset_from_parquet(dataset_name)

    # Limit vectors for quick test
    if quick_test:
        print("Quick test mode: Using first 1000 vectors for brute force computation")
        vectors = vectors[:1000]

    results = generate_brute_force_results(dataset_name, vectors, quick_test=quick_test)
    del vectors
    return results


def pre_process_vector(id, values):
    corrected_values = [float(v) for v in values]
    return {
        "id": str(id),  # Keep as string for server compatibility
        "dense_values": corrected_values,
        "document_id": f"doc_{id // 10}",  # Group vectors into documents
    }


def read_dataset_from_parquet(dataset_name):
    """Read dataset from parquet files in the datasets directory"""
    metadata = datasets[dataset_name]
    dfs = []

    # Check if datasets directory exists
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        print(f"Created datasets directory at {os.path.abspath(datasets_dir)}")

    # Check if dataset directory exists
    dataset_dir = os.path.join(datasets_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created dataset directory at {os.path.abspath(dataset_dir)}")
        raise FileNotFoundError(
            f"No parquet files found for dataset {dataset_name}. "
            f"Please place your parquet files in: {os.path.abspath(dataset_dir)}"
        )

    path = os.path.join(dataset_dir, "test0.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No parquet files found for dataset {dataset_name}. "
            f"Please place your parquet files in: {os.path.abspath(dataset_dir)}"
        )

    while os.path.exists(path):
        dfs.append(pd.read_parquet(path))
        count = len(dfs)
        path = os.path.join(dataset_dir, f"test{count}.parquet")

    if not dfs:
        raise FileNotFoundError(
            f"No parquet files found for dataset {dataset_name}. "
            f"Please place your parquet files in: {os.path.abspath(dataset_dir)}"
        )

    df = pd.concat(dfs, ignore_index=True)

    print("Pre-processing ...")
    dataset = (
        df[[metadata["id"], metadata["embeddings"]]].values.tolist()
        if metadata["id"] is not None
        else list(enumerate(row[0] for row in df[[metadata["embeddings"]]].values))
    )

    vectors = []
    print("Dimension:", len(dataset[0][1]))
    print("Size: ", len(dataset))

    for row in dataset:
        vector = pre_process_vector(row[0], row[1])
        vectors.append(vector)

    return vectors


def read_single_parquet_file(path, dataset_name, file_index, base_id, quick_test=False):
    """Read and process a single parquet file"""
    try:
        print(f"Reading file {file_index}: {path}")
        df = pd.read_parquet(path)

        # For quick test, only read first 1000 rows
        if quick_test:
            df = df.head(1000)
            print("Quick test mode: Limiting to first 1000 vectors")

        dataset_config = datasets[dataset_name]
        id_col = dataset_config["id"]
        emb_col = dataset_config["embeddings"]

        if id_col is not None:
            dataset = df[[id_col, emb_col]].values.tolist()
        else:
            dataset = list(
                enumerate((row[0] for row in df[[emb_col]].values), start=base_id)
            )

        vectors = []
        for row in dataset:
            vector = pre_process_vector(row[0], row[1])
            vectors.append(vector)

        print(f"Processed {len(vectors)} vectors from file {file_index}")
        return vectors

    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return None


def process_vectors_batch(vectors, collection, batch_size):
    """Process a batch of vectors and insert them into the database"""
    try:
        # Ensure all IDs are strings
        for vector in vectors:
            vector["id"] = str(vector["id"])
        txn_id = None
        with collection.transaction() as txn:
            txn.batch_upsert_vectors(vectors)
            txn_id = txn.transaction_id
        return txn_id
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None


def process_parquet_files(
    dataset_name,
    collection,
    brute_force_results,
    batch_size=100,
    matches_sample_size=100,
    rps_sample_size=100000,
    quick_test=False,
):
    """
    Process parquet files asynchronously and upsert vectors to the server.
    Collects random samples of vectors for testing purposes.
    """
    # Adjust sample sizes for quick test
    if quick_test:
        matches_sample_size = 10
        rps_sample_size = 1000
        batch_size = 50

    file_count = 0
    total_vectors_inserted = 0
    total_insertion_time = 0
    id_counter = 0
    matches_test_vectors = []
    rps_test_vectors = []

    # For quick test, use the first 10 vectors as test vectors
    if quick_test:
        matches_test_vector_ids_set = set(
            str(result["query_id"]) for result in brute_force_results[:10]
        )
    else:
        matches_test_vector_ids_set = set(
            str(result["query_id"]) for result in brute_force_results
        )

    def get_next_file_path(count):
        return os.path.join("datasets", dataset_name, f"test{count}.parquet")

    start_time = time.time()

    txn_ids = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        current_path = get_next_file_path(file_count)
        if not os.path.exists(current_path):
            print(
                f"No parquet files found in {os.path.abspath(os.path.dirname(current_path))}"
            )
            return matches_test_vectors, rps_test_vectors

        future = executor.submit(
            read_single_parquet_file,
            current_path,
            dataset_name,
            file_count,
            id_counter,
            quick_test,
        )

        while True:
            try:
                vectors = future.result()
                if not vectors:
                    break

                id_counter += len(vectors)
                file_count += 1
                next_path = get_next_file_path(file_count)
                if os.path.exists(next_path):
                    future = executor.submit(
                        read_single_parquet_file,
                        next_path,
                        dataset_name,
                        file_count,
                        id_counter,
                        quick_test,
                    )

                if len(matches_test_vectors) < matches_sample_size:
                    for vector in vectors:
                        if vector["id"] in matches_test_vector_ids_set:
                            matches_test_vectors.append(vector)

                if len(rps_test_vectors) < rps_sample_size:
                    sample_size = min(
                        rps_sample_size - len(rps_test_vectors), len(vectors)
                    )
                    rps_test_vectors.extend(random.sample(vectors, sample_size))

                insertion_start = time.time()
                txn_id = process_vectors_batch(vectors, collection, batch_size)
                print(f"Transaction id for batch upsert: {txn_id}")
                if txn_id is not None:
                    txn_ids.append(txn_id)
                insertion_end = time.time()
                insertion_time = insertion_end - insertion_start
                total_insertion_time += insertion_time
                total_vectors_inserted += len(vectors)

                print(f"\nProcessing file: {current_path}")
                print(f"File {file_count - 1} statistics:")
                print(f"Vectors inserted: {len(vectors)}")
                print(
                    f"Matches test vectors collected: {len(matches_test_vectors)}/{matches_sample_size}"
                )
                print(
                    f"RPS test vectors collected: {len(rps_test_vectors)}/{rps_sample_size}"
                )

                del vectors
                current_path = next_path

                if not os.path.exists(next_path):
                    break

            except Exception as e:
                print(f"Error processing file {current_path}: {e}")
                break

    end_time = time.time()
    total_time = end_time - start_time

    print("Waiting for transactions to complete")
    for txn_id in txn_ids:
        print(f"Polling transaction {txn_id}...")
        final_status, success = txn_id.poll_completion(
            target_status="complete", max_attempts=10, sleep_interval=30
        )

        if not success:
            print(
                f"Transaction {txn_id} did not complete successfully. Final status: {final_status}"
            )
        else:
            print(f"Transaction {txn_id} completed successfully")

    print("\nProcessing complete!")
    print(f"Total files processed: {file_count}")
    print(f"Total vectors inserted: {total_vectors_inserted}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total insertion time: {total_insertion_time:.2f} seconds")
    print(
        f"Average insertion time per vector: {(total_insertion_time / total_vectors_inserted) * 1000:.2f} ms"
    )
    print(f"Final matches test vectors collected: {len(matches_test_vectors)}")
    print(f"Final RPS test vectors collected: {len(rps_test_vectors)}")

    return matches_test_vectors, rps_test_vectors


def run_matching_tests(test_vectors, collection, brute_force_results):
    """Run matching accuracy tests and measure query latencies"""
    print(f"\nStarting similarity search tests with {len(test_vectors)} queries...")

    total_recall = 0
    total_queries = 0
    latencies = []

    for i, query_vec in enumerate(test_vectors):
        try:
            test_vec = next(
                test_vec
                for test_vec in brute_force_results
                if str(query_vec["id"])
                == str(test_vec["query_id"])  # Compare as strings
            )

            # Measure query latency
            query_start_time = time.time()
            results = collection.search.dense(
                query_vector=query_vec["dense_values"], top_k=5, return_raw_text=True
            )
            query_end_time = time.time()
            query_latency = (query_end_time - query_start_time) * 1000  # Convert to ms
            latencies.append(query_latency)

            if "results" in results:
                query_id = test_vec["query_id"]
                print(f"\nQuery {i + 1} (Vector ID: {query_id}):")
                print(f"Query latency: {query_latency:.2f} ms")

                server_top5 = [
                    str(match["id"]) for match in results["results"][:5]
                ]  # Convert to string
                print("Server top 5:", server_top5)

                brute_force_top5 = [
                    str(test_vec[f"top{j}_id"]) for j in range(1, 6)
                ]  # Convert to string
                print("Brute force top 5:", brute_force_top5)

                matches = sum(1 for id in server_top5 if id in brute_force_top5)
                recall = (matches / 5) * 100
                total_recall += recall
                total_queries += 1

                print(f"Recall@5 for this query: {recall}% ({matches}/5 matches)")

                time.sleep(0.1)

        except Exception as e:
            print(f"Error in query {i + 1}: {e}")

    # Calculate and display latency statistics
    if latencies:
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = latencies[int(len(latencies) * 0.5)]
        p90_latency = latencies[int(len(latencies) * 0.9)]
        p95_latency = latencies[int(len(latencies) * 0.95)]
        min_latency = min(latencies)
        max_latency = max(latencies)

        print("\nLatency Statistics (ms):")
        print(f"Average: {avg_latency:.2f}")
        print(f"p50: {p50_latency:.2f}")
        print(f"p90: {p90_latency:.2f}")
        print(f"p95: {p95_latency:.2f}")
        print(f"Min: {min_latency:.2f}")
        print(f"Max: {max_latency:.2f}")

    if total_queries > 0:
        average_recall = total_recall / total_queries
        print("\nFinal Matching Results:")
        print(f"Average Recall@5: {average_recall:.2f}%")
    else:
        print("No valid queries completed")


def run_rps_tests(rps_test_vectors, collection, batch_size=100):
    """Run RPS (Requests Per Second) tests"""
    print(f"Using {len(rps_test_vectors)} different test vectors for RPS testing")

    start_time_rps = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(0, len(rps_test_vectors), batch_size):
            batch = rps_test_vectors[i : i + batch_size]
            futures.append(executor.submit(batch_ann_search, collection, batch))

        for future in as_completed(futures):
            try:
                future.result()
                results.append(True)
            except Exception as e:
                print(f"Error in RPS test: {e}")
                results.append(False)

    end_time_rps = time.time()
    actual_duration = end_time_rps - start_time_rps

    successful_requests = sum(results) * batch_size
    failed_requests = (len(results) * batch_size) - successful_requests
    total_requests = len(results) * batch_size
    rps = successful_requests / actual_duration

    print("\nRPS Test Results:")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_requests}")
    print(f"Test Duration: {actual_duration:.2f} seconds")
    print(f"Requests Per Second (RPS): {rps:.2f}")
    print(f"Success Rate: {(successful_requests / total_requests * 100):.2f}%")


def batch_ann_search(collection, vectors):
    """Perform batch ANN search using the Client class"""
    try:
        # Format the vectors for batch search
        queries = [{"vector": vector["dense_values"], "top_k": 5} for vector in vectors]

        # Use the collection's search method
        results = collection.search.batch_dense(queries)
        return results
    except Exception as e:
        print(f"Error in batch search: {e}")
        raise


if __name__ == "__main__":
    # Get password from .env file or prompt securely
    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter your database password: ")

    # Initialize client
    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    username = os.getenv("COSDATA_USERNAME", "admin")
    client = Client(host=host, username=username, password=password, verify=False)

    # Configuration
    batch_size = 100
    num_match_test_vectors = 100  # Number of vectors to test
    num_rps_test_vectors = 100_000
    rps_batch_size = 100

    dataset_name, dataset_metadata, quick_test = prompt_and_get_dataset_metadata()

    # Create dynamic collection name based on dataset and test mode
    test_mode_suffix = "quick" if quick_test else "full"
    vector_db_name = f"{dataset_name}-{test_mode_suffix}"
    print(f"Using collection name: {vector_db_name}")

    # Load or generate brute force results
    brute_force_results = load_or_generate_brute_force_results(dataset_name, quick_test)
    print(f"Loaded {len(brute_force_results)} pre-computed brute force results")

    collection = None
    try:
        # Create collection
        print("Creating collection...")
        collection = client.create_collection(
            name=vector_db_name,
            dimension=dataset_metadata["dimension"],
            description=f"Test collection for {dataset_metadata['description']} - {test_mode_suffix} mode",
        )

        # Create index
        print("Creating index...")
        collection.create_index(
            distance_metric="cosine",
            num_layers=10,
            max_cache_size=1000,
            ef_construction=64,
            ef_search=64,
            neighbors_count=16,
            level_0_neighbors_count=32,
        )

        matches_test_vectors, rps_test_vectors = process_parquet_files(
            dataset_name,
            collection,
            brute_force_results,
            batch_size,
            num_match_test_vectors,
            num_rps_test_vectors,
            quick_test,
        )

        run_matching_tests(matches_test_vectors, collection, brute_force_results)

        run_rps_tests(rps_test_vectors, collection, rps_batch_size)

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        # Cleanup
        if collection is not None:
            try:
                # Ask user if they want to delete the collection
                print(f"\nCollection '{vector_db_name}' was created for testing.")
                delete_choice = (
                    input("Do you want to delete the test collection? (y/N): ")
                    .lower()
                    .strip()
                )

                if delete_choice in ["y", "yes"]:
                    collection.delete()
                    print("Test collection deleted")
                else:
                    print(
                        f"Test collection '{vector_db_name}' preserved for future use"
                    )
            except Exception as e:
                print(f"Error during cleanup: {e}")
