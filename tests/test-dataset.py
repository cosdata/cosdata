#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import getpass
from tqdm.auto import tqdm
from cosdata import Client
import requests
import urllib3
import zipfile
from datasets import load_dataset
from dotenv import load_dotenv
import heapq

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
        "embeddings": "embeddings",
        "size": 1_200_000,
        "dimension": 100,
        "dataset_id": "open-vdb/glove-100-angular",
        "description": "GloVe 100-dimensional word embeddings",
        "url": "https://nlp.stanford.edu/data/glove.6B.zip",
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
    """Prepare GloVe dataset by converting it to parquet format with chunking"""
    zip_path = os.path.join(dataset_dir, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print("Downloading GloVe dataset...")
        download_file(datasets["glove-100"]["url"], zip_path)

    print("Extracting GloVe dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract("glove.6B.100d.txt", dataset_dir)

    print("Converting GloVe to parquet format with chunking...")
    # Read the GloVe text file in chunks
    chunk_size = 200_000
    chunk = []
    file_index = 0
    
    with open(
        os.path.join(dataset_dir, "glove.6B.100d.txt"), "r", encoding="utf-8"
    ) as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = [float(x) for x in values[1:]]
            chunk.append({"id": word, "embeddings": vector})
            
            if (i + 1) % chunk_size == 0:
                # Convert chunk to DataFrame and save
                df = pd.DataFrame(chunk)
                parquet_path = os.path.join(dataset_dir, f"test{file_index}.parquet")
                print(f"Saving chunk {file_index} to {parquet_path}")
                df.to_parquet(parquet_path)
                
                # Clear chunk for next batch
                chunk.clear()
                file_index += 1
        
        # Save any remaining data
        if chunk:
            df = pd.DataFrame(chunk)
            parquet_path = os.path.join(dataset_dir, f"test{file_index}.parquet")
            print(f"Saving remaining chunk to {parquet_path}")
            df.to_parquet(parquet_path)

    # Clean up temporary files
    os.remove(os.path.join(dataset_dir, "glove.6B.100d.txt"))
    os.remove(zip_path)


def prepare_dataset(dataset_name):
    """Download and prepare a dataset"""
    dataset_dir = os.path.join("datasets", dataset_name)
    dataset_config = datasets[dataset_name]

    print(f"Downloading {dataset_name}...")
    if dataset_name == "glove-100":
        prepare_glove_dataset(dataset_dir)
    else:
        prepare_huggingface_dataset(dataset_config["dataset_id"], dataset_dir)

    print(f"Dataset {dataset_name} is ready at {dataset_dir}")


def prepare_huggingface_dataset(dataset_id, base_destination):
    """Download and prepare a dataset from Hugging Face"""
    print(f"Loading dataset {dataset_id} from Hugging Face with streaming...")

    # Find dataset configuration by dataset_id
    dataset_config = None
    for name, config in datasets.items():
        if config["dataset_id"] == dataset_id:
            dataset_config = config
            break
    
    if not dataset_config:
        raise ValueError(f"No configuration found for dataset {dataset_id}")

    try:
        # Stream the dataset and process in chunks
        dataset = load_dataset(dataset_id, split='train', streaming=True)
        chunk_size = 200_000
        chunk = []
        file_index = 0

        for i, example in enumerate(dataset):
            chunk.append(example)
            if (i + 1) % chunk_size == 0:
                # Convert chunk to DataFrame
                chunk_df = pd.DataFrame(chunk)

                # Ensure columns
                if "id" not in chunk_df.columns:
                    offset = chunk_size * file_index
                    chunk_df["id"] = range(offset, offset + len(chunk_df))

                if "dense_values" not in chunk_df.columns:
                    # Use the predefined embedding column from configuration
                    embedding_col = dataset_config["embeddings"]
                    if embedding_col not in chunk_df.columns:
                        raise ValueError(
                            f"Configured embedding column '{embedding_col}' not found in dataset {dataset_id}. Available columns: {chunk_df.columns.tolist()}"
                        )
                    chunk_df["dense_values"] = chunk_df[embedding_col]

                # Save the chunk as parquet
                destination = f"{base_destination}/test{file_index}.parquet"
                print(f"\nSaving chunk to {destination}")
                chunk_df.to_parquet(destination)

                # Clear chunk for next batch
                chunk.clear()
                file_index += 1

        # Save any remaining data
        if chunk:
            chunk_df = pd.DataFrame(chunk)
            destination = f"{base_destination}/test{file_index}.parquet"
            print(f"\nSaving remaining chunk to {destination}")
            chunk_df.to_parquet(destination)

    except Exception as e:
        print(f"Error in processing dataset {dataset_id}: {e}")
        raise


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
    A = normalize(A, norm='l2', axis=1)
    B = normalize(B, norm='l2', axis=1)
    return np.dot(A, B.T)


def generate_brute_force_results(dataset_name, quick_test=False):
    """Generate brute force results by processing dataset in chunks"""
    print("Generating brute force results using chunk processing...")
    
    dataset_config = datasets[dataset_name]
    dataset_dir = os.path.join("datasets", dataset_name)
    
    # Determine sample size
    k = 10 if quick_test else 100
    reservoir = []
    total_vectors = 0
    file_count = 0
    
    # Pass 1: Reservoir sampling to select query vectors
    random.seed(42)
    while True:
        path = os.path.join(dataset_dir, f"test{file_count}.parquet")
        if not os.path.exists(path):
            break
            
        df = pd.read_parquet(path)
        for index, row in df.iterrows():
            total_vectors += 1
            # Get vector data
            id_val = row[dataset_config["id"]] if dataset_config["id"] else index
            embedding = row[dataset_config["embeddings"]]
            vector = pre_process_vector(id_val, embedding)
            
            # Reservoir sampling
            if len(reservoir) < k:
                reservoir.append((vector["id"], vector["dense_values"]))
            else:
                j = random.randint(0, total_vectors-1)
                if j < k:
                    reservoir[j] = (vector["id"], vector["dense_values"])
        file_count += 1

    # Prepare query data
    query_ids = [item[0] for item in reservoir]
    query_vectors = np.array([item[1] for item in reservoir], dtype=np.float32)
    
    # Initialize heaps for each query (min-heap for top-k)
    heaps = [[] for _ in range(len(query_vectors))]
    
    # Pass 2: Process dataset in chunks to find top matches
    file_count = 0
    while True:
        path = os.path.join(dataset_dir, f"test{file_count}.parquet")
        if not os.path.exists(path):
            break
            
        df = pd.read_parquet(path)
        chunk_vectors = []
        chunk_ids = []
        
        for index, row in df.iterrows():
            id_val = row[dataset_config["id"]] if dataset_config["id"] else index
            embedding = row[dataset_config["embeddings"]]
            vector = pre_process_vector(id_val, embedding)
            chunk_vectors.append(vector["dense_values"])
            chunk_ids.append(vector["id"])
        
        # Convert to numpy array
        chunk_vectors_arr = np.array(chunk_vectors, dtype=np.float32)
        
        # Compute cosine similarities
        query_norm = normalize(query_vectors, norm='l2', axis=1)
        chunk_norm = normalize(chunk_vectors_arr, norm='l2', axis=1)
        sim_matrix = np.dot(query_norm, chunk_norm.T)
        
        # Update heaps for each query
        for i in range(len(query_vectors)):
            for j in range(len(chunk_vectors)):
                sim = sim_matrix[i, j]
                if len(heaps[i]) < 5:
                    heapq.heappush(heaps[i], (sim, chunk_ids[j]))
                else:
                    if sim > heaps[i][0][0]:
                        heapq.heapreplace(heaps[i], (sim, chunk_ids[j]))
        file_count += 1
    
    # Prepare results
    results = []
    for i in range(len(query_vectors)):
        top5 = sorted(heaps[i], key=lambda x: x[0], reverse=True)
        result = {"query_id": query_ids[i]}
        for rank, (sim, vec_id) in enumerate(top5):
            result[f"top{rank+1}_id"] = vec_id
            result[f"top{rank+1}_sim"] = sim
        results.append(result)
    
    print(f"Generated brute force results for {len(results)} queries")
    return results


def pre_process_vector(id, values):
    corrected_values = [float(v) for v in values]
    
    result = {
        "id": str(id),  # Keep as string for server compatibility
        "dense_values": corrected_values,
    }
    
    # Only add document_id if id is an integer
    if isinstance(id, int):
        result["document_id"] = f"doc_{id // 10}"
    
    return result


def read_dataset_from_parquet(dataset_name, max_vectors=50000):
    """Read dataset from parquet files with a strict limit to prevent memory issues."""
    metadata = datasets[dataset_name]

    datasets_dir = "datasets"
    dataset_dir = os.path.join(datasets_dir, dataset_name)

    path = os.path.join(dataset_dir, "test0.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No parquet files found for dataset {dataset_name}. "
            f"Please place your parquet files in: {os.path.abspath(dataset_dir)}"
        )

    vectors = []
    file_count = 0
    
    print(f"Reading dataset with limit of {max_vectors} vectors to prevent memory issues...")
    
    while os.path.exists(path) and len(vectors) < max_vectors:
        # Read the full parquet file but limit processing
        df = pd.read_parquet(path)
        
        # Limit rows to prevent memory overload
        rows_to_read = min(len(df), max_vectors - len(vectors))
        df = df.head(rows_to_read)
        
        emb_col = "dense_values" if "dense_values" in df.columns else metadata["embeddings"]
        
        if metadata["id"] is not None:
            id_col = metadata["id"]
            dataset_chunk = df[[id_col, emb_col]].values.tolist()
        else:
            dataset_chunk = list(enumerate(df[emb_col].values))

        for row in dataset_chunk:
            if len(vectors) >= max_vectors:
                break
            vector = pre_process_vector(row[0], row[1])
            vectors.append(vector)

        file_count += 1
        path = os.path.join(dataset_dir, f"test{file_count}.parquet")
        
        # Clear dataframe from memory
        del df

    if not vectors:
        raise ValueError(f"No data found in dataset {dataset_name}")

    print(f"Loaded {len(vectors)} vectors from dataset {dataset_name}")
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
        
        # Handle case where embedding data might be in 'dense_values' column
        if "dense_values" in df.columns:
            emb_col = "dense_values"
        elif emb_col not in df.columns:
            raise ValueError(
                f"Embedding column '{emb_col}' not found in dataset. Available columns: {df.columns.tolist()}"
            )

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
        txn_ = None
        with collection.transaction() as txn:
            txn.batch_upsert_vectors(vectors)
            txn_ = txn
        return txn_
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

    txns = []

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
                txn = process_vectors_batch(vectors, collection, batch_size)
                print(f"Transaction id for batch upsert: {txn.transaction_id}")
                txns.append(txn)
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
    for txn in txns:
        print(f"Polling transaction {txn.transaction_id}...")
        final_status, success = txn.poll_completion(
            target_status="complete", max_attempts=10, sleep_interval=30
        )

        if not success:
            print(
                f"Transaction {txn.transaction_id} did not complete successfully. Final status: {final_status}"
            )
        else:
            print(f"Transaction {txn.transaction_id} completed successfully")

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
    brute_force_results = generate_brute_force_results(dataset_name, quick_test)
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
            ef_construction=128,
            ef_search=64,
            neighbors_count=32,
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
