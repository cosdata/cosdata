import pandas as pd
import getpass
from concurrent.futures import ThreadPoolExecutor
import urllib3
import os
from cosdata import Client
import importlib.util
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

spec = importlib.util.spec_from_file_location("test_dataset", "test-dataset.py")
test_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_dataset)

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def read_dataset_from_parquet(dataset_name, ids, num_rps_test_vectors):
    metadata = test_dataset.datasets[dataset_name]
    dfs = []

    path = f"datasets/{dataset_name}/test0.parquet"

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        while os.path.exists(path):
            futures.append(executor.submit(pd.read_parquet, path))
            count = len(futures)
            path = f"datasets/{dataset_name}/test{count}.parquet"

        for future in futures:
            dfs.append(future.result())

    df = pd.concat(dfs, ignore_index=True)

    print("Pre-processing ...")
    dataset = (
        df[[metadata["id"], metadata["embeddings"]]].values.tolist()
        if metadata["id"] is not None
        else list(enumerate(row[0] for row in df[[metadata["embeddings"]]].values))
    )

    matches_test_vectors = []
    rps_test_vectors = []

    for row in dataset:
        vector = test_dataset.pre_process_vector(row[0], row[1])
        if vector["id"] in ids:
            matches_test_vectors.append(vector)
        if len(rps_test_vectors) < num_rps_test_vectors:
            rps_test_vectors.append(vector)

    return matches_test_vectors, rps_test_vectors


if __name__ == "__main__":
    # Get password from .env file or prompt securely
    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter admin password: ")

    # Get host and username from environment variables with fallbacks
    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    username = os.getenv("COSDATA_USERNAME", "admin")

    client = Client(host=host, username=username, password=password, verify=False)

    num_rps_test_vectors = 100_000
    rps_batch_size = 100

    dataset_name, dataset_metadata, quick_test = (
        test_dataset.prompt_and_get_dataset_metadata()
    )

    # Create dynamic collection name based on dataset and test mode (same as test-dataset.py)
    test_mode_suffix = "quick" if quick_test else "full"
    vector_db_name = f"{dataset_name}-{test_mode_suffix}"
    print(f"Using collection name: {vector_db_name}")

    collection = client.get_collection(name=vector_db_name)

    brute_force_results = test_dataset.load_or_generate_brute_force_results(
        dataset_name
    )
    brute_force_results = (
        brute_force_results[:10] if quick_test else brute_force_results
    )
    print(f"Loaded {len(brute_force_results)} pre-computed brute force results")

    # matches_test_vectors = []

    test_vec_ids = {str(test_vec["query_id"]) for test_vec in brute_force_results}
    # test_vec_ids = {149}
    matches_test_vectors, rps_test_vectors = read_dataset_from_parquet(
        dataset_name, test_vec_ids, num_rps_test_vectors
    )

    test_dataset.run_matching_tests(
        matches_test_vectors, collection, brute_force_results
    )

    test_dataset.run_rps_tests(rps_test_vectors, collection, rps_batch_size)
