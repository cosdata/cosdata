# import requests
# import json
# import numpy as np
import pandas as pd

# import time
from concurrent.futures import ThreadPoolExecutor
import urllib3
import os
# import math
# import random
# from test_dataset import load_or_generate_brute_force_results, datasets, prompt_and_get_dataset_metadata, run_matching_tests, read_dataset_from_parquet

import importlib.util

spec = importlib.util.spec_from_file_location("test_dataset", "tests/test-dataset.py")
test_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_dataset)

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def read_dataset_from_parquet(dataset_name, ids):
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

    vectors = []

    for row in dataset:
        vector = test_dataset.pre_process_vector(row[0], row[1])
        if vector["id"] in ids:
            vectors.append(vector)

    return vectors


if __name__ == "__main__":
    session_response = test_dataset.create_session()
    print("Session Response:", session_response)

    vector_db_name = "testdb"

    dataset_name, dataset_metadata = test_dataset.prompt_and_get_dataset_metadata()

    brute_force_results = test_dataset.load_or_generate_brute_force_results(
        dataset_name
    )
    print(f"Loaded {len(brute_force_results)} pre-computed brute force results")

    # matches_test_vectors = []

    test_vec_ids = {test_vec["query_id"] for test_vec in brute_force_results}
    # test_vec_ids = {149}
    vectors = read_dataset_from_parquet(dataset_name, test_vec_ids)

    # for vector in vectors:
    #     if (vector["id"] in test_vec_ids):
    #         vectors.append(vector)

    test_dataset.run_matching_tests(vectors, vector_db_name, brute_force_results)
