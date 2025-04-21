import beir.util
import time
import beir
from pathlib import Path
from tqdm.auto import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import os
import urllib3
import getpass
import requests
import json
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
# import subprocess
# import signal

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
TOKEN = None
HOST = "http://127.0.0.1:8443"
BASE_URL = f"{HOST}/vectordb"

type Document = Tuple[int, str]
type IdsMap = Dict[int, str]


def merge_cqa_dupstack(data_path: str, verbose: bool = False):
    data_path = Path(data_path)
    dataset = data_path.name
    assert dataset == "cqadupstack", "Dataset must be CQADupStack"

    # check if corpus.jsonl exists
    corpus_path = data_path / "corpus.jsonl"
    if not corpus_path.exists():
        # combine all the corpus files into one
        # corpus files are located under cqadupstack/<name>/corpus.jsonl
        corpus_files = list(data_path.glob("*/corpus.jsonl"))
        with open(corpus_path, "w") as f:
            for file in tqdm(corpus_files, desc="Merging Corpus", disable=not verbose):
                # get the name of the corpus
                corpus_name = file.parent.name

                with open(file, "r") as f2:
                    for line in tqdm(
                        f2,
                        desc=f"Merging {corpus_name} Corpus",
                        leave=False,
                        disable=not verbose,
                    ):
                        # first, read with ujson
                        line = json.loads(line)
                        # add the corpus name to _id
                        line["_id"] = f"{corpus_name}_{line['_id']}"
                        # write back to file
                        f.write(json.dumps(line))
                        f.write("\n")

    # now, do the same for queries.jsonl
    queries_path = data_path / "queries.jsonl"
    if not queries_path.exists():
        queries_files = list(data_path.glob("*/queries.jsonl"))
        with open(queries_path, "w") as f:
            for file in tqdm(
                queries_files, desc="Merging Queries", disable=not verbose
            ):
                # get the name of the corpus
                corpus_name = file.parent.name

                with open(file, "r") as f2:
                    for line in tqdm(
                        f2,
                        desc=f"Merging {corpus_name} Queries",
                        leave=False,
                        disable=not verbose,
                    ):
                        # first, read with ujson
                        line = json.loads(line)
                        # add the corpus name to _id
                        line["_id"] = f"{corpus_name}_{line['_id']}"
                        # write back to file
                        f.write(json.dumps(line))
                        f.write("\n")

    # now, do the same for qrels/test.tsv
    qrels_path = data_path / "qrels" / "test.tsv"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)

    if not qrels_path.exists():
        qrels_files = list(data_path.glob("*/qrels/test.tsv"))
        with open(qrels_path, "w") as f:
            # First, write the columns: query-id	corpus-id	score
            f.write("query-id\tcorpus-id\tscore\n")
            for file in tqdm(qrels_files, desc="Merging Qrels", disable=not verbose):
                # get the name of the corpus
                corpus_name = file.parent.parent.name
                with open(file, "r") as f2:
                    # skip first line
                    next(f2)

                    for line in tqdm(
                        f2,
                        desc=f"Merging {corpus_name} Qrels",
                        leave=False,
                        disable=not verbose,
                    ):
                        # since it's a tsv, split by tab
                        qid, cid, score = line.strip().split("\t")
                        # add the corpus name to _id
                        qid = f"{corpus_name}_{qid}"
                        cid = f"{corpus_name}_{cid}"
                        # write back to file
                        f.write(f"{qid}\t{cid}\t{score}\n")


def generate_headers():
    return {"Authorization": f"Bearer {TOKEN}", "Content-type": "application/json"}


def create_session():
    url = f"{HOST}/auth/create-session"
    if "ADMIN_PASSWORD" in os.environ:
        password = os.environ["ADMIN_PASSWORD"]
    else:
        password = getpass.getpass("Enter admin password: ")

    data = {"username": "admin", "password": password}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    session = response.json()
    global TOKEN
    TOKEN = session["access_token"]
    return TOKEN


def create_db(name: str, description: str | None = None):
    url = f"{BASE_URL}/collections"
    data = {
        "name": name,
        "description": description,
        "dense_vector": {
            "enabled": False,
            "dimension": 1,
        },
        "sparse_vector": {"enabled": False},
        "tf_idf_options": {"enabled": True},
        "metadata_schema": None,
        "config": {"max_vectors": None, "replication_factor": None},
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def create_index(name: str, k1: float, b: float):
    data = {
        "name": name,
        "sample_threshold": 1000,
        "store_raw_text": False,
        "k1": k1,
        "b": b,
    }
    response = requests.post(
        f"{BASE_URL}/collections/{name}/indexes/tf-idf",
        headers=generate_headers(),
        data=json.dumps(data),
        verify=False,
    )
    return response.json()


def create_transaction(collection_name: str) -> str:
    url = f"{BASE_URL}/collections/{collection_name}/transactions"
    response = requests.post(
        url, headers=generate_headers(), verify=False
    )
    result = response.json()
    return result["transaction_id"]


def upsert_vectors(collection_name: str, txn_id: str, documents: List[Document]):
    documents = [{"id": document[0], "text": document[1]} for document in documents]

    url = f"{BASE_URL}/collections/{collection_name}/transactions/{txn_id}/upsert"
    data = {"vectors": documents}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code not in [200, 204]:
        raise Exception(
            f"Failed to create vector: {response.status_code} - {response.text}"
        )


def index(
    collection_name: str, txn_id: str, documents: List[Document], batch_size: int = 100
) -> float:
    start = time.time()
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start : batch_start + batch_size]
            futures.append(
                executor.submit(upsert_vectors, collection_name, txn_id, batch)
            )

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            try:
                future.result()
            except Exception as e:
                print(f"Batch {i + 1} failed: {e}")
    end = time.time()
    return end - start


def commit_transaction(collection_name: str, txn_id: str):
    url = f"{BASE_URL}/collections/{collection_name}/transactions/{txn_id}/commit"
    response = requests.post(
        url, headers=generate_headers(), verify=False
    )
    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to commit transaction: {response.status_code}")


def search_document(
    collection_name: str, query: str, top_k: int
) -> List[Tuple[int, float]]:
    url = f"{BASE_URL}/collections/{collection_name}/search/tf-idf"
    data = {
        "query": query,
        "top_k": top_k,
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return [(result["id"], result["score"]) for result in response.json()["results"]]


def search_documents(
    collection_name: str,
    queries: List[Tuple[str, str]],
    ids_map: Dict[int, str],
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    results = {}
    for query in tqdm(queries, desc="Searching vectors"):
        qid = query[0]
        query_results = search_document(collection_name, query[1], top_k)
        results[qid] = {ids_map[doc_id]: score for doc_id, score in query_results}
    return results


def batch_search_documents(collection_name: str, queries: List[str], top_k: int):
    url = f"{BASE_URL}/collections/{collection_name}/search/batch-tf-idf"
    data = {
        "queries": queries,
        "top_k": top_k,
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code not in [200, 204]:
        print(f"Error response ({response.status_code}): {response.text}")


def preprocess_corpus(
    corpus: Dict[str, Dict[str, str]],
) -> Tuple[List[Document], IdsMap]:
    ids_map = {}
    documents = []
    for idx, (k, v) in tqdm(
        enumerate(corpus.items()), total=len(corpus), desc="Mapping corpus IDs"
    ):
        ids_map[idx] = k
        documents.append((idx, v["title"] + " " + v["text"]))
    return documents, ids_map


def run_qps_test(
    collection_name: str, qps_queries: List[str], batch_size: int, top_k: int
) -> Tuple[float, float, int, int, int]:
    start_time = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(0, len(qps_queries), batch_size):
            batch = qps_queries[i : i + batch_size]

            futures.append(
                executor.submit(batch_search_documents, collection_name, batch, top_k)
            )

        for future in as_completed(futures):
            try:
                future.result()
                results.append(True)
            except Exception as e:
                print(f"Error in QPS test: {e}")
                results.append(False)
    end_time = time.perf_counter()
    duration = end_time - start_time
    total_queries = len(results) * batch_size
    successful_queries = sum(results) * batch_size
    failed_queries = total_queries - successful_queries
    qps = successful_queries / duration

    print("QPS Test Results:")
    print(f"Total Queries: {total_queries}")
    print(f"Successful Queries: {successful_queries}")
    print(f"Failed Queries: {failed_queries}")
    print(f"Test Duration: {duration:.2f} seconds")
    print(f"Queries Per Second (QPS): {qps:.2f}")
    print(f"Success Rate: {(successful_queries / total_queries * 100):.2f}%")

    return (qps, duration, successful_queries, failed_queries, total_queries)


def run_latency_test(
    collection_name: str, queries: List[str], top_k: int
) -> Tuple[float, float]:
    latencies: List[float] = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []

        def timed_query(query: str) -> Tuple[List[Tuple[int, float]], float]:
            start = time.perf_counter()
            response = search_document(collection_name, query, top_k)
            end = time.perf_counter()
            latency = (end - start) * 1000.0
            return response, latency

        for query in queries:
            futures.append(executor.submit(timed_query, query))
        for future in as_completed(futures):
            try:
                _, latency = future.result()
                latencies.append(latency)
            except Exception as e:
                print(f"Lantency test query failed: {e}")

    latencies.sort()
    p50_latency = latencies[int(len(latencies) * 0.5)]
    p95_latency = latencies[int(len(latencies) * 0.95)]

    print("p50 latency:", p50_latency)
    print("p95 latency:", p95_latency)

    return p50_latency, p95_latency


def main(
    dataset: str,
    top_k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
    batch_size: int = 100,
):
    save_dir = f"sparse_idf_dataset_{dataset}"
    collection_name = f"sparse_idf_{dataset}_db"
    #### Download dataset and unzip the dataset
    base_url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
    )
    data_path = beir.util.download_and_unzip(base_url.format(dataset), save_dir)

    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    loader = GenericDataLoader(data_folder=data_path)
    loader.check(fIn=loader.corpus_file, ext="jsonl")
    loader.check(fIn=loader.query_file, ext="jsonl")
    loader._load_corpus()
    loader._load_queries()
    corpus, qps_queries = loader.corpus, loader.queries
    loader.qrels_file = os.path.join(loader.qrels_folder, split + ".tsv")
    loader._load_qrels()
    queries = [(qid, loader.queries[qid]) for qid in loader.qrels]
    qrels = loader.qrels
    num_docs = len(corpus)
    num_queries = len(queries)

    print("=" * 50)
    print("Dataset: ", dataset)
    print(f"Corpus Size: {num_docs:,}")
    print(f"Queries Size: {num_queries:,}")

    vectors, ids_map = preprocess_corpus(corpus)

    create_session()
    create_db(collection_name)
    create_index(collection_name, k1, b)
    txn_id = create_transaction(collection_name)
    indexing_time = index(collection_name, txn_id, vectors)
    commit_transaction(collection_name, txn_id)
    results = search_documents(collection_name, queries, ids_map, top_k)
    ndcg, _map, recall, _precision = EvaluateRetrieval.evaluate(qrels, results, [1, 10])

    print("NDCG@10:", ndcg["NDCG@10"])
    print("Recall@10:", recall["Recall@10"])
    
    qps_queries = [v for (_, v) in qps_queries.items()][:10_000]
    qps, _, _, _, _ = run_qps_test(collection_name, qps_queries, batch_size, top_k)

    p50_latency, p95_latency = run_latency_test(
        collection_name, qps_queries[: max(len(qps_queries) // 10, 1000)], top_k
    )

    return {
        "dataset": dataset,
        "corpus_size": num_docs,
        "queries_size": num_queries,
        "insertion_time": indexing_time,
        "recall": recall["Recall@10"],
        "qps": qps,
        "ndcg": ndcg["NDCG@10"],
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
    }


# def start_server():
#     # Run the shell command to clean, build, and run the server with taskset
#     cmd = 'rm -rf data && cargo b -r && taskset 0xFF ./target/release/cosdata --admin-key "" --confirmed'
#     return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


# def stop_server(process):
#     # Kill the entire process group
#     os.killpg(os.getpgid(process.pid), signal.SIGTERM)

# def get_memory_usage_gb():
#     result = subprocess.run(['ps', '-p', str(find_pid_by_name("cosdata")), '-o', 'rss='],
#                             stdout=subprocess.PIPE, text=True)
#     return int(result.stdout.strip()) / (1024 * 1024)

# def find_pid_by_name(name):
#     result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, text=True)
#     lines = result.stdout.splitlines()

#     for line in lines:
#         if name in line and 'grep' not in line:
#             parts = line.split()
#             pid = int(parts[1])
#             return pid
#     return None

if __name__ == "__main__":
    main(dataset="arguana")
    # datasets = ["trec-covid", "fiqa", "arguana", "webis-touche2020", "quora", "scidocs", "scifact", "nq", "msmarco", "fever", "climate-fever"]
    # results = []
    # for dataset in datasets:
    #     time.sleep(5)
    #     server_proc = start_server()
    #     time.sleep(5)
    #     try:
    #         result = main(dataset)
    #         memory_usage = get_memory_usage_gb()
    #         result["memory_usage"] = memory_usage
    #     except Exception as e:
    #         print(f"Error with dataset {dataset}: {e}")
    #         stop_server(server_proc)
    #         continue
    #     results.append(result)
    #     with open("results.csv", "w", newline="") as csvfile:
    #         fieldnames = ["dataset", "corpus_size", "queries_size", "insertion_time", "recall", "ndcg", "qps", "p50_latency", "p95_latency", "memory_usage"]
    #         labels = ["Dataset", "Corpus Size", "Queries Size", "Insertion Time (seconds)", "Recall@10", "NDCG@10", "QPS", "p50 Latency (milliseconds)", "p95 Latency (milliseconds)", "Memory Usage (GB)"]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #         writer.writerow({ fieldname: label for fieldname, label in zip(fieldnames, labels) })
    #         writer.writerows(results)
    #     stop_server(server_proc)
