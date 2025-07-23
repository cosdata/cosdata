"""Script for testing the metadata filtering functionality

How it works?

1. Create a collection with a metadata schema

2. Create index explicitly

3. Insert 100 randomly generated vectors with randomly generated
metadata fields

4. From the vectors that were inserted, randomly select some vectors
and generate search query such that it either strongly matches or
strongly doesn't match the particular vector.

"""

# TODO: use cosdata client sdk instead of direct HTTP requests
import argparse
import os
import getpass
import requests
import json
import random
import numpy as np
from functools import partial
import time
from dotenv import load_dotenv
from utils import poll_transaction_completion

# Load environment variables from .env file
load_dotenv()


token = None
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"


class SimpleClient:
    """Simple client wrapper to work with the polling utility"""

    def __init__(self, host, token):
        self.host = host
        self.token = token

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-type": "application/json",
        }


def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}


def create_session():
    url = f"{host}/auth/create-session"
    # Get password from .env file or prompt securely
    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter admin password: ")

    # Get username from .env file or use default
    username = os.getenv("COSDATA_USERNAME", "admin")

    data = {"username": username, "password": password}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    session = response.json()
    global token
    token = session["access_token"]
    return token


def create_db(vcoll):
    url = f"{base_url}/collections"
    data = vcoll.create_db_payload()
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def create_explicit_index(name):
    data = {
        "name": name,  # Name of the index
        "distance_metric_type": "cosine",  # Type of distance metric (e.g., cosine, euclidean)
        "quantization": {"type": "auto", "properties": {"sample_threshold": 100}},
        "index": {
            "type": "hnsw",
            "properties": {
                "num_layers": 9,
                "max_cache_size": 1000,
                "ef_construction": 512,
                "ef_search": 256,
                "neighbors_count": 32,
                "level_0_neighbors_count": 64,
            },
        },
    }
    response = requests.post(
        f"{base_url}/collections/{name}/indexes/dense",
        headers=generate_headers(),
        data=json.dumps(data),
        verify=False,
    )

    if response.status_code not in [200, 201, 204]:  # Allow 201
        raise Exception(
            f"Failed to create index: {response.status_code} ({response.text})"
        )
    # Return empty dict for 201/204 as body is {} or empty
    return response.json() if response.status_code == 200 and response.text else {}


def prob(percent):
    assert percent <= 100
    return random.random() <= (percent / 100)


def gen_vectors(num, vcoll):
    for i in range(num):
        vid = i + 1
        values = np.random.uniform(-1, 1, vcoll.num_dimensions).tolist()
        metadata = vcoll.gen_metadata_fields()
        yield {"id": str(vid), "values": values, "metadata": metadata}


def create_transaction(collection_name: str) -> str:
    url = f"{base_url}/collections/{collection_name}/transactions"
    response = requests.post(url, headers=generate_headers(), verify=False)
    result = response.json()
    return result["transaction_id"]


def commit_transaction(collection_name: str, txn_id: str):
    url = f"{base_url}/collections/{collection_name}/transactions/{txn_id}/commit"
    response = requests.post(url, headers=generate_headers(), verify=False)
    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to commit transaction: {response.status_code}")


def insert_vectors(coll_id, vectors):
    vec_index = {}
    headers = generate_headers()
    txn_id = create_transaction(coll_id)
    url = f"{base_url}/collections/{coll_id}/transactions/{txn_id}/vectors"
    for vector in vectors:
        data = {
            "id": str(vector["id"]),
            "dense_values": vector["values"],
            "metadata": vector["metadata"],
        }
        print(data)
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code != 200:
            print("Response error:", resp.text)
            resp.raise_for_status()
        vec_index[vector["id"]] = vector
    commit_transaction(coll_id, txn_id)
    return vec_index, txn_id


def search_ann(coll_name, query_vec, metadata_filter):
    url = f"{base_url}/collections/{coll_name}/search/dense"
    headers = generate_headers()
    data = {"query_vector": query_vec, "filter": metadata_filter, "top_k": 5}
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 200:
        print("Response error:", resp.text)
        resp.raise_for_status()
    return resp.json()


def get_vector_by_id(coll_id, vector_id):
    url = f"{base_url}/collections/{coll_id}/vectors/{vector_id}"
    headers = generate_headers()
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print("Response error:", resp.text)
        resp.raise_for_status()
    return resp.json()


def nested_lookup(m, k1, k2):
    v1 = m.get(k1)
    if v1:
        v2 = v1.get(k2)
        if v2:
            return v2
    return None


def search_results_to_matching_ids(data):
    matches = data.get("results", [])
    return set([m.get("id") for m in matches if m.get("id") is not None])


def must_match(vec_id, results):
    matching_ids = search_results_to_matching_ids(results)
    return (f"must_match({vec_id})", vec_id in matching_ids)


def must_not_match(vec_id, results):
    matching_ids = search_results_to_matching_ids(results)
    return (f"must_not_match({vec_id})", vec_id not in matching_ids)


def make_predicate_json(field_name, operator_str, field_value):
    """Creates the JSON dictionary for a Predicate."""
    return {
        "field_name": field_name,
        "field_value": field_value,
        "operator": operator_str,
    }


def is_filter_json(field_name, operator_str, field_value):
    """Creates the JSON dictionary for a Filter::Is variant."""
    return {"Is": make_predicate_json(field_name, operator_str, field_value)}


def and_filter_json(predicates_list):
    """Creates the JSON dictionary for a Filter::And variant."""
    return {"And": predicates_list}


def check_search_results(vector_index, vector_db_name):
    max_queries_each = 5
    queries_eq_age = []
    queries_eq_color = []
    queries_eq_age_color = []
    queries_ne_age = []
    queries_ne_color = []
    queries_ne_age_color = []
    for vec in vector_index.values():
        m_age = nested_lookup(vec, "metadata", "age")
        m_color = nested_lookup(vec, "metadata", "color")
        qvec = vec["values"]
        if m_age and m_color:
            if len(queries_eq_age_color) <= max_queries_each:
                queries_eq_age_color.append(
                    {
                        "vec": qvec,
                        "filter": and_filter_json(
                            [
                                make_predicate_json("age", "Equal", m_age),
                                make_predicate_json("color", "Equal", m_color),
                            ]
                        ),
                        "test": partial(must_match, vec["id"]),
                    }
                )
                continue
            elif len(queries_ne_age_color) <= max_queries_each:
                queries_ne_age_color.append(
                    {
                        "vec": qvec,
                        "filter": and_filter_json(
                            [
                                make_predicate_json("age", "NotEqual", m_age),
                                make_predicate_json("color", "NotEqual", m_color),
                            ]
                        ),
                        "test": partial(must_not_match, vec["id"]),
                    }
                )
                continue
        if m_age:
            if len(queries_eq_age) <= max_queries_each:
                queries_eq_age.append(
                    {
                        "vec": qvec,
                        "filter": is_filter_json("age", "Equal", m_age),
                        "test": partial(must_match, vec["id"]),
                    }
                )
                continue
            elif len(queries_ne_age) <= max_queries_each:
                queries_ne_age.append(
                    {
                        "vec": qvec,
                        "filter": is_filter_json("age", "NotEqual", m_age),
                        "test": partial(must_not_match, vec["id"]),
                    }
                )
                continue
        if m_color:
            if len(queries_eq_color) <= max_queries_each:
                queries_eq_color.append(
                    {
                        "vec": qvec,
                        "filter": is_filter_json("color", "Equal", m_color),
                        "test": partial(must_match, vec["id"]),
                    }
                )
                continue
            elif len(queries_ne_color) <= max_queries_each:
                queries_ne_color.append(
                    {
                        "vec": qvec,
                        "filter": is_filter_json("color", "NotEqual", m_color),
                        "test": partial(must_not_match, vec["id"]),
                    }
                )
                continue

    queries = (
        queries_eq_age
        + queries_eq_color
        + queries_eq_age_color
        + queries_ne_age
        + queries_ne_color
        + queries_ne_age_color
    )


def search_and_compare(vector_db_name, test_cases):
    for tc in test_cases:
        print("Filter:", tc["filter"])
        res = search_ann(vector_db_name, tc["vec"], tc["filter"])
        print("Result:", res)
        test_func = tc["test"]
        (test_name, test_output) = test_func(res)
        test_result = "passed" if test_output else "failed"
        print(f"Test: {test_name} ({test_result})")
        print("-" * 100)


class VectorCollection:
    def __init__(self, name, description, num_dimensions, metadata_schema=None):
        self.name = name
        self.description = description
        self.num_dimensions = num_dimensions
        self.metadata_schema = metadata_schema

    def create_db_payload(self):
        return {
            "name": self.name,
            "description": self.description,
            "dense_vector": {
                "enabled": True,
                "dimension": self.num_dimensions,
            },
            "sparse_vector": {"enabled": False},
            "tf_idf_options": {"enabled": False},
            "metadata_schema": self.metadata_schema,
            "config": {"max_vectors": None, "replication_factor": None},
        }

    def gen_metadata_fields(self):
        if self.metadata_schema is None:
            return None
        if prob(2):
            return None
        mfields = {}
        for field in self.metadata_schema["fields"]:
            # if prob(70):
            mfields[field["name"]] = random.choice(field["values"])
        return mfields

    def test_cases(self):
        raise NotImplementedError


class VecWithAgeColor(VectorCollection):
    def __init__(self, name, num_dimensions):
        age = {
            "name": "age",
            "values": [x for x in range(25, 50)],
        }
        color = {"name": "color", "values": ["red", "blue", "green"]}
        fields = [age, color]
        conds = [
            {"op": "and", "field_names": ["age", "color"]},
            {"op": "or", "field_names": ["age", "color"]},
        ]
        metadata_schema = {
            "fields": fields,
            "supported_conditions": conds,
        }
        super().__init__(
            name=name,
            num_dimensions=num_dimensions,
            description="vec with fields age, color",
            metadata_schema=metadata_schema,
        )

    def test_cases(self, vector_index):
        max_queries_each = 5
        queries_eq_age = []
        queries_eq_color = []
        queries_eq_age_color = []
        queries_ne_age = []
        queries_ne_color = []
        queries_ne_age_color = []
        for vec in vector_index.values():
            m_age = nested_lookup(vec, "metadata", "age")
            m_color = nested_lookup(vec, "metadata", "color")
            qvec = vec["values"]
            if m_age and m_color:
                if len(queries_eq_age_color) <= max_queries_each:
                    queries_eq_age_color.append(
                        {
                            "vec": qvec,
                            "filter": and_filter_json(
                                [
                                    make_predicate_json("age", "Equal", m_age),
                                    make_predicate_json("color", "Equal", m_color),
                                ]
                            ),
                            "test": partial(must_match, vec["id"]),
                        }
                    )
                    continue
                elif len(queries_ne_age_color) <= max_queries_each:
                    queries_ne_age_color.append(
                        {
                            "vec": qvec,
                            "filter": and_filter_json(
                                [
                                    make_predicate_json("age", "NotEqual", m_age),
                                    make_predicate_json("color", "NotEqual", m_color),
                                ]
                            ),
                            "test": partial(must_not_match, vec["id"]),
                        }
                    )
                    continue
            if m_age:
                if len(queries_eq_age) <= max_queries_each:
                    queries_eq_age.append(
                        {
                            "vec": qvec,
                            "filter": is_filter_json("age", "Equal", m_age),
                            "test": partial(must_match, vec["id"]),
                        }
                    )
                    continue
                elif len(queries_ne_age) <= max_queries_each:
                    queries_ne_age.append(
                        {
                            "vec": qvec,
                            "filter": is_filter_json("age", "NotEqual", m_age),
                            "test": partial(must_not_match, vec["id"]),
                        }
                    )
                    continue
            if m_color:
                if len(queries_eq_color) <= max_queries_each:
                    queries_eq_color.append(
                        {
                            "vec": qvec,
                            "filter": is_filter_json("color", "Equal", m_color),
                            "test": partial(must_match, vec["id"]),
                        }
                    )
                    continue
                elif len(queries_ne_color) <= max_queries_each:
                    queries_ne_color.append(
                        {
                            "vec": qvec,
                            "filter": is_filter_json("color", "NotEqual", m_color),
                            "test": partial(must_not_match, vec["id"]),
                        }
                    )
                    continue
        return (
            queries_eq_age
            + queries_eq_color
            + queries_eq_age_color
            + queries_ne_age
            + queries_ne_color
            + queries_ne_age_color
        )


class VecWithBinaryStatus(VectorCollection):
    def __init__(self, name, num_dimensions):
        status = {
            "name": "status",
            "values": ["todo", "done"],
        }
        fields = [status]
        metadata_schema = {
            "fields": fields,
            "supported_conditions": [],
        }
        super().__init__(
            name=name,
            num_dimensions=num_dimensions,
            description="vec with binary status",
            metadata_schema=metadata_schema,
        )

    def test_cases(self, vector_index):
        # Filter vectors to only include those with valid status metadata
        valid_vecs = []
        for vec in vector_index.values():
            status = nested_lookup(vec, "metadata", "status")
            if status is not None:
                valid_vecs.append(vec)

        if len(valid_vecs) < 5:
            print(f"Warning: Only {len(valid_vecs)} vectors have valid status metadata")
            # If we don't have enough vectors with metadata, just return a simple test case
            if len(valid_vecs) == 0:
                return []
            vecs = valid_vecs
        else:
            vecs = random.choices(valid_vecs, k=5)

        get_status = lambda x: nested_lookup(x, "metadata", "status")

        test_cases = []

        # With no filter - use first vector regardless of metadata
        all_vecs = list(vector_index.values())
        if all_vecs:
            test_cases.append(
                {
                    "vec": all_vecs[0]["values"],
                    "filter": None,
                    "test": partial(must_match, all_vecs[0]["id"]),
                }
            )

        # Add test cases only for vectors with valid status
        for i, vec in enumerate(
            vecs[:4]
        ):  # Use up to 4 vectors for the remaining tests
            status = get_status(vec)
            if status is not None:
                if i % 2 == 0:
                    # Test equal filter
                    test_cases.append(
                        {
                            "vec": vec["values"],
                            "filter": is_filter_json("status", "Equal", status),
                            "test": partial(must_match, vec["id"]),
                        }
                    )
                else:
                    # Test not equal filter
                    test_cases.append(
                        {
                            "vec": vec["values"],
                            "filter": is_filter_json("status", "NotEqual", status),
                            "test": partial(must_not_match, vec["id"]),
                        }
                    )

        return test_cases


def cmd_insert_and_check(ctx, args):
    db_name = ctx["vector_db_name"]
    # @NOTE: Replace the following with `VecWithAgeColor` to test that
    # example. Similarly more such classes can be implemented to test
    # with different examples.
    vcoll = VecWithBinaryStatus(db_name, args.num_dims)
    print("Creating a new db/collection")
    create_collection_response = create_db(vcoll)
    coll_id = create_collection_response["id"]
    print("  Create Collection(DB) Response:", create_collection_response)

    print("Creating index explicitly")
    create_index_response = create_explicit_index(db_name)
    print("  Create index response:", create_index_response)

    num_to_insert = args.num_vecs
    if num_to_insert > 0:
        vectors = gen_vectors(num_to_insert, vcoll)
        print(f"Inserting {num_to_insert} vectors")
        vidx, txn_id = insert_vectors(coll_id, vectors)

        # Create client wrapper for polling
        client = SimpleClient(host, ctx["token"])

        # Wait for transaction to complete using polling
        print("Waiting for transaction to complete...")
        final_status, success = txn_id.poll_completion(
            target_status="complete",
            max_attempts=30,
            sleep_interval=2,
        )

        if not success:
            print(
                f"Warning: Transaction may not have completed. Final status: {final_status}"
            )
            print("Proceeding with search queries anyway...")

        tcs = vcoll.test_cases(vidx)

        print("Running search queries")
        search_and_compare(db_name, tcs)


def cmd_query(ctx, args):
    vec_id = args.vector_id
    vec = get_vector_by_id(ctx["vector_db_name"], vec_id)
    values = vec["dense_values"]
    print("Vector metadata:", vec["metadata"])
    metadata_filter = json.loads(args.metadata_filter) if args.metadata_filter else None
    res = search_ann(ctx["vector_db_name"], values, metadata_filter)
    print("Result:", res)


def init_ctx():
    print("Creating session")
    token = create_session()
    # Get host from environment or use default
    global host
    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    global base_url
    base_url = f"{host}/vectordb"

    return {
        "vector_db_name": "testdb",
        # "max_val": 1.0
        # "min_val": -1.0
        # perturbation_degree: 0.95,
        "token": token,
    }


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_insert = subparsers.add_parser("insert")
    parser_insert.add_argument("-n", "--num-vecs", type=int, default=100)
    parser_insert.add_argument("-d", "--num-dims", type=int, default=1024)
    parser_insert.set_defaults(func=cmd_insert_and_check)

    parser_query = subparsers.add_parser("query")
    parser_query.add_argument("vector_id", type=int)
    parser_query.add_argument("-m", "--metadata-filter", type=str, default=None)
    parser_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    ctx = init_ctx()
    args.func(ctx, args)


if __name__ == "__main__":
    main()
