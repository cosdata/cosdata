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

import argparse
import os
import getpass
import requests
import json
import random
import numpy as np
from functools import partial


token = None
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"


def generate_headers():
    return {
        "Authorization": f"Bearer {token}",
        "Content-type": "application/json"
    }


def create_session():
    url = f"{host}/auth/create-session"
    # Use environment variable if available, otherwise prompt
    if "ADMIN_PASSWORD" in os.environ:
        password = os.environ["ADMIN_PASSWORD"]
    else:
        password = getpass.getpass("Enter admin password: ")

    data = {"username": "admin", "password": password}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    session = response.json()
    global token
    token = session["access_token"]
    return token


def create_db(name, description=None, dimension=1024, metadata_schema=None):
    url = f"{base_url}/collections"
    data = {
        "name": name,
        "description": description,
        "dense_vector": {
            "enabled": True,
            "auto_create_index": False,
            "dimension": dimension,
        },
        "sparse_vector": {"enabled": False, "auto_create_index": False},
        "metadata_schema": metadata_schema,
        "config": {"max_vectors": None, "replication_factor": None},
    }
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
                "num_layers": 7,
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

    return response.json()


def construct_metadata_schema():
    age = {
        "name": "age",
        "values": [x for x in range(25, 50)],
    }
    color = {
        "name": "color",
        "values": ["red", "blue", "green"]
    }
    fields = [age, color]
    conds = [
        {"op": "and", "field_names": ["age", "color"]},
        {"op": "or", "field_names": ["age", "color"]},
    ]
    return {
        "fields": fields,
        "supported_conditions": conds,
    }


def prob(percent):
    assert percent <= 100
    return random.random() <= (percent / 100)


def gen_metadata_fields(metadata_schema):
    if metadata_schema is None:
        return None
    if prob(2):
        return None
    mfields = {}
    for field in metadata_schema["fields"]:
        # if prob(70):
        mfields[field["name"]] = random.choice(field["values"])
    return mfields


def gen_vectors(num, num_dims, metadata_schema):
    for i in range(num):
        vid = i + 1
        values = np.random.uniform(-1, 1, num_dims).tolist()
        metadata = gen_metadata_fields(metadata_schema)
        yield {"id": vid, "values": values, "metadata": metadata}


def insert_vectors(coll_id, vectors):
    vec_index = {}
    url = f"{base_url}/collections/{coll_id}/vectors"
    headers = generate_headers()
    for vector in vectors:
        data = {"dense": vector}
        # print(data)
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code != 200:
            print("Response error:", resp.text)
            resp.raise_for_status()
        vec_index[vector["id"]] = vector
    return vec_index


def search_ann(coll_name, query_vec, metadata_filter):
    url = f"{base_url}/search"
    headers = generate_headers()
    data = {
        "vector_db_name": coll_name,
        "vector": query_vec,
        "filter": metadata_filter,
        "nn_count": 5
    }
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
    matches = data["RespVectorKNN"]["knn"]
    return set([x[0] for x in matches])


def must_match(vec_id, results):
    matching_ids = search_results_to_matching_ids(results)
    return (f"must_match({vec_id})", vec_id in matching_ids)


def must_not_match(vec_id, results):
    matching_ids = search_results_to_matching_ids(results)
    return (f"must_not_match({vec_id})", vec_id not in matching_ids)


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
                queries_eq_age_color.append({
                    "vec": qvec,
                    "filter": {"age": {"$eq": m_age}, "color": {"$eq": m_color}},
                    "test": partial(must_match, vec["id"])
                })
                continue
            elif len(queries_ne_age_color) <= max_queries_each:
                queries_ne_age_color.append({
                    "vec": qvec,
                    "filter": {"age": {"$ne": m_age}, "color": {"$ne": m_color}},
                    "test": partial(must_not_match, vec["id"])
                })
                continue
        if m_age:
            if len(queries_eq_age) <= max_queries_each:
                queries_eq_age.append({
                    "vec": qvec,
                    "filter": {"age": {"$eq": m_age}},
                    "test": partial(must_match, vec["id"])
                })
                continue
            elif len(queries_ne_age) <= max_queries_each:
                queries_ne_age.append({
                    "vec": qvec,
                    "filter": {"age": {"$ne": m_age}},
                    "test": partial(must_not_match, vec["id"])
                })
                continue
        if m_color:
            if len(queries_eq_color) <= max_queries_each:
                queries_eq_color.append({
                    "vec": qvec,
                    "filter": {"color": {"$eq": m_color}},
                    "test": partial(must_match, vec["id"])
                })
                continue
            elif len(queries_ne_color) <= max_queries_each:
                queries_ne_color.append({
                    "vec": qvec,
                    "filter": {"color": {"$ne": m_color}},
                    "test": partial(must_not_match, vec["id"])
                })
                continue

    queries = (
        queries_eq_age +
        queries_eq_color +
        queries_eq_age_color +
        queries_ne_age +
        queries_ne_color +
        queries_ne_age_color
    )

    for query in queries:
        print("Filter:", query["filter"])
        res = search_ann(vector_db_name, query["vec"], query["filter"])
        print("Result:", res)
        test_func = query["test"]
        (test_name, test_output) = test_func(res)
        test_result = "passed" if test_output else "failed"
        print(f"Test: {test_name} ({test_result})")
        print("-" * 100)


def cmd_insert_and_check(ctx, args):
    metadata_schema = construct_metadata_schema()
    db_name = ctx["vector_db_name"]
    print("Creating a new db/collection")
    create_collection_response = create_db(
        name=db_name,
        description="Test collection for vector database",
        dimension=ctx["dimensions"],
        metadata_schema=metadata_schema,
    )
    coll_id = create_collection_response["id"]
    print("  Create Collection(DB) Response:", create_collection_response)

    print("Creating index explicitly")
    create_index_response = create_explicit_index(db_name)
    print("  Create index response:", create_index_response)

    num_to_insert = args.num_vecs
    vectors = gen_vectors(num_to_insert, ctx["dimensions"], metadata_schema)
    print(f"Inserting {num_to_insert} vectors")
    vidx = insert_vectors(coll_id, vectors)

    print("Running search queries")
    check_search_results(vidx, db_name)


def cmd_query(ctx, args):
    vec_id = args.vector_id
    result = get_vector_by_id(ctx["vector_db_name"], vec_id)
    vec = result["Dense"]
    values = vec["values"]
    print("Vector metadata:", vec["metadata"])
    metadata_filter = json.loads(args.metadata_filter) if args.metadata_filter else None
    res = search_ann(ctx["vector_db_name"], values, metadata_filter)
    print("Result:", res)


def init_ctx():
    print("Creating session")
    token = create_session()
    return {
        "vector_db_name": "testdb",
        "dimensions": 1024,
        # "max_val": 1.0
        # "min_val": -1.0
        # perturbation_degree: 0.95,
        "token": token,
    }


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_insert = subparsers.add_parser('insert')
    parser_insert.add_argument('-n', '--num-vecs', type=int, default=100)
    parser_insert.set_defaults(func=cmd_insert_and_check)

    parser_query = subparsers.add_parser('query')
    parser_query.add_argument('vector_id', type=int)
    parser_query.add_argument('-m', '--metadata-filter', type=str, default=None)
    parser_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    ctx = init_ctx()
    args.func(ctx, args)


if __name__ == '__main__':
    main()
