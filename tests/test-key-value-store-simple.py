import os
import urllib3
import getpass
import json
from dotenv import load_dotenv
from cosdata import Client
import requests

# Load environment variables from .env file
load_dotenv()

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
client = None
host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")

def create_session():
    """Initialize the cosdata client"""
    # Use environment variable from .env file if available, otherwise prompt
    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter admin password: ")

    username = os.getenv("COSDATA_USERNAME", "admin")

    global client
    client = Client(host=host, username=username, password=password, verify=False)
    return client


def create_db(name: str, description: str | None = None):
    client._ensure_session()

    url = f"{client.base_url}/collections"
    data = {
        "name": name,
        "description": description,
        "key_value_options": { "enabled": True },
        "config": {"max_vectors": None, "replication_factor": None},
        "store_raw_text": False,
    }

    response = requests.post(
        url,
        headers=client._get_headers(),
        data=json.dumps(data),
        verify=client.verify_ssl,
    )

    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to create collection: {response.text}")


def create_index(name):
    url = f"{client.base_url}/collections/{name}/indexes/key-value"
    data = {"name": name}

    response = requests.post(
        url,
        headers=client._get_headers(),
        data=json.dumps(data),
        verify=client.verify_ssl,
    )

    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to create TF-IDF index: {response.text}")

def create_transaction(name):
    url = f"{client.base_url}/collections/{name}/transactions"
    response = requests.post(
        url,
        headers=client._get_headers(),
        verify=client.verify_ssl,
    )

    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to create transaction: {response.text}")

    result = response.json()
    transaction_id = result["transaction_id"]
    return transaction_id

def commit_transaction(name, txn_id):
    url = f"{client.base_url}/collections/{name}/transactions/{txn_id}/commit"
    response = requests.post(
        url,
        headers=client._get_headers(),
        verify=client.verify_ssl,
    )

    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to commit transaction: {response.text}")

def abort_transaction(name, txn_id):
    url = f"{client.base_url}/collections/{name}/transactions/{txn_id}/abort"
    response = requests.post(
        url,
        headers=client._get_headers(),
        verify=client.verify_ssl,
    )

    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to abort transaction: {response.text}")

def upsert(name, txn_id, key: str, value: str):
    url = f"{client.base_url}/collections/{name}/transactions/{txn_id}/upsert"
    data = {"vectors": [
        {
            "id": key,
            "bytes": list(value.encode("utf-8"))
        }
    ]}

    response = requests.post(
        url,
        headers=client._get_headers(),
        data=json.dumps(data),
        verify=client.verify_ssl,
    )

    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to upsert vectors: {response.text}")


def lookup(name, key):
    url = f"{client.base_url}/collections/{name}/search/key-value"
    data = {
        "key": key
    }

    response = requests.post(
        url,
        headers=client._get_headers(),
        data=json.dumps(data),
        verify=client.verify_ssl,
    )

    if response.status_code != 200:
        raise Exception(f"Failed to search dense vector: {response.text}")

    return bytes(response.json()["bytes"]).decode("utf-8")

def main():
    # Initialize client session
    create_session()

    name = "key_value_store_test"

    # Create collection and index
    create_db(name)
    create_index(name)

    # Create transaction and index vectors
    print("Creating transaction and indexing vectors...")

    txn_id = create_transaction(name)

    try:
        upsert(name, txn_id, "a", "Hello")
        upsert(name, txn_id, "b", "World")
        commit_transaction(name, txn_id)
    except Exception:
        abort_transaction(name, txn_id)
        

    a = lookup(name, "a")
    b = lookup(name, "b")

    print(f"Expected: \"Hello\", Found: \"{a}\"")
    print(f"Expected: \"World\", Found: \"{b}\"")
    

if __name__ == "__main__":
    main()
