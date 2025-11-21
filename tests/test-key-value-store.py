import os
import time
import urllib3
import getpass
import json
from dotenv import load_dotenv
from cosdata import Client
import requests

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

client = None
host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")

def create_session():
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
        "key_value_options": {"enabled": True},
        "config": {"max_vectors": None, "replication_factor": None},
        "store_raw_text": False,
    }
    response = requests.post(url, headers=client._get_headers(), data=json.dumps(data), verify=client.verify_ssl)
    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to create collection: {response.text}")

def create_index(name):
    url = f"{client.base_url}/collections/{name}/indexes/key-value"
    data = {"name": name}
    response = requests.post(url, headers=client._get_headers(), data=json.dumps(data), verify=client.verify_ssl)
    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to create TF-IDF index: {response.text}")

def create_transaction(name):
    url = f"{client.base_url}/collections/{name}/transactions"
    response = requests.post(url, headers=client._get_headers(), verify=client.verify_ssl)
    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to create transaction: {response.text}")
    return response.json()["transaction_id"]

def commit_transaction(name, txn_id):
    url = f"{client.base_url}/collections/{name}/transactions/{txn_id}/commit"
    response = requests.post(url, headers=client._get_headers(), verify=client.verify_ssl)
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to commit transaction: {response.text}")

def abort_transaction(name, txn_id):
    url = f"{client.base_url}/collections/{name}/transactions/{txn_id}/abort"
    response = requests.post(url, headers=client._get_headers(), verify=client.verify_ssl)
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to abort transaction: {response.text}")

def upsert(name, txn_id, key: str, value: str):
    url = f"{client.base_url}/collections/{name}/transactions/{txn_id}/upsert"
    data = {"vectors": [{"id": key, "bytes": list(value.encode("utf-8"))}]}
    response = requests.post(url, headers=client._get_headers(), data=json.dumps(data), verify=client.verify_ssl)
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to upsert vectors: {response.text}")

def bulk_upsert(name, txn_id, items: list[tuple[str, str]]):
    """Upsert multiple key-value pairs in a single request"""
    url = f"{client.base_url}/collections/{name}/transactions/{txn_id}/upsert"
    data = {"vectors": [{"id": k, "bytes": list(v.encode("utf-8"))} for k, v in items]}
    response = requests.post(url, headers=client._get_headers(), data=json.dumps(data), verify=client.verify_ssl)
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to bulk upsert: {response.text}")

def lookup(name, key):
    url = f"{client.base_url}/collections/{name}/search/key-value"
    data = {"key": key}
    response = requests.post(url, headers=client._get_headers(), data=json.dumps(data), verify=client.verify_ssl)
    if response.status_code != 200:
        raise Exception(f"Failed to search: {response.text}")
    return bytes(response.json()["bytes"]).decode("utf-8")

def delete_collection(name):
    url = f"{client.base_url}/collections/{name}"
    response = requests.delete(url, headers=client._get_headers(), verify=client.verify_ssl)
    if response.status_code not in [200, 204, 404]:
        raise Exception(f"Failed to delete collection: {response.text}")

# ============== TEST UTILITIES ==============

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  ✓ {name}")

    def fail(self, name, expected, actual):
        self.failed += 1
        self.errors.append((name, expected, actual))
        print(f"  ✗ {name}: expected {expected!r}, got {actual!r}")

    def error(self, name, exc):
        self.failed += 1
        self.errors.append((name, "no error", str(exc)))
        print(f"  ✗ {name}: {exc}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} passed")
        if self.failed:
            print(f"Failed tests: {self.failed}")

results = TestResult()

def run_in_txn(name, fn):
    """Helper to run operations in a transaction with auto-commit/abort"""
    txn_id = create_transaction(name)
    try:
        fn(txn_id)
        commit_transaction(name, txn_id)
    except Exception:
        abort_transaction(name, txn_id)
        raise

    time.sleep(1)

# ============== TESTS ==============

def test_basic_upsert_lookup(name):
    """Test basic insert and retrieval"""
    print("\n[Test: Basic Upsert/Lookup]")
    
    def do_upsert(txn_id):
        upsert(name, txn_id, "key1", "value1")
        upsert(name, txn_id, "key2", "value2")
    
    run_in_txn(name, do_upsert)
    
    v1 = lookup(name, "key1")
    v2 = lookup(name, "key2")
    
    if v1 == "value1":
        results.ok("key1 lookup")
    else:
        results.fail("key1 lookup", "value1", v1)
    
    if v2 == "value2":
        results.ok("key2 lookup")
    else:
        results.fail("key2 lookup", "value2", v2)

def test_unicode_content(name):
    """Test Unicode characters in keys and values"""
    print("\n[Test: Unicode Content]")
    
    test_cases = [
        ("emoji_key_🔑", "emoji_value_🎉"),
        ("chinese_key_中文", "chinese_value_你好世界"),
        ("arabic_مفتاح", "arabic_قيمة"),
        ("mixed_αβγ_123", "Ω_ñ_ü_∞"),
    ]
    
    run_in_txn(name, lambda t: [upsert(name, t, k, v) for k, v in test_cases])
    
    for key, expected in test_cases:
        actual = lookup(name, key)
        if actual == expected:
            results.ok(f"unicode: {key[:20]}")
        else:
            results.fail(f"unicode: {key[:20]}", expected, actual)

def test_special_characters(name):
    """Test special characters and escape sequences"""
    print("\n[Test: Special Characters]")
    
    test_cases = [
        ("newline_key", "line1\nline2\nline3"),
        ("tab_key", "col1\tcol2\tcol3"),
        ("quote_key", 'single\' and "double" quotes'),
        ("backslash_key", "path\\to\\file"),
        ("null_char_key", "before\x00after"),
        ("mixed_special", "a\tb\nc\\d\"e'f"),
    ]
    
    run_in_txn(name, lambda t: [upsert(name, t, k, v) for k, v in test_cases])
    
    for key, expected in test_cases:
        actual = lookup(name, key)
        if actual == expected:
            results.ok(f"special: {key}")
        else:
            results.fail(f"special: {key}", expected, actual)

def test_large_value(name):
    """Test storing large values"""
    print("\n[Test: Large Values]")
    
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    
    for size in sizes:
        key = f"large_{size}"
        value = "x" * size
        
        try:
            run_in_txn(name, lambda t, k=key, v=value: upsert(name, t, k, v))
            actual = lookup(name, key)
            if len(actual) == size and actual == value:
                results.ok(f"large value ({size:,} bytes)")
            else:
                results.fail(f"large value ({size:,} bytes)", f"len={size}", f"len={len(actual)}")
        except Exception as e:
            results.error(f"large value ({size:,} bytes)", e)

def test_json_value(name):
    """Test storing JSON as value"""
    print("\n[Test: JSON Values]")
    
    test_data = {
        "simple_json": {"name": "test", "value": 123},
        "nested_json": {"level1": {"level2": {"level3": [1, 2, 3]}}},
        "array_json": [1, "two", 3.0, None, True, False],
        "complex_json": {
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "metadata": {"created": "2024-01-01", "version": 1.5}
        }
    }
    
    def do_upsert(txn_id):
        for k, v in test_data.items():
            upsert(name, txn_id, k, json.dumps(v))
    
    run_in_txn(name, do_upsert)
    
    for key, expected in test_data.items():
        actual_str = lookup(name, key)
        actual = json.loads(actual_str)
        if actual == expected:
            results.ok(f"json: {key}")
        else:
            results.fail(f"json: {key}", expected, actual)

def test_bulk_upsert(name):
    """Test bulk upsert of multiple items"""
    print("\n[Test: Bulk Upsert]")
    
    items = [(f"bulk_{i}", f"value_{i}") for i in range(100)]
    
    run_in_txn(name, lambda t: bulk_upsert(name, t, items))
    
    # Verify random samples
    sample_indices = [0, 25, 50, 75, 99]
    all_ok = True
    for i in sample_indices:
        key, expected = items[i]
        actual = lookup(name, key)
        if actual != expected:
            all_ok = False
            results.fail(f"bulk item {i}", expected, actual)
    
    if all_ok:
        results.ok("bulk upsert (100 items, sampled)")

def test_key_not_found(name):
    """Test lookup of non-existent key"""
    print("\n[Test: Key Not Found]")
    
    try:
        lookup(name, "nonexistent_key_xyz_123")
        results.fail("nonexistent key", "exception", "no exception")
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e) or "Failed" in str(e):
            results.ok("nonexistent key raises exception")
        else:
            results.ok(f"nonexistent key raises exception: {e}")

def test_transaction_abort(name):
    """Test that aborted transactions don't persist"""
    print("\n[Test: Transaction Abort]")
    
    txn_id = create_transaction(name)
    upsert(name, txn_id, "abort_test_key", "should_not_exist")
    abort_transaction(name, txn_id)
    
    try:
        lookup(name, "abort_test_key")
        results.fail("aborted key", "not found", "found")
    except Exception:
        results.ok("aborted transaction not persisted")

def test_multiple_transactions(name):
    """Test multiple sequential transactions"""
    print("\n[Test: Multiple Transactions]")
    
    for i in range(5):
        run_in_txn(name, lambda t, i=i: upsert(name, t, f"txn_{i}", f"value_{i}"))
    
    all_ok = True
    for i in range(5):
        actual = lookup(name, f"txn_{i}")
        if actual != f"value_{i}":
            all_ok = False
            results.fail(f"txn_{i}", f"value_{i}", actual)
    
    if all_ok:
        results.ok("multiple sequential transactions")

def test_binary_like_content(name):
    """Test values that look like binary data"""
    print("\n[Test: Binary-like Content]")
    
    # Base64-encoded data
    import base64
    binary_data = bytes(range(256))
    b64_value = base64.b64encode(binary_data).decode('ascii')
    
    run_in_txn(name, lambda t: upsert(name, t, "base64_key", b64_value))
    
    actual = lookup(name, "base64_key")
    if actual == b64_value:
        results.ok("base64 encoded data")
    else:
        results.fail("base64 encoded data", b64_value[:50], actual[:50])

def test_key_variations(name):
    """Test various key formats"""
    print("\n[Test: Key Variations]")
    
    keys = [
        ("short", "a"),
        ("numeric", "12345"),
        ("uuid", "550e8400-e29b-41d4-a716-446655440000"),
        ("path_like", "users/123/profile"),
        ("dot_separated", "com.example.app.setting"),
        ("long_key", "k" * 200),
    ]
    
    run_in_txn(name, lambda t: [upsert(name, t, k, f"val_{n}") for n, k in keys])
    
    for label, key in keys:
        actual = lookup(name, key)
        expected = f"val_{label}"
        if actual == expected:
            results.ok(f"key: {label}")
        else:
            results.fail(f"key: {label}", expected, actual)

# ============== MAIN ==============

def main():
    create_session()
    name = "kv_store_comprehensive_test"
    
    create_db(name, "Comprehensive KV store test")
    create_index(name)
    
    print("="*50)
    print("Running Comprehensive Key-Value Store Tests")
    print("="*50)
    
    # Run all tests
    test_basic_upsert_lookup(name)
    test_unicode_content(name)
    test_special_characters(name)
    test_large_value(name)
    test_json_value(name)
    test_bulk_upsert(name)
    test_key_not_found(name)
    test_transaction_abort(name)
    test_multiple_transactions(name)
    test_binary_like_content(name)
    test_key_variations(name)
    
    results.summary()
    
    # Cleanup
    print("\nCleaning up...")
    delete_collection(name)
    print("Done.")

if __name__ == "__main__":
    main()
