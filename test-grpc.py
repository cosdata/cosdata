import grpc
import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the generated gRPC code
import vector_service_pb2 as vector_service
import vector_service_pb2_grpc as vector_service_grpc

# gRPC server address
grpc_server_address = '[::1]:50051'

def create_grpc_channel():
    """Creates a gRPC channel."""
    return grpc.insecure_channel(grpc_server_address)

def create_collection_with_metadata(stub):
    """Test creating collection with metadata schema"""
    request = vector_service.CreateCollectionRequest(
        name="test_metadata_collection",
        description="Test collection with metadata schema",
        dense_vector=vector_service.DenseVectorOptions(
            dimension=128,
            enabled=True,
            auto_create_index=True
        ),
        sparse_vector=vector_service.SparseVectorOptions(
            enabled=False,
            auto_create_index=False
        ),
        metadata_schema=vector_service.MetadataSchema(
            fields=[
                vector_service.MetadataField(
                    name="category",
                    values=[
                        vector_service.FieldValue(string_value="electronics"),
                        vector_service.FieldValue(string_value="books"),
                        vector_service.FieldValue(string_value="clothing")
                    ]
                ),
                vector_service.MetadataField(
                    name="rating",
                    values=[
                        vector_service.FieldValue(int_value=1),
                        vector_service.FieldValue(int_value=2),
                        vector_service.FieldValue(int_value=3),
                        vector_service.FieldValue(int_value=4),
                        vector_service.FieldValue(int_value=5)
                    ]
                )
            ],
            supported_conditions=[
                vector_service.SupportedCondition(
                    op=vector_service.SupportedCondition.OperationType.AND,
                    field_names=["category", "rating"]
                )
            ]
        ),
        config=vector_service.CollectionConfig(
            max_vectors=1000000,
            replication_factor=1
        )
    )
    return stub.CreateCollection(request)

def create_simple_collection(stub, name="test_collection", dimension=128):
    """Creates a simple collection without metadata schema"""
    request = vector_service.CreateCollectionRequest(
        name=name,
        description="Test collection via gRPC",
        dense_vector=vector_service.DenseVectorOptions(
            dimension=dimension,
            enabled=True,
            auto_create_index=True
        ),
        sparse_vector=vector_service.SparseVectorOptions(
            enabled=False,
            auto_create_index=False
        ),
        config=vector_service.CollectionConfig(
            max_vectors=1000000,
            replication_factor=1
        )
    )
    return stub.CreateCollection(request)

def get_collections(stub):
    """Gets all collections using the gRPC server."""
    try:
        request = vector_service.GetCollectionsRequest()
        return stub.GetCollections(request)
    except grpc.RpcError as e:
        print(f"Error getting collections: {e.details()}")
        raise

def get_collection(stub, collection_id):
    """Gets a specific collection using the gRPC server."""
    try:
        request = vector_service.GetCollectionRequest(id=collection_id)
        return stub.GetCollection(request)
    except grpc.RpcError as e:
        print(f"Error getting collection: {e.details()}")
        raise

def delete_collection(stub, collection_id):
    """Deletes a collection using the gRPC server."""
    try:
        request = vector_service.DeleteCollectionRequest(id=collection_id)
        return stub.DeleteCollection(request)
    except grpc.RpcError as e:
        print(f"Error deleting collection: {e.details()}")
        raise

def test_basic_collection_operations(stub):
    """Test basic collection operations without metadata"""
    print("\n=== Testing Basic Collection Operations ===")
    collection_name = "test_basic_collection"

    print("\n1. Creating basic collection...")
    try:
        response = create_simple_collection(stub, collection_name)
        print(f"Collection created successfully: {response}")
    except grpc.RpcError as e:
        print(f"Failed to create collection: {e.details()}")
        return False

    print("\n2. Getting all collections...")
    try:
        response = get_collections(stub)
        print(f"Got collections successfully: {response}")
        if not any(c.name == collection_name for c in response.collections):
            print("Created collection not found in list!")
            return False
    except grpc.RpcError as e:
        print(f"Failed to get collections: {e.details()}")
        return False

    print("\n3. Getting specific collection...")
    try:
        response = get_collection(stub, collection_name)
        print(f"Got collection successfully: {response}")
    except grpc.RpcError as e:
        print(f"Failed to get collection: {e.details()}")
        return False

    print("\n4. Deleting collection...")
    try:
        delete_collection(stub, collection_name)
        print("Collection deleted successfully")
    except grpc.RpcError as e:
        print(f"Failed to delete collection: {e.details()}")
        return False

    return True

def test_metadata_collection_operations(stub):
    """Test collection operations with metadata schema"""
    print("\n=== Testing Metadata Collection Operations ===")

    print("\n1. Creating collection with metadata schema...")
    try:
        response = create_collection_with_metadata(stub)
        print(f"Collection created successfully with metadata: {response}")
        collection_id = response.id
    except grpc.RpcError as e:
        print(f"Failed to create collection with metadata: {e.details()}")
        return False

    print("\n2. Verifying collection with metadata...")
    try:
        response = get_collection(stub, collection_id)
        print(f"Retrieved collection: {response}")
    except grpc.RpcError as e:
        print(f"Failed to verify collection: {e.details()}")
        return False

    print("\n3. Cleaning up metadata collection...")
    try:
        delete_collection(stub, collection_id)
        print("Collection with metadata deleted successfully")
    except grpc.RpcError as e:
        print(f"Failed to delete collection: {e.details()}")
        return False

    return True

def test_auth_operations(stub):
    """Test authentication operations"""
    print("\n=== Testing Authentication Operations ===")

    print("\n1. Creating session with invalid credentials (should fail)...")
    try:
        request = vector_service.CreateSessionRequest(
            username="test_user",
            password="test_password"
        )
        response = stub.CreateSession(request)
        print("WARNING: Authentication succeeded with test credentials!")
        print(f"Response: {response}")
        return False  # Should not succeed with test credentials
    except grpc.RpcError as e:
        if "Invalid credentials" in e.details():
            print(f"Successfully received expected error: {e.details()}")
            return True  # This is the expected behavior
        else:
            print(f"Received unexpected error: {e.details()}")
            return False




def create_dense_index(stub, collection_id):
    """Creates a dense index for a collection"""
    # Create the request with scalar quantization
    request = vector_service.CreateDenseIndexRequest(
        collection_id=collection_id,
        name="test_dense_index",
        distance_metric_type="cosine",
        # Use scalar quantization
        scalar=vector_service.ScalarQuantization(
            data_type=vector_service.DataType.F32,
            range=vector_service.ValuesRange(min=-1.0, max=1.0)
        ),
        hnsw_params=vector_service.HNSWParams(
            ef_construction=128,
            ef_search=64,
            num_layers=16,
            max_cache_size=1000,
            level_0_neighbors_count=12,
            neighbors_count=8
        )
    )

    return stub.CreateDenseIndex(request)

def create_sparse_index(stub, collection_id):
    """Creates a sparse index for a collection"""
    request = vector_service.CreateSparseIndexRequest(
        collection_id=collection_id,
        name="test_sparse_index",
        quantization=64  # Using 64-bit quantization
    )
    return stub.CreateSparseIndex(request)

def test_index_operations(collection_stub, indexes_stub):
    """Test index creation operations"""
    print("\n=== Testing Index Operations ===")

    # First create a test collection
    collection_name = f"test_index_collection_{int(time.time())}"
    try:
        collection_response = create_simple_collection(collection_stub, collection_name)
        print(f"Created test collection: {collection_response}")
    except grpc.RpcError as e:
        print(f"Failed to create test collection: {e.details()}")
        return False

    # Test dense index creation
    print("\n1. Creating dense index...")
    try:
        create_dense_index(indexes_stub, collection_name)
        print("Dense index created successfully")
    except grpc.RpcError as e:
        print(f"Failed to create dense index: {e.details()}")
        delete_collection(collection_stub, collection_name)  # Clean up
        return False

    # Test sparse index creation
    print("\n2. Creating sparse index...")
    try:
        create_sparse_index(indexes_stub, collection_name)
        print("Sparse index created successfully")
    except grpc.RpcError as e:
        print(f"Failed to create sparse index: {e.details()}")
        delete_collection(collection_stub, collection_name)  # Clean up
        return False

    # Clean up
    print("\n3. Cleaning up test collection...")
    try:
        delete_collection(collection_stub, collection_name)
        print("Test collection deleted successfully")
    except grpc.RpcError as e:
        print(f"Failed to delete test collection: {e.details()}")
        return False

    return True

def run_all_tests():
    """Run all gRPC tests"""
    print("Starting gRPC tests...")

    with create_grpc_channel() as channel:
        # Test collections operations
        collections_stub = vector_service_grpc.CollectionsServiceStub(channel)

        # Test basic operations
        if not test_basic_collection_operations(collections_stub):
            print("\n❌ Basic collection operations tests failed!")
        else:
            print("\n✅ Basic collection operations tests passed!")

        # Test metadata operations
        if not test_metadata_collection_operations(collections_stub):
            print("\n❌ Metadata collection operations tests failed!")
        else:
            print("\n✅ Metadata collection operations tests passed!")

        # Test auth operations
        auth_stub = vector_service_grpc.AuthServiceStub(channel)
        if not test_auth_operations(auth_stub):
            print("\n❌ Authentication operations tests failed!")
        else:
            print("\n✅ Authentication operations tests passed!")

        # Test index operations
        indexes_stub = vector_service_grpc.IndexesServiceStub(channel)
        if not test_index_operations(collections_stub, indexes_stub):
            print("\n❌ Index operations tests failed!")
        else:
            print("\n✅ Index operations tests passed!")

if __name__ == "__main__":
    run_all_tests()
