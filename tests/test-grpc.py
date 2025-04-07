import grpc
import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the generated gRPC code
import vector_service_pb2 as vector_service
import vector_service_pb2_grpc as vector_service_grpc

# gRPC server address
grpc_server_address = "[::1]:50051"


def create_grpc_channel():
    """Creates a gRPC channel."""
    return grpc.insecure_channel(grpc_server_address)


def create_collection_with_metadata(stub):
    """Test creating collection with metadata schema"""
    request = vector_service.CreateCollectionRequest(
        name="test_metadata_collection",
        description="Test collection with metadata schema",
        dense_vector=vector_service.DenseVectorOptions(
            dimension=128, enabled=True, auto_create_index=True
        ),
        sparse_vector=vector_service.SparseVectorOptions(
            enabled=False, auto_create_index=False
        ),
        metadata_schema=vector_service.MetadataSchema(
            fields=[
                vector_service.MetadataField(
                    name="category",
                    values=[
                        vector_service.FieldValue(string_value="electronics"),
                        vector_service.FieldValue(string_value="books"),
                        vector_service.FieldValue(string_value="clothing"),
                    ],
                ),
                vector_service.MetadataField(
                    name="rating",
                    values=[
                        vector_service.FieldValue(int_value=1),
                        vector_service.FieldValue(int_value=2),
                        vector_service.FieldValue(int_value=3),
                        vector_service.FieldValue(int_value=4),
                        vector_service.FieldValue(int_value=5),
                    ],
                ),
            ],
            supported_conditions=[
                vector_service.SupportedCondition(
                    op=vector_service.SupportedCondition.OperationType.AND,
                    field_names=["category", "rating"],
                )
            ],
        ),
        config=vector_service.CollectionConfig(
            max_vectors=1000000, replication_factor=1
        ),
    )
    return stub.CreateCollection(request)


def create_simple_collection(stub, name="test_collection", dimension=128):
    """Creates a simple collection without metadata schema"""
    request = vector_service.CreateCollectionRequest(
        name=name,
        description="Test collection via gRPC",
        dense_vector=vector_service.DenseVectorOptions(
            dimension=dimension, enabled=True, auto_create_index=True
        ),
        sparse_vector=vector_service.SparseVectorOptions(
            enabled=False, auto_create_index=False
        ),
        config=vector_service.CollectionConfig(
            max_vectors=1000000, replication_factor=1
        ),
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


def run_all_tests():
    """Run all gRPC tests"""
    print("Starting gRPC tests...")

    with create_grpc_channel() as channel:
        stub = vector_service_grpc.CollectionsServiceStub(channel)

        # Test basic operations
        if not test_basic_collection_operations(stub):
            print("\n❌ Basic collection operations tests failed!")
        else:
            print("\n✅ Basic collection operations tests passed!")

        # Test metadata operations
        if not test_metadata_collection_operations(stub):
            print("\n❌ Metadata collection operations tests failed!")
        else:
            print("\n✅ Metadata collection operations tests passed!")


if __name__ == "__main__":
    run_all_tests()
