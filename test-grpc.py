import grpc
import time
import numpy as np
import pandas as pd
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

# --- Collection Service Functions ---

def create_collection_grpc(stub, name, description, dimension):
    """Creates a collection using the gRPC server."""
    try:
        request = vector_service.CreateCollectionRequest(
            name=name,
            description=description,
            dense_vector=vector_service.DenseVectorOptions(dimension=dimension),
            config=vector_service.CollectionConfig(on_disk_payload=False)
        )
        response = stub.CreateCollection(request)
        return response
    except grpc.RpcError as e:
        print(f"Error creating collection: {e.details()}")
        raise

def get_collections_grpc(stub):
    """Gets all collections using the gRPC server."""
    try:
        request = vector_service.GetCollectionsRequest()
        response = stub.GetCollections(request)
        return response
    except grpc.RpcError as e:
        print(f"Error getting collections: {e.details()}")
        raise

def get_collection_grpc(stub, collection_id):
    """Gets a specific collection using the gRPC server."""
    try:
        request = vector_service.GetCollectionRequest(id=collection_id)
        response = stub.GetCollection(request)
        return response
    except grpc.RpcError as e:
        print(f"Error getting collection: {e.details()}")
        raise

def delete_collection_grpc(stub, collection_id):
    """Deletes a collection using the gRPC server."""
    try:
        request = vector_service.DeleteCollectionRequest(id=collection_id)
        response = stub.DeleteCollection(request)
        return response
    except grpc.RpcError as e:
        print(f"Error deleting collection: {e.details()}")
        raise

def run_collection_tests():
    """Test all Collection service operations"""
    with create_grpc_channel() as channel:
        stub = vector_service_grpc.CollectionsServiceStub(channel)

        test_collection_name = "test_collection"

        print("\n=== Testing Collections Service ===")

        # Test 1: Create Collection
        print("\n1. Testing CreateCollection...")
        try:
            response = create_collection_grpc(
                stub,
                test_collection_name,
                "Test collection via gRPC",
                768  # Example dimension
            )
            print(f"Collection created successfully: {response}")
        except grpc.RpcError as e:
            print(f"CreateCollection failed: {e.details()}")
            return

        # Test 2: Get All Collections
        print("\n2. Testing GetCollections...")
        try:
            response = get_collections_grpc(stub)
            print(f"Got collections successfully: {response}")
        except grpc.RpcError as e:
            print(f"GetCollections failed: {e.details()}")

        # Test 3: Get Specific Collection
        print("\n3. Testing GetCollection...")
        try:
            response = get_collection_grpc(stub, test_collection_name)
            print(f"Got collection successfully: {response}")
        except grpc.RpcError as e:
            print(f"GetCollection failed: {e.details()}")

        # Test 4: Delete Collection
        print("\n4. Testing DeleteCollection...")
        try:
            response = delete_collection_grpc(stub, test_collection_name)
            print("Collection deleted successfully")
        except grpc.RpcError as e:
            print(f"DeleteCollection failed: {e.details()}")

        # Verify deletion
        print("\n5. Verifying deletion...")
        try:
            collections = get_collections_grpc(stub)
            print(f"Collections after deletion: {collections}")
        except grpc.RpcError as e:
            print(f"Verification failed: {e.details()}")

def test_unimplemented_services():
    """Test other services that haven't been implemented yet"""
    with create_grpc_channel() as channel:
        print("\n=== Testing Unimplemented Services ===")

        # Test Auth Service
        print("\n1. Testing Auth Service (expected to be unimplemented)...")
        auth_stub = vector_service_grpc.AuthServiceStub(channel)
        try:
            request = vector_service.LoginRequest(
                username="test",
                password="test",
                pretty_print=True
            )
            response = auth_stub.Login(request)
            print("Unexpected success: Auth service is implemented")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                print("✓ Auth service correctly returns unimplemented")
            else:
                print(f"Unexpected error: {e.details()}")

        # Test Vectors Service
        print("\n2. Testing Vectors Service (expected to be unimplemented)...")
        vectors_stub = vector_service_grpc.VectorsServiceStub(channel)
        try:
            request = vector_service.CreateVectorRequest(
                collection_id="test",
                vector=vector_service.Vector(id=1, values=[1.0, 2.0, 3.0])
            )
            response = vectors_stub.CreateVector(request)
            print("Unexpected success: Vectors service is implemented")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                print("✓ Vectors service correctly returns unimplemented")
            else:
                print(f"Unexpected error: {e.details()}")

if __name__ == "__main__":
    print("Starting gRPC tests...")

    # Test implemented Collections service
    run_collection_tests()

    # Test unimplemented services
    test_unimplemented_services()
