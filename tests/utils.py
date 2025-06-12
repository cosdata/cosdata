import requests
import time

def get_transaction_status(client, coll_name, txn_id):
    """Get the status of a transaction"""
    host = client.host
    url = f"{host}/vectordb/collections/{coll_name}/transactions/{txn_id}/status"
    resp = requests.get(url, headers=client._get_headers(), verify=False)
    result = resp.json()
    return result['status']

def poll_transaction_completion(client, collection_name, txn_id, target_status='complete', 
                               max_attempts=10, sleep_interval=1):
    """
    Poll transaction status until it reaches the target status or max attempts are exceeded.
    
    Args:
        client: The cosdata client instance
        collection_name: Name of the collection
        txn_id: Transaction ID to poll
        target_status: Target status to wait for (default: 'complete')
        max_attempts: Maximum number of polling attempts
        sleep_interval: Time to sleep between attempts in seconds
    
    Returns:
        tuple: (final_status, success_boolean)
    """
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt + 1}: Waiting for transaction {txn_id} to complete...")
            
            # Get actual transaction status
            status = get_transaction_status(client, collection_name, txn_id)
            
            if status == target_status:
                print(f"Transaction {txn_id} completed successfully")
                return status, True
            
            if attempt < max_attempts - 1:
                time.sleep(sleep_interval)
                
        except Exception as e:
            print(f"Error polling transaction status: {e}")
            if attempt < max_attempts - 1:
                time.sleep(sleep_interval)
    
    print(f"Transaction {txn_id} may not have completed within {max_attempts} attempts")
    return "unknown", False
