import os
import getpass
from cosdata import Client
import csv
import requests


def read_csv_to_dicts(file_path: str) -> list[dict[str, str]]:
    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{str(k): str(v) for k, v in row.items()} for row in reader]


def main():
    DB_NAME = "db_name"
    ZONE = "common_zone"

    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter your database password: ")

    # Initialize client
    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    username = os.getenv("COSDATA_USERNAME", "admin")
    client = Client(host=host, username=username, password=password, verify=False)

    restaurants = read_csv_to_dicts("restaurants.csv")
    dishes = read_csv_to_dicts("dishes.csv")

    collection = client.create_collection(name=DB_NAME, sparse_vector={"enabled": True})

    collection.create_sparse_index(name=DB_NAME)

    # Insert dishes - 20 per restaurant with unique IDs
    dish_upserts = []
    dish_counter = 0
    for restaurant_idx in range(len(restaurants)):
        for _ in range(20):
            # Cycle through available dish vectors
            dish_entry = dishes[dish_counter % len(dishes)]
            dish_upserts.append(
                {
                    "id": f"dish_{dish_counter}",
                    "document_id": f"restaurant_{restaurant_idx}",
                    "geo_fence_values": dish_entry,
                }
            )
            dish_counter += 1

    with collection.transaction() as txn:
        batch = [
            {
                "id": f"restaurant_{i}",
                "document_id": f"restaurant_{i}",
                "geo_fence_values": restaurant,
                "geo_fence_metadata": {
                    "weight": 1.0,
                    "coordinates": [0.0, 0.0],
                    "zone": ZONE,
                },
            }
            for i, restaurant in enumerate(restaurants)
        ]
        txn.batch_upsert_vectors(batch)

    txn.poll_completion(target_status="complete", max_attempts=30, sleep_interval=1)

    with collection.transaction() as txn:
        txn.batch_upsert_vectors(dish_upserts)

    txn.poll_completion(target_status="complete", max_attempts=30, sleep_interval=1)

    while True:
        query = input("\nSearch query: ").strip().lower()
        if query in ["exit", "quit"]:
            break

        url = f"{client.base_url}/collections/{DB_NAME}/search/geofence"
        headers = client._get_headers()

        payload = {
            "query": query,
            "zones": [ZONE],
            "coordinates": [0.0, 0.0],
            "sort_by_distance": False,
            "top_k": 5,
            "early_terminate_threshold": 0.0,
        }

        response = requests.post(url, headers=headers, json=payload, verify=False)

        results = response.json()["results"]

        # Display results with matches
        print(f"\nTop {len(results)} results for '{query}':")
        for i, result in enumerate(results):
            result_id = result["id"]
            doc_id = result["document_id"]
            restaurant_idx = int(doc_id.split("_")[1])
            restaurant_name = restaurants[restaurant_idx]["restaurant_name"]

            # For restaurant results
            if result_id.startswith("restaurant_"):
                print(
                    f"{i + 1}. Restaurant: {restaurant_name} (Score: {result['score']:.4f})"
                )

            # For dish results
            elif result_id.startswith("dish_"):
                dish_idx = int(result_id.split("_")[1]) % len(dishes)
                dish_name = dishes[dish_idx]["dish_name"]
                print(
                    f"{i + 1}. Dish: {dish_name} from '{restaurant_name}' (Score: {result['score']:.4f})"
                )

            # Display matches for all result types
            matches = result.get("matches", {})
            if matches:
                print("    Matches:")
                for field, words in matches.items():
                    word_str = ""
                    for i, (word, score) in enumerate(words):
                        word_str += f"{word} ({score:.4})"
                        if i != (len(words) - 1):
                            word_str += ", "
                    if words:  # Only show fields with matches
                        print(f"      • {field}: {word_str}")
            else:
                print("    No match details available")


if __name__ == "__main__":
    main()
