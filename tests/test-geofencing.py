import os
import getpass
import requests
import random
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from cosdata import Client
from random import choice, randint

def read_csv_to_dicts(file_path: str) -> list[dict[str, str]]:
    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{str(k): str(v) for k, v in row.items()} for row in reader]

def generate_queries(dishes, restaurants, num_queries):
    query_templates = [
        "Looking for {dish} near {landmark} in {neighborhood} for {meal_timing}",
        "Best places to get {cuisine} {dish} in {city} with {dietary} options under {price_range}",
        "Where can I find {restaurant}'s signature {dish} with {cooking_method} {protein} and {key_vegetables}",
        "Top rated {spice_level} {cuisine} restaurants in {neighborhood} near {landmark}",
        "Looking for {category} dishes with {ingredient} from {region} cuisine in {city}",
        "Family-style {meal_type} options from {restaurant} with {dietary} accommodations",
        "Best {cooking_method} {protein} with {signature_spices} near {major_road_landmark}",
        "Where to find authentic {region} {dish} in {city_district} for {meal_timing}",
        "Restaurants in {neighborhood} offering {cuisine} {category} with {dietary} options",
        "Places near {landmark} serving {spice_level} {main_protein} dishes with {key_vegetables}",
        "Best {price_range} {meal_type} delivery from {restaurant} in {neighborhood}",
        "Looking for {dietary} friendly {cuisine} options near {major_road_landmark}",
        "Where to get {cooking_method} {vegetable} dishes with {signature_spices} in {city}",
        "Top restaurants in {city_district} for {typical_meal_timing} {category}",
        "Best {restaurant} {meal_type} specials with {main_protein} and {key_vegetables}",
        "Traditional {region} {dish} recipe style restaurants near {landmark}",
        "Where to find {cuisine} {category} for {meal_timing} in {neighborhood}",
        "Best places for {dish} with {dietary} options in {city} near {landmark}",
        "{restaurant} {dietary} options for {meal_type} with {main_protein}",
        "Authentic {region} cuisine with {signature_spices} and {key_vegetables} near {major_road_landmark}"
    ]
    
    generated_queries = set()
    
    while len(generated_queries) < num_queries:
        template = choice(query_templates)
        dish = choice(dishes)
        restaurant = choice(restaurants)
        
        replacements = {
            'dish': dish['dish_name'],
            'cuisine': dish['cuisine_origin'],
            'landmark': restaurant['major_road_landmark'],
            'neighborhood': restaurant['local_area_neighborhood'],
            'city': restaurant['city_district'],
            'restaurant': restaurant['restaurant_name'],
            'cooking_method': dish['primary_cooking_method'],
            'protein': dish['main_protein'] or 'vegetarian',
            'spice_level': dish['spice_level'],
            'dietary': dish['dietary_classifications'] or 'diet-friendly',
            'category': dish['dish_category'],
            'ingredient': dish['key_vegetables'] or dish['signature_spices'],
            'region': dish['region'],
            'price_range': restaurant['minimum_order_requirements'],
            'meal_type': choice(restaurant['meal_types_offered'].split('|')),
            'vegetable': dish['key_vegetables'],
            'meal_timing': choice(dish['typical_meal_timing'].split('|')),
            'major_road_landmark': restaurant['major_road_landmark'],
            'city_district': restaurant['city_district'],
            'signature_spices': dish['signature_spices'],
            'main_protein': dish['main_protein'] or 'plant-based protein',
            'key_vegetables': dish['key_vegetables'],
            'typical_meal_timing': choice(dish['typical_meal_timing'].split('|'))
        }
        
        try:
            query = template.format(**replacements)
            # Add random variations
            variations = [
                lambda q: q + " with delivery",
                lambda q: q + " and takeaway",
                lambda q: q.replace("near ", "in the vicinity of "),
                lambda q: q.replace("Best ", "Top rated "),
                lambda q: q + " for large groups" if randint(0,1) else q,
                lambda q: q + " with outdoor seating" if randint(0,1) else q,
                lambda q: q + " open now" if randint(0,1) else q
            ]
            
            for var in variations:
                if randint(0, 2) == 0:  # 33% chance for each variation
                    query = var(query)
                    
            if randint(0, 1):
                query = query.lower()
                
            generated_queries.add(query)
        except KeyError as _e:
            continue
            
    return list(generated_queries)

def execute_batch_query(queries, url, headers, zone):
    payload = {
        "queries": queries,
        "zones": [zone],
        "coordinates": [0.0, 0.0],
        "sort_by_distance": False,
        "top_k": 5,
        "early_terminate_threshold": 0.0,
    }
    response = requests.post(url, headers=headers, json=payload, verify=False)
    return response.status_code == 200

def main():
    DB_NAME = "db_name"
    ZONE = "common_zone"
    NUM_QUERIES = 100_000
    MAX_WORKERS = 50  # Adjust based on your system capabilities

    password = os.getenv("COSDATA_PASSWORD")
    if not password:
        password = getpass.getpass("Enter your database password: ")

    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    username = os.getenv("COSDATA_USERNAME", "admin")
    client = Client(host=host, username=username, password=password, verify=False)

    restaurants = read_csv_to_dicts("restaurants.csv")
    dishes = read_csv_to_dicts("dishes.csv")

    collection = client.create_collection(name=DB_NAME, sparse_vector={"enabled": True})
    collection.create_sparse_index(name=DB_NAME)

    # Insert restaurants
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

    # Insert dishes
    dish_upserts = []
    dish_counter = 0
    for restaurant_idx in range(len(restaurants)):
        for _ in range(20):
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
        txn.batch_upsert_vectors(dish_upserts)
    txn.poll_completion(target_status="complete", max_attempts=30, sleep_interval=1)

    # Generate queries and prepare for execution
    queries = generate_queries(dishes, restaurants, NUM_QUERIES)
    random.shuffle(queries)
    print(f"Generated {len(queries)} queries for testing")

    url = f"{client.base_url}/collections/{DB_NAME}/search/batch-geofence"
    headers = client._get_headers()

    # Execute queries with threading
    start_time = time.time()
    successful_queries = 0
    failed_queries = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(execute_batch_query, queries[start:start+100], url, headers, ZONE)
            for start in range(0, len(queries), 100)
        ]
        
        for future in as_completed(futures):
            if future.result():
                successful_queries += 100
            else:
                failed_queries += 100

    total_time = time.time() - start_time
    qps = len(queries) / total_time

    print(f"\nCompleted {len(queries)} queries in {total_time:.2f} seconds")
    print(f"QPS: {qps:.2f}")
    print(f"Successful queries: {successful_queries}")
    print(f"Failed queries: {failed_queries}")

if __name__ == "__main__":
    main()
