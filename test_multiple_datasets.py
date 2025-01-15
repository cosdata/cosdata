from concurrent.futures import ThreadPoolExecutor
import random
import time
import polars as pl
import os
from termcolor import colored
import numpy as np
import gc
from scipy.spatial.distance import cdist

from test import ann_vector, commit_transaction, create_transaction, login, create_db, upsert_in_transaction

def display_and_process_selections():
	datasets = [
		"arxiv-embeddings-ada-002",
		"job_listing_embeddings",
		"million-text-embeddings",
		"cohere-wiki-embedding-100k"
	]
	dataset_urls = {
		"arxiv-embeddings-ada-002": "hf://datasets/Tychos/arxiv-embeddings-ada-002/**/*.parquet",
		"job_listing_embeddings": "hf://datasets/serbog/job_listing_embeddings/job_listings.parquet",
		"million-text-embeddings": "hf://datasets/Sreenath/million-text-embeddings/data/train-*.parquet",
		"cohere-wiki-embedding-100k": "hf://datasets/ashraq/cohere-wiki-embedding-100k/data/train-*-of-*.parquet"
	}
	
	print(colored("Select dataset to benchmark:", "cyan"))
	for idx, dataset in enumerate(datasets, start=1):
		print(colored(f"{idx}. {dataset}", "yellow"))
	
	while True:
		try:
			choice = int(input(colored("Enter option number: ", "green")))
			if 1 <= choice <= len(datasets):
				break
			print("Invalid choice! Please enter a number within the valid range.")
		except ValueError:
			print("Please enter a valid number!")
	
	selected_dataset = datasets[choice - 1]
	dataset_url = dataset_urls[selected_dataset]

	subdir = os.path.join("datasets", selected_dataset)
	os.makedirs(subdir, exist_ok=True)
	parquet_file_path = os.path.join(subdir, f"{selected_dataset}.parquet")

	if not os.path.exists(parquet_file_path):
		print(colored("Parquet file doesn't exist, downloading form huggingface...", "blue"))
		df = pl.scan_parquet(source=dataset_url)
		df.sink_parquet(parquet_file_path)
		print(colored(f"Parquet file downloaded at {parquet_file_path}!","light_green"))
		select_query_vectors_and_process_bruteforced_results(parquet_file_path=parquet_file_path)
	else:
		print("Parquet file already exists, calculating bruteforced results..")
		select_query_vectors_and_process_bruteforced_results(parquet_file_path=parquet_file_path)


def select_query_vectors_and_process_bruteforced_results(parquet_file_path: str):
	bf_csv_path = os.path.join(os.path.dirname(parquet_file_path), f"bf_{os.path.basename(parquet_file_path).replace('.parquet','.csv')}")

	if os.path.exists(bf_csv_path):
		print(f"Brute force results stored in {bf_csv_path}, getting server results..")
		calculate_server_similarities(parquet_file_path, bf_csv_path)
		return
		
	df = pl.read_parquet(parquet_file_path, low_memory=True)

	if 'emb' not in df.columns:
		if 'embeddings' in df.columns:
			df = df.rename({"embeddings": "emb"})
		elif 'embedding' in df.columns:
			df = df.rename({"embedding": "emb"})
	
	if 'id' not in df.columns:
		df = df.with_row_index(name='id')

	corrected_values = df.get_column('emb').list.eval(
		pl.element().clip(-0.9878, 0.987890)
	)
	df = df.replace_column(df.get_column_index("emb"), corrected_values)
	df.write_parquet(parquet_file_path)
	
	df = df[["id","emb"]]
	vectors_corrected = []
	for v in df.iter_rows():
		vectors_corrected.append({
			'id': v[0],
			'values': v[1]
		})
	total_vectors = len(vectors_corrected)
	print(f"Total vectors read from file: {total_vectors}")
	print("Computing brute forced similarities:")
	np.random.seed(25)
	test_indices = np.random.choice(total_vectors, 100, replace=False)
	test_matrix = np.array([vectors_corrected[i]['values'] for i in test_indices], dtype=np.float32)
	all_matrix = np.array([v['values'] for v in vectors_corrected], dtype=np.float32)

	distances = cdist(test_matrix, all_matrix, metric='cosine')
	similarities = 1 - distances

	results = []
	for i, row in enumerate(similarities):
		top_indices = np.argpartition(row, -6)[-6:]
		top_indices = top_indices[np.argsort(row[top_indices])][::-1]
		top_5 = top_indices[1:6]
		query_id = vectors_corrected[test_indices[i]]['id']
		top_ids = [vectors_corrected[idx]['id'] for idx in top_5]
		top_sims = [float(row[idx]) for idx in top_5]

		results.append({
			'query_id': query_id,
			'top1_id': top_ids[0],
			'top1_score': top_sims[0],
			'top2_id': top_ids[1],
			'top2_score': top_sims[1],
			'top3_id': top_ids[2],
			'top3_score': top_sims[2],
			'top4_id': top_ids[3],
			'top4_score': top_sims[3],
			'top5_id': top_ids[4],
			'top5_score': top_sims[4]
		})
	df = pl.DataFrame(results)
	df.write_csv(bf_csv_path)
 
	print(colored(f"Brute force results stored in {bf_csv_path}, getting server results...", "green"))
 
	del df
	del vectors_corrected
	del test_indices
	del test_matrix
	del distances
	del similarities
	gc.collect()
 
	calculate_server_similarities(parquet_file_path, bf_csv_path)

def calculate_server_similarities(parquet_path: str, bf_csv_path: str):
	df = pl.read_parquet(parquet_path, low_memory=True)
	constants = {
		"COLLECTION_NAME": os.path.splitext(os.path.basename(parquet_path))[0],
		"DIMENSIONS": max(len(emb) for emb in df['emb']),
		"MAX_VAL": 1.0,
		"MIN_VAL": -1.0,
		"BATCH_SIZE": 500,
		"BATCH_COUNT": None
	}
	
	vectors = []
	df = df[['id','emb']]
	for v in df.iter_rows():
		vector = {
			'id': v[0],
			'values': v[1]
		}
		vectors.append(vector)
	constants['TOKEN'] = login()
	create_db(constants["COLLECTION_NAME"], "Embeddings from dataset", constants['DIMENSIONS'])
	
	def upsert_with_retry(start_idx, retries=20):
		for attempt in range(retries):
			try:
				upsert_in_transaction(constants["COLLECTION_NAME"], txn_id, vectors[start_idx : start_idx + batch_size])
				return
			except Exception as e:
				print(f"Upsert attempt {attempt + 1} failed for batch starting at index {start_idx}: {e}")
				time.sleep(random.uniform(1, 3)) 
		print(f"Failed to upsert batch starting at index {start_idx} after {retries} attempts")

	transaction = create_transaction(constants["COLLECTION_NAME"])
	txn_id = transaction["transaction_id"]

	batch_size = constants['BATCH_SIZE']
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(upsert_with_retry, i) for i in range(0, len(vectors), batch_size)]
		for future in futures:
			future.result()

	commit_transaction(constants["COLLECTION_NAME"], txn_id)
	print(colored("Vectors upserted successfully!", "green"))
	
	def search_dataset_vectors(query_id, query_emb):
		result = ann_vector(query_id, constants["COLLECTION_NAME"], query_emb)
		return result

	query_vecs = pl.read_csv(bf_csv_path)
	df = pl.read_parquet(parquet_path, low_memory=True)
	results = []
	for row in query_vecs.iter_rows():
		query_id = row[0]
		embedding = df.filter(pl.col("id") == query_id)["emb"].to_list()[0]
		if embedding:
			embedding = [max(-0.9878, min(float(v), 0.987890)) for v in embedding]
			search_results = search_dataset_vectors(query_id, embedding)
			top5_results = search_results[1]['RespVectorKNN']['knn'][1:6]
			results.append({
				'query_id': query_id,
				'top1_id': top5_results[0][0],
				'top1_sim': top5_results[0][1]['CosineSimilarity'],
				'top2_id': top5_results[1][0],
				'top2_sim': top5_results[1][1]['CosineSimilarity'],
				'top3_id': top5_results[2][0],
				'top3_sim': top5_results[2][1]['CosineSimilarity'],
				'top4_id': top5_results[3][0],
				'top4_sim': top5_results[3][1]['CosineSimilarity'],
			})

	server_csv_path = bf_csv_path.replace("bf_", "server_")
	results_df = pl.DataFrame(results)
	results_df.write_csv(server_csv_path)
	print(colored(f"Finished storing search results in {server_csv_path}", "green"))
	compare_server_and_bruteforce(bf_csv_path)

def compare_server_and_bruteforce(bf_csv_path: str):
	bf = pl.read_csv(bf_csv_path)
	base_dir, bf_file_name = os.path.split(bf_csv_path)
	server_file_name = bf_file_name.replace("bf_", "server_")
	server_path = os.path.join(base_dir, server_file_name)
	sr = pl.read_csv(server_path)

	total_queries = sr.shape[0]
	hits = 0
	recall_sum = 0.0

	for row in sr.iter_rows(named=True):
		qid = row["query_id"]
		match = bf.filter(pl.col("query_id") == qid).row(0)

		bf_top_ids = [match[1], match[3], match[5], match[7]]
		bf_top_sims = [match[2], match[4], match[6], match[8]]
		srv_top_ids = [row["top1_id"], row["top2_id"], row["top3_id"], row["top4_id"]]
		srv_top_sims = [row["top1_sim"], row["top2_sim"], row["top3_sim"], row["top4_sim"]]

		print(colored(f"Query ID: {qid}", "yellow"))
		print(colored(f"BF top4 IDs: {bf_top_ids}", "blue"), colored(f"sims: {bf_top_sims}", "blue"))
		print(colored(f"Server top4 IDs: {srv_top_ids}", "magenta"), colored(f"sims: {srv_top_sims}", "magenta"))
		print("-" * 40)

		top_srv = set(srv_top_ids)
		top_bf = set(bf_top_ids)
		common = top_srv.intersection(top_bf)
		hits += len(common)
		recall_sum += len(common) / 4

	print(colored(f"Total queries compared: {total_queries}", "cyan"))
	print(colored(f"Total matching embeddings in server top4: {hits}", "cyan"))
	recall_percentage = (recall_sum / total_queries) * 100
	print(colored(f"Mean recall: {recall_percentage:.2f}%", "green"))

if __name__ == "__main__":
	display_and_process_selections()
	
