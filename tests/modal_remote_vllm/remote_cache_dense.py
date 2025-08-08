import os
import argparse
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np
import requests
import tqdm
from openai import OpenAI

import bm25s.utils.beir as beir
import warnings
import logging

# Suppress all tokenizer warnings more aggressively
warnings.filterwarnings("ignore", message="Token indices sequence length*")
warnings.filterwarnings("ignore", message=".*sequence length.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------
EMB_BATCH = 256  # Match vLLM max_num_seqs for optimal GPU utilization
CONCURRENT_BATCHES = 8  # Increased concurrent requests to maximize throughput
DIM = 768
MAX_TOKENS = 512
DATASETS = [
    "trec-covid",
    # "nfcorpus",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "cqadupstack",
    "quora",
    "scidocs",
    "scifact",
]

# Initialize tokenizer lazily
TOKENIZER = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def get_tokenizer():
    """Get tokenizer instance, loading it lazily."""
    global TOKENIZER
    if TOKENIZER is None:
        from transformers import AutoTokenizer

        TOKENIZER = AutoTokenizer.from_pretrained(
            "Snowflake/snowflake-arctic-embed-m-v1.5"
        )
    return TOKENIZER


def chunk_text_by_tokens(
    text: str, max_tokens: int = 512, overlap: int = 50
) -> List[str]:
    """
    Split text into chunks that respect token limits with optional overlap.
    """
    # Use safe tokenization
    tokens = safe_tokenize(text, add_special_tokens=False)

    if len(tokens) <= max_tokens - 2:  # Reserve 2 for special tokens
        return [text]

    chunks = []
    max_content_tokens = max_tokens - 2  # Reserve for special tokens

    start = 0
    while start < len(tokens):
        end = min(start + max_content_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = get_tokenizer().decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        # Move start position with overlap consideration
        if end >= len(tokens):
            break
        start = end - overlap if overlap > 0 else end

    return chunks


def safe_tokenize(text: str, add_special_tokens: bool = True) -> List[int]:
    """
    Safely tokenize text without warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_tokenizer().encode(
            text, add_special_tokens=add_special_tokens, truncation=True, max_length=512
        )


def clip_to_max_tokens(text: str, max_len: int = 512) -> str:
    """
    Clip text to maximum token length, ensuring it doesn't exceed the limit.
    """
    # Use truncation directly in tokenization to avoid warnings
    tokens = safe_tokenize(text, add_special_tokens=False)

    # Reserve 2 tokens for special tokens like <s> and </s>
    max_content = max_len - 2

    if len(tokens) <= max_content:
        return text

    # Clip to max_content tokens
    clipped_tokens = tokens[:max_content]
    return get_tokenizer().decode(clipped_tokens, skip_special_tokens=True)


def validate_token_length(text: str, max_len: int = 512) -> bool:
    """
    Validate that text doesn't exceed token limit when encoded.
    """
    tokens = safe_tokenize(text, add_special_tokens=True)
    return len(tokens) <= max_len


def load_queries_robust(dataset: str, save_dir: str) -> dict:
    """
    Robust query loading that handles malformed JSON escape sequences.
    Fallback for when beir.load_queries fails due to JSON parsing errors.
    """
    import json
    from pathlib import Path

    queries_file = Path(save_dir) / dataset / "queries.jsonl"
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    queries = {}
    skipped_lines = 0

    with open(queries_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Try standard JSON parsing first
                query_data = json.loads(line)
                query_id = query_data.get("_id", str(line_num))

                # Extract text, handling different possible field names
                text = (
                    query_data.get("text")
                    or query_data.get("query")
                    or query_data.get("body", "")
                )

                queries[query_id] = {"text": text}

            except json.JSONDecodeError:
                # Handle malformed escape sequences
                try:
                    # Fix common escape sequence issues
                    fixed_line = line
                    # Replace problematic \x sequences with unicode equivalents
                    import re

                    # Replace \xef with proper unicode
                    fixed_line = re.sub(
                        r"\\x([0-9a-fA-F]{2})",
                        lambda m: chr(int(m.group(1), 16)),
                        fixed_line,
                    )

                    query_data = json.loads(fixed_line)
                    query_id = query_data.get("_id", str(line_num))

                    text = (
                        query_data.get("text")
                        or query_data.get("query")
                        or query_data.get("body", "")
                    )

                    queries[query_id] = {"text": text}

                except Exception as fix_error:
                    print(
                        f"Warning: Skipping malformed query at line {line_num}: {fix_error}"
                    )
                    skipped_lines += 1
                    continue

    if skipped_lines > 0:
        print(f"Warning: Skipped {skipped_lines} malformed query lines")

    return queries


class RemoteEmbeddings:
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        max_concurrent: int = CONCURRENT_BATCHES,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_concurrent = max_concurrent

    def embed(self, texts: List[str], max_retries: int = 3) -> List[np.ndarray]:
        # Validate all texts before sending
        validated_texts = []
        for text in texts:
            if not validate_token_length(text, MAX_TOKENS):
                # If validation fails, re-clip the text more aggressively
                text = clip_to_max_tokens(text, MAX_TOKENS - 10)  # Extra buffer
                if not validate_token_length(text, MAX_TOKENS):
                    # Last resort: truncate very aggressively
                    tokens = get_tokenizer().encode(text, add_special_tokens=False)
                    safe_tokens = tokens[: MAX_TOKENS - 10]
                    text = get_tokenizer().decode(safe_tokens, skip_special_tokens=True)
            validated_texts.append(text)

        # Retry logic for network resilience under high load
        for attempt in range(max_retries):
            try:
                resp = self.client.embeddings.create(
                    model="Snowflake/snowflake-arctic-embed-m-v1.5",
                    input=validated_texts,
                    encoding_format="float",
                )
                # Convert to float32 for downstream usage
                return [np.array(e.embedding, dtype=np.float32) for e in resp.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                # Exponential backoff on retry
                wait_time = (2**attempt) * 0.5  # 0.5s, 1s, 2s
                time.sleep(wait_time)

    def embed_concurrent(self, texts: List[str]) -> List[np.ndarray]:
        """
        Process embeddings with concurrent batches to maximize vLLM server utilization.
        """
        if len(texts) <= EMB_BATCH:
            return self.embed(texts)

        # Split texts into batches
        batches = [texts[i : i + EMB_BATCH] for i in range(0, len(texts), EMB_BATCH)]

        embeddings = []

        # Process batches concurrently with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.embed, batch): i for i, batch in enumerate(batches)
            }

            # Collect results in order
            batch_results = [None] * len(batches)

            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    batch_results[batch_idx] = batch_embeddings
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    # Fallback: create zero vectors
                    batch_size = len(batches[batch_idx])
                    batch_results[batch_idx] = [
                        np.zeros(DIM, dtype=np.float32) for _ in range(batch_size)
                    ]

            # Flatten results
            for batch_result in batch_results:
                if batch_result:
                    embeddings.extend(batch_result)

        return embeddings


def ensure_corpus(ds: str) -> dict:
    save_dir = Path("datasets") / f"hybrid_{ds}"
    save_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = save_dir / ds / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"[{ds}] downloading …")
        # Use the correct base URL directly
        base_url = (
            "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
        )
        print(f"[{ds}] Debug: Using URL: {base_url.format(ds)}")
        beir.download_dataset(ds, base_url=base_url, save_dir=str(save_dir))
        print(f"[{ds}] download complete")

    print(f"[{ds}] loading corpus …")
    corpus = beir.load_corpus(ds, save_dir=str(save_dir))
    print(f"[{ds}] loaded {len(corpus):,} documents")
    return corpus


def cache_dataset(
    ds: str, embedder: RemoteEmbeddings, use_chunking: bool = False
) -> None:
    cache_file = Path("datasets") / f"hybrid_{ds}" / "dense_embeddings.npz"
    if cache_file.exists():
        print(f"[{ds}] already cached – skipping")
        return

    corpus = ensure_corpus(ds)

    # Load queries as well
    print(f"[{ds}] loading queries ...")
    import bm25s.utils.beir as beir

    try:
        queries = beir.load_queries(ds, save_dir=str(Path("datasets") / f"hybrid_{ds}"))
        print(f"[{ds}] loaded {len(queries):,} queries")
    except Exception as e:
        print(f"[{ds}] Error loading queries with beir, attempting manual parsing: {e}")
        queries = load_queries_robust(
            ds, save_dir=str(Path("datasets") / f"hybrid_{ds}")
        )
        print(f"[{ds}] loaded {len(queries):,} queries (robust parsing)")

    if use_chunking:
        # Use chunking approach for very long documents
        print(f"[{ds}] processing documents with chunking...")
        texts = []
        corpus_ids = []
        for doc_id, doc in tqdm.tqdm(
            corpus.items(), desc=f"{ds} chunking", unit="docs"
        ):
            combined_text = f"{doc.get('title', '')} {doc['text']}".strip()
            chunks = chunk_text_by_tokens(combined_text, max_tokens=MAX_TOKENS)

            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                # Create unique IDs for chunks
                chunk_id = f"{doc_id}_chunk_{i}" if len(chunks) > 1 else doc_id
                corpus_ids.append(chunk_id)
        print(
            f"[{ds}] created {len(texts):,} text chunks from {len(corpus):,} documents"
        )
    else:
        # Use simple clipping approach
        print(f"[{ds}] processing documents with clipping...")
        texts = []
        corpus_ids = list(corpus.keys())

        for doc_id in tqdm.tqdm(corpus_ids, desc=f"{ds} clipping", unit="docs"):
            doc = corpus[doc_id]
            combined_text = f"{doc.get('title', '')} {doc['text']}".strip()
            clipped_text = clip_to_max_tokens(combined_text, max_len=MAX_TOKENS)

            # Double-check the clipped text
            if not validate_token_length(clipped_text, MAX_TOKENS):
                print(
                    f"\n[{ds}] Warning: Document {doc_id} still too long, applying aggressive clipping"
                )
                # More aggressive clipping
                tokens = safe_tokenize(clipped_text, add_special_tokens=False)
                safe_tokens = tokens[: MAX_TOKENS - 10]  # Extra safety margin
                clipped_text = get_tokenizer().decode(
                    safe_tokens, skip_special_tokens=True
                )

            texts.append(clipped_text)
        print(f"[{ds}] processed {len(texts):,} documents")

    print(f"[{ds}] generating {len(texts):,} dense vectors with concurrent processing")

    # Use concurrent embedding for maximum vLLM server utilization
    try:
        with tqdm.tqdm(total=len(texts), desc=f"{ds} embeddings", unit="docs") as pbar:
            # Process all texts with concurrent batching
            embeddings = embedder.embed_concurrent(texts)
            pbar.update(len(texts))
    except Exception as e:
        print(
            f"\n[{ds}] Error with concurrent processing, falling back to sequential: {e}"
        )
        # Fallback to sequential processing
        embeddings = []
        with tqdm.tqdm(
            total=len(texts), desc=f"{ds} embeddings (fallback)", unit="docs"
        ) as pbar:
            for start in range(0, len(texts), EMB_BATCH):
                batch = texts[start : start + EMB_BATCH]
                try:
                    batch_embeddings = embedder.embed(batch)
                    embeddings.extend(batch_embeddings)
                    pbar.update(len(batch))
                except Exception as batch_e:
                    print(
                        f"\n[{ds}] Error processing batch {start // EMB_BATCH}: {batch_e}"
                    )
                    # Process items individually if batch fails
                    for text in batch:
                        try:
                            single_embedding = embedder.embed([text])
                            embeddings.extend(single_embedding)
                            pbar.update(1)
                        except Exception as single_e:
                            print(
                                f"\n[{ds}] Failed to embed text (length: {len(text)}): {single_e}"
                            )
                            # Add zero vector as fallback
                            embeddings.append(np.zeros(DIM, dtype=np.float32))
                            pbar.update(1)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Generate query embeddings as well
    print(f"[{ds}] generating {len(queries):,} query embeddings...")
    query_ids = list(queries.keys())

    # Process query texts with the same clipping logic as corpus
    query_texts = []
    for qid in query_ids:
        query_text = queries[qid]["text"]
        clipped_text = clip_to_max_tokens(query_text, max_len=MAX_TOKENS)

        # Double-check the clipped text
        if not validate_token_length(clipped_text, MAX_TOKENS):
            print(
                f"\n[{ds}] Warning: Query {qid} still too long, applying aggressive clipping"
            )
            # More aggressive clipping
            tokens = safe_tokenize(clipped_text, add_special_tokens=False)
            safe_tokens = tokens[: MAX_TOKENS - 10]  # Extra safety margin
            clipped_text = get_tokenizer().decode(safe_tokens, skip_special_tokens=True)

        query_texts.append(clipped_text)

    try:
        with tqdm.tqdm(
            total=len(query_texts), desc=f"{ds} query embeddings", unit="queries"
        ) as pbar:
            query_embeddings = embedder.embed_concurrent(query_texts)
            pbar.update(len(query_texts))
    except Exception as e:
        print(
            f"\n[{ds}] Error with concurrent query processing, falling back to sequential: {e}"
        )
        query_embeddings = []
        with tqdm.tqdm(
            total=len(query_texts),
            desc=f"{ds} query embeddings (fallback)",
            unit="queries",
        ) as pbar:
            for start in range(0, len(query_texts), EMB_BATCH):
                batch = query_texts[start : start + EMB_BATCH]
                try:
                    batch_embeddings = embedder.embed(batch)
                    query_embeddings.extend(batch_embeddings)
                    pbar.update(len(batch))
                except Exception as batch_e:
                    print(
                        f"\n[{ds}] Error processing query batch {start // EMB_BATCH}: {batch_e}"
                    )
                    for text in batch:
                        try:
                            single_embedding = embedder.embed([text])
                            query_embeddings.extend(single_embedding)
                            pbar.update(1)
                        except Exception as single_e:
                            print(
                                f"\n[{ds}] Failed to embed query (length: {len(text)}): {single_e}"
                            )
                            query_embeddings.append(np.zeros(DIM, dtype=np.float32))
                            pbar.update(1)

    query_embeddings = np.array(query_embeddings, dtype=np.float32)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        corpus_ids=corpus_ids,
        corpus_embeddings=embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
    )
    print(
        f"[{ds}] cached {len(embeddings)} corpus vectors and {len(query_embeddings)} query vectors"
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    # Declare globals at the top of the function
    global EMB_BATCH, CONCURRENT_BATCHES

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        help="Run for a single dataset (optional). Default: all.",
    )
    parser.add_argument(
        "--use-chunking",
        action="store_true",
        help="Use text chunking instead of simple clipping for long documents",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding generation (default: 256)",
    )
    parser.add_argument(
        "--concurrent-batches",
        type=int,
        default=8,
        help="Number of concurrent API calls (default: 8)",
    )
    args = parser.parse_args()

    # Update globals based on arguments
    EMB_BATCH = args.batch_size
    CONCURRENT_BATCHES = args.concurrent_batches

    base_url = os.getenv(
        "REMOTE_BASE_URL", "https://rycerzes--arctic-embeddings-serve.modal.run/v1"
    )

    print(f"Using remote embedding server: {base_url}")
    print(
        f"Configuration: batch_size={EMB_BATCH}, concurrent_batches={CONCURRENT_BATCHES}"
    )

    # quick health-check
    try:
        print("Checking server health...")
        requests.get(base_url.replace("/v1", "/health"), timeout=5).raise_for_status()
        print("Server is healthy ✓")
    except Exception as e:
        parser.error(f"Remote server not reachable: {e}")

    print("Initializing tokenizer...")
    # Force tokenizer loading here to show progress
    _ = get_tokenizer().encode("test", add_special_tokens=True)
    print("Tokenizer ready ✓")

    embedder = RemoteEmbeddings(base_url, max_concurrent=CONCURRENT_BATCHES)

    todo = [args.dataset] if args.dataset else DATASETS
    print(f"Processing {len(todo)} dataset(s): {', '.join(todo)}")

    for ds in todo:
        cache_dataset(ds, embedder, use_chunking=args.use_chunking)

    print("All requested datasets cached.")


if __name__ == "__main__":
    main()
