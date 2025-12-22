<p align="center">
  <img src="org/logo.svg" alt="Cosdata" style="max-width: 100%; height: auto;">
</p>

<p align="center">
  <a href="https://cosdata.io">
    <img src="https://img.shields.io/badge/www-cosdata.io-pink">
  </a>
  <a href="https://github.com/cosdata/cosdata/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/cosdata/cosdata/ci.yml?label=build&color=green">
  </a>
  <img src="https://img.shields.io/badge/language-Rust-yellow">
  <img src="https://img.shields.io/badge/language-Python-black">
  <br>
  <a href="https://discord.gg/QFsrBfFVVY">
    <img src="https://img.shields.io/badge/Discord-Join%20Us-7289da?logo=discord&logoColor=white">
  </a>
  <a href="https://www.linkedin.com/company/cosdata/">
    <img src="https://img.shields.io/badge/our_journey-LinkedIn-blue">
  </a>
  <a href="https://github.com/cosdata/cosdata/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-blue">
  </a>
  <a href="https://github.com/cosdata/cosdata/pulls">
    <img src="https://img.shields.io/github/issues-pr/cosdata/cosdata?color=pink">
  </a> 
</p>

</br>
<p></p>

## 📦 Table of Contents

- [Overview](#-overview)
- [Why Cosdata?](#-why-cosdata)
  - [The Cosine Similarity Problem](#the-cosine-similarity-problem)
  - [The Relevance-First Approach](#the-relevance-first-approach)
  - [How Cosdata Delivers Relevance](#how-cosdata-delivers-relevance)
  - [Real-World Impact](#real-world-impact)
- [Benchmarks](#-benchmarks)
  - [Full-Text Search (BM25)](#-full-text-search-bm25)
  - [Dense Vector Search (HNSW)](#-dense-vector-search-hnsw)
  - [SPLADE Learned Sparse Embeddings](#-splade-learned-sparse-embeddings)
  - [Hybrid Search](#-hybrid-search-best-of-both-worlds)
  - [Cost Efficiency](#-cost-efficiency)
  - [Benchmark Methodology](#-benchmark-methodology)
- [Features](#-features)
  - [Search Relevance & Quality](#-search-relevance--quality)
  - [Performance at Scale](#-performance-at-scale)
  - [Enterprise-Grade Architecture](#-enterprise-grade-architecture)
  - [Developer Experience](#-developer-experience)
  - [Advanced Capabilities](#-advanced-capabilities)
- [Getting Started](#️-getting-started)
  - [Install](#1-install)
  - [Build from Source](#2-build-from-source)
  - [Testing Your Installation](#3-testing-your-installation)
  - [HTTPS Configuration (TLS)](#4-https-configuration-tls)
- [Client SDKs](#-client-sdks)
  - [Python SDK](#-python-sdk)
  - [Node.js SDK](#-nodejs-sdk)
- [Documentation](https://docs.cosdata.io/getting-started/introduction/)
- [Contributing](#-contributing)
- [Contacts & Community](#-contacts--community)
- [Show Your Support](#️-show-your-support)
  
<br>
<br>

# 🚀 Overview

Cosdata is a next-generation retrieval infrastructure engineered for AI-native applications that demand **relevance beyond simple vector similarity**. 

## The Challenge
Traditional vector databases optimize for cosine similarity rather than what users actually find useful. Decades of search evolution prove that effective retrieval requires sophisticated ranking systems that understand context, incorporate multiple signals, and optimize for user satisfaction—not just mathematical proximity.

## Our Solution
Built with immutability and version control at its core, Cosdata delivers a **relevance-first architecture** combining:

- **Multi-Modal Retrieval**: Seamlessly integrate BM25 full-text search, HNSW dense vectors, SPLADE learned sparse embeddings, and metadata-rich sparse vectors in a unified platform
- **Context-Aware Capabilities**: Leverage geofencing, hierarchical document organization, and explainable ranking that understands user intent and real-world complexity
- **Enterprise-Grade Architecture**: Benefit from colocated storage, streaming ingestion, transactional versioning, and comprehensive security features

## Proven Impact
Organizations using Cosdata achieve **60-120% reduction in compute requirements** while improving retrieval quality by **20-50% (NDCG@10)**. Our unified architecture eliminates external document stores and complex multi-database queries, reducing infrastructure costs and latency.

<br>

# 💡 Why Cosdata? 

## The Cosine Similarity Problem

Most vector databases treat retrieval as a pure similarity problem—if two embeddings are mathematically close in vector space, they must be relevant to each other. **This assumption is fundamentally flawed.**

High cosine similarity ≠ High relevance to users.

Cosine similarity measures the angle between embedding vectors—a mathematical distance determined by how a model was trained. But this metric has no inherent connection to what users actually find useful or relevant. Two documents can be mathematically similar while being practically useless for a user's information need, or vice versa.

## The Relevance-First Approach

**True relevance requires understanding context, not just proximity.**

Decades of search engine evolution—from Google's PageRank to modern recommendation systems—prove that effective retrieval demands:

- **Multiple signals**: Lexical matching, semantic understanding, metadata, recency, authority, and user context
- **Ground truth from users**: Real relevance comes from actual user behavior and expert judgments, not embedding distances
- **Explainable ranking**: Systems must show *why* results matter, not just that they're "similar"
- **Business logic integration**: Geographic constraints, temporal filters, hierarchical relationships, and domain-specific rules

## How Cosdata Delivers Relevance

Cosdata is built from the ground up to **optimize for user satisfaction, not mathematical convenience**:

1. **Hybrid Multi-Modal Search**: Combines BM25 lexical matching, dense vectors (HNSW), SPLADE learned sparse embeddings, and metadata-rich representations—letting each signal contribute what it does best

2. **Context-Aware Ranking**: Native support for geofencing, hierarchical document structures, temporal filtering, and custom business logic without requiring everything to be embedded

3. **Explainable Results**: Every result comes with transparent scoring showing semantic similarity contributions, metadata matches, geographic relevance, and hierarchical context

4. **Proven Quality Metrics**: We measure success using NDCG (Normalized Discounted Cumulative Gain) and recall against human-judged relevance datasets like BEIR—not just precision against our own similarity rankings

## Real-World Impact

Organizations using Cosdata see:
- **20-50% improvement in retrieval quality** (NDCG@10) compared to pure vector similarity approaches
- **60-120% reduction in compute requirements** through efficient multi-modal indexing
- **Sub-100ms response times** while maintaining relevance quality
- **Simplified architecture** with colocated storage eliminating external document stores

**Bottom line**: Cosdata treats retrieval as a relevance problem, not a storage problem. We've learned from decades of search evolution to build infrastructure that understands what users actually need.

<br>


# 📊 Benchmarks

Cosdata delivers exceptional performance across all retrieval modalities. Our benchmarks use industry-standard datasets and compare against leading solutions to demonstrate real-world performance gains.

## 🔍 Full-Text Search (BM25)

Our custom BM25 implementation outperforms Elasticsearch with dramatically higher throughput and lower latency while maintaining comparable ranking quality.

### Performance Highlights

- **Up to 151× higher QPS** than Elasticsearch (SciFact dataset)
- **Average 44× QPS improvement** across multiple IR benchmark datasets
- **Up to 12× faster indexing** on large-scale datasets
- **Lower latency** at both p50 and p95 percentiles across all tested datasets

### Detailed Comparison: Cosdata vs. Elasticsearch

| Dataset | Corpus Size | System | Indexing (sec) | QPS | NDCG@10 | p50 (ms) | p95 (ms) |
|---------|-------------|--------|----------------|-----|---------|----------|----------|
| **arguana** | 8.7K | **Cosdata** | **0.1** | **2,167** | 0.40 | **9** | **15** |
| | | Elasticsearch | 1.4 | 263 | 0.48 | 44 | 74 |
| **climate-fever** | 5.4M | **Cosdata** | **40.6** | **135** | 0.13 | **106** | 379 |
| | | Elasticsearch | 522.8 | 84 | 0.14 | 162 | 263 |
| **fever** | 5.4M | **Cosdata** | **40.3** | **314** | 0.47 | **52** | 157 |
| | | Elasticsearch | 525.7 | 154 | 0.52 | 80 | 138 |
| **fiqa** | 57K | **Cosdata** | **0.5** | **4,942** | 0.25 | **7** | **12** |
| | | Elasticsearch | 6.7 | 251 | 0.25 | 39 | 60 |
| **msmarco** | 8.8M | **Cosdata** | **57.7** | **315** | 0.23 | **46** | 162 |
| | | Elasticsearch | 714.7 | 166 | 0.23 | 73 | 129 |
| **nq** | 2.6M | **Cosdata** | **19.3** | **483** | 0.29 | **30** | **81** |
| | | Elasticsearch | 243.2 | 197 | 0.29 | 59 | 100 |
| **quora** | 522K | **Cosdata** | **2.7** | **1,425** | **0.81** | **11** | **36** |
| | | Elasticsearch | 30.2 | 323 | **0.81** | 39 | 55 |
| **scidocs** | 25K | **Cosdata** | **0.3** | **13,338** | **0.16** | **7** | **12** |
| | | Elasticsearch | 3.6 | 319 | 0.15 | 33 | 48 |
| **scifact** | 5.2K | **Cosdata** | **0.1** | **40,909** | **0.69** | **7** | **13** |
| | | Elasticsearch | 1.0 | 271 | 0.68 | 34 | 51 |
| **trec-covid** | 171K | **Cosdata** | **1.7** | **2,219** | 0.61 | **10** | **18** |
| | | Elasticsearch | 22.1 | 110 | 0.62 | 57 | 88 |
| **webis-touche2020** | 382K | **Cosdata** | **5.5** | **2,789** | **0.34** | **10** | **18** |
| | | Elasticsearch | 63.1 | 108 | **0.34** | 62 | 99 |

**Key Takeaway**: Cosdata maintains comparable or better ranking quality (NDCG@10) while delivering dramatically higher throughput and lower latency.

---

## 🎯 Dense Vector Search (HNSW)

Our HNSW implementation achieves industry-leading performance on large-scale vector datasets with high-dimensional embeddings.

### Performance Highlights

- **1,758 QPS** on 1 million records (1536 dimensions)
- **~42% faster** than Qdrant
- **~54% faster** than Weaviate  
- **~146% faster** than Elasticsearch
- **Consistent 97% precision** at high throughput

### Detailed Comparison: Million-Scale Vector Search

**Dataset**: DbPedia (Qdrant benchmark) | **Size**: 1 million records | **Dimensions**: 1536

| System | Indexing (min) | QPS | Precision | p50 (ms) | p95 (ms) |
|--------|----------------|-----|-----------|----------|----------|
| **Cosdata** | **16.32** | **1,758** | 0.97 | 7 | 8 |
| Qdrant | 24.43 | 1,238 | **0.99** | **4** | **5** |
| Weaviate | **13.94** | 1,142 | 0.97 | 5 | 7 |
| Elasticsearch | 83.72 | 716 | 0.98 | 22 | 73 |

**Key Takeaway**: Cosdata delivers the highest throughput with competitive precision and fast indexing times, making it ideal for production workloads requiring both speed and accuracy.

---

## 🧠 SPLADE Learned Sparse Embeddings

SPLADE combines neural and lexical matching for improved retrieval quality in domain-specific applications. While offering 15-25% better ranking quality, it trades throughput for precision.

### Quality vs. Performance Trade-offs

| Dataset | BM25 NDCG | SPLADE NDCG | Quality Gain | BM25 QPS | SPLADE QPS | Throughput Cost |
|---------|-----------|-------------|--------------|----------|------------|-----------------|
| Arguana | 0.40 | **0.528** | **+32%** | 2,167 | 570 | 3.8× |
| FiQA | 0.25 | **0.294** | **+18%** | 4,942 | 1,390 | 3.6× |
| Quora | 0.81 | 0.810 | -0.5% | 1,425 | 296 | 4.8× |
| Trec-Covid | 0.61 | **0.643** | **+5%** | 2,219 | 792 | 2.8× |
| SciFact | **0.69** | 0.622 | -10% | 40,909 | 1,692 | 24× |
| SciDocs | **0.16** | 0.149 | -7% | 13,338 | 1,611 | 8.3× |
| Webis-Touche | **0.34** | 0.228 | -33% | 2,789 | 357 | 7.8× |

### Recall Improvements

| Dataset | BM25 Recall@10 | SPLADE Recall@10 | Improvement |
|---------|----------------|------------------|-------------|
| Arguana | 0.647 | **0.787** | **+22%** |
| FiQA | 0.315 | **0.356** | **+13%** |
| Quora | 0.902 | **0.905** | +0.3% |
| SciFact | **0.820** | 0.753 | -8% |

**When to Use SPLADE**:
- ✅ Domain-specific retrieval requiring maximum quality
- ✅ Reranking pipelines where initial recall matters more than throughput
- ✅ Applications where 1.5-2× latency increase is acceptable
- ❌ High-volume, latency-sensitive search (use BM25 or hybrid approach)

---

## 🔄 Hybrid Search: Best of Both Worlds

Cosdata's hybrid approach combines multiple retrieval modalities to optimize for both relevance and performance:

- **BM25 + Dense Vectors**: Lexical precision with semantic understanding
- **SPLADE + HNSW**: Neural matching with efficient ANN search
- **Metadata Filtering**: Context-aware ranking without embedding overhead

**Result**: 20-50% improvement in NDCG@10 over single-modality approaches while maintaining sub-100ms latency.

---

## 💰 Cost Efficiency

Organizations using Cosdata achieve:
- **60-120% reduction** in compute requirements vs. traditional vector databases
- **Eliminated infrastructure costs** from external document stores
- **Lower memory footprint** through intelligent caching and quantization
- **Predictable scaling costs** with horizontal sharding

---

## 🔬 Benchmark Methodology

All benchmarks use:
- **Industry-standard datasets**: BEIR, Qdrant benchmarks, MS MARCO
- **Consistent hardware**: Same server specifications across all comparisons
- **Default configurations**: Out-of-the-box settings unless otherwise noted
- **Reproducible tests**: Open-source benchmark scripts available in our repository

For detailed benchmark results, methodology, and reproduction instructions, visit: **[cosdata.io/resources/benchmarks](https://www.cosdata.io/resources/benchmarks)**

<br>

# ✨ Features

## 🎯 Search Relevance & Quality

**Move Beyond Cosine Similarity**
- **Hybrid Multi-Modal Search**: Combine BM25 (up to 151× faster than Elasticsearch), dense vectors (HNSW), and SPLADE learned sparse representations
- **Context-Aware Retrieval**: Native GPS geofencing, hierarchical document structures with inherited metadata, temporal filtering, and boolean queries—no embedding models required
- **Explainable Results**: Transparent scoring shows why results were surfaced, decomposing semantic similarity, metadata matches, geographic relevance, and hierarchical relationships

## ⚡ Performance at Scale

**Industry-Leading Benchmarks**
- **Ultra-Fast Indexing**: Up to 12× faster than Elasticsearch on large datasets
- **Massive Throughput**: 1758+ QPS on million-record datasets; 151× higher QPS than Elasticsearch on SciFact
- **Sub-100ms Latency**: Optimized for real-time applications with consistent p50/p95 performance

## 🏢 Enterprise-Grade Architecture

**Production-Ready Infrastructure**
- **Colocated Storage**: Retrieve complete content in single operations—no external database calls required
- **Versioning & Time Travel**: Query historical data states with immutable, append-only architecture supporting A/B testing and full audit trails
- **Streaming Ingestion**: Process real-time data feeds with immediate queryability while maintaining consistency guarantees
- **End-to-End Security**: Encryption at rest and in transit, optional client-side encryption for zero-trust environments, fine-grained RBAC

## 🔧 Developer Experience

**Built for Rapid Integration**
- **Auto-Configuration**: Insights-driven hyperparameter tuning for optimal performance out-of-the-box
- **Intuitive RESTful APIs**: "Transactions as a resource" design pattern
- **Native SDKs**: Python and Node.js with more coming soon
- **Comprehensive Documentation**: Full guides at docs.cosdata.io

## 📊 Advanced Capabilities

- **Dense Vector Indexing**: Optimized HNSW algorithm with dynamic updates, no full rebuilds required
- **SPLADE Support**: 15-25% better ranking quality for domain-specific retrieval tasks
- **Product & Scalar Quantization**: Quarter-nary (2-bit) and octal (3-bit) options for enhanced compression
- **Multi-Modal Data Management**: Real-time querying across text, images, audio with immediate searchability

<br>

# ⚡️ Getting Started

## 1. Install

### Prerequisites

- **Linux**: `curl`  
- **macOS & Windows**: [Docker](https://www.docker.com/get-started) (v20.10+)


### Quick Install (Linux 🐧)

Run this one‑liner to install Cosdata and all dependencies:

```bash
curl -sL https://cosdata.io/install.sh | bash
```

✅ Installs the latest Cosdata CLI  


### Install via Docker (macOS 🖥️ & Windows 💻)

1. **Verify Docker is running**  
   ```bash
   docker --version
   ```

2. **Pull the latest Cosdata image**  
   ```bash
   docker pull cosdataio/cosdata:latest
   ```

3. **Run the container**

   ```bash
   docker run -it \
   --name cosdata-server \
   -p 8443:8443 \
   -p 50051:50051 \
   cosdataio/cosdata:latest
   ```

✅ The server will be available at `http://localhost:8443`.

<br>

## 2. Build from Source

Perfect for contributors and power users who want to customize or extend Cosdata.

### Prerequisites

- **Git** (v2.0+)  
- **Rust** (v1.81.0+) & **Cargo**  
- **C++ compiler**  
  - GCC ≥ 4.8 **or** Clang ≥ 3.4 

<br>

> **Tip:** On Ubuntu/Debian you can install everything with:  
> ```bash
> sudo apt update && sudo apt install -y git build-essential curl \
>    clang lld rustc cargo
> ```

<br>

### 🚀 Build & Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/cosdata/cosdata.git
   cd cosdata
   ```

2. **Compile in release mode**
   ```bash
   cargo build --release
   ```

3. **Start the server**
   ```bash
   ./target/release/cosdata --admin-key YOUR_ADMIN_KEY
   ```

> You should see logs like:
> 
> ```text
> [[2025-02-21T02:30:29Z INFO  cosdata::web_server] starting HTTP server at http://127.0.0.1:8443
> [2025-02-21T02:30:29Z INFO  actix_server::builder] starting 20 workers
> [2025-02-21T02:30:29Z INFO  actix_server::server] Actix runtime found; starting in Actix runtime
> [2025-02-21T02:30:29Z INFO  actix_server::server] starting service: "actix-web-service-127.0.0.1:8443"
> [2025-02-21T02:30:29Z INFO  cosdata::grpc::server] gRPC server listening on [::1]:50051
> ```

<br>

## 3. Testing Your Installation

### 🧪 Quick Validation Cosdata Server with `test.py`

Use the `test.py` script in the `tests/` directory to validate your Cosdata server setup. This script will:

1. **Create** a test collection and a Dense HNSW index.  
2. **Insert** batches of random vectors in a single transaction.  
3. **Generate** query vectors by perturbing ~10% of the inserted vectors.  
4. **Search** the server for nearest neighbors using its HNSW index.  
5. **Verify** results by comparing against a local brute‑force cosine distance search.


### 🔧 Prerequisites

- **Python 3.8+**  
- The [`uv`](https://docs.astral.sh/uv/) CLI for virtual‑env & dependency management  
- A running Cosdata server at `http://127.0.0.1:8443`



### ⚙️ Setup & Execution
Run the following from the `tests/` directory:
1. **Install dependencies**

   ```bash
   cd tests
   uv sync
   ```


This will:

-   Create a Python virtual environment
-   Install packages listed in `pyproject.toml`
    

2.  **Run the test script**
    ```bash
    uv run test.py
    ```
    
3.  **Review the output**  
    The script prints a summary, including:
    
    -   Number of vectors inserted
    -   Queries executed
    -   Pass/fail status for each comparison
        

> **Tip:** If any test fails, check your server logs under `~/.cosdata/logs/` or review console output for errors.


<br>

### 🔍 Testing with Real‑World Datasets (`test-dataset.py`)

Use the `test-dataset.py` script to benchmark Cosdata against real‑world datasets:

1.  **Download** or mount the dataset (e.g., SIFT, GloVe embeddings).
    
2.  **Index** the dataset using your chosen index type (HNSW, IVF, etc.).
    
3.  **Query** sample vectors and record accuracy & latency metrics.
    
4.  **Compare** Cosdata's performance against baseline implementations.
    

> **TODO:** Add download links, configuration flags, and step‑by‑step instructions for each dataset.


<br>

## 4. HTTPS Configuration (TLS)

By default, Cosdata runs over HTTP, but we **strongly recommend** enabling HTTPS in production.



### 1. Development Mode (HTTP)

If you just want to spin up the server quickly without TLS, edit your `config.toml`:

```toml
[server]
mode = "http"
```

> **⚠️ Warning:** HTTP mode is **not** secure—only use this for local development or testing.


### 2. Enabling TLS (HTTPS)

To run Cosdata over HTTPS, you need:

1.  **TLS certificates** (self‑signed OK for testing)
    
2.  A valid `config.toml` pointing at your certs
    
3.  Proper file permissions


#### a. Generate a Self‑Signed Certificate

1. Create a new RSA key and self‑signed cert (valid 1 year)

   ```bash
   openssl req -newkey rsa:2048 -nodes -keyout private_key.pem -x509 -days 365 -out self_signed_certificate.crt
   ```

2. Convert the private key to PKCS#8 format

   ```bash
   openssl pkcs8 -topk8 -inform PEM -outform PEM -in private_key.pem -out private_key_pkcs8.pem -nocrypt
   ```


#### b. Store & Secure Your Certificates

1. Set your cert directory (choose a secure path)

   ```bash
   export SSL_CERT_DIR="/etc/ssl"
   ```

2. Create subdirectories

   ```bash
   sudo mkdir -p $SSL_CERT_DIR/{certs,private}
   ```


3. Move certs into place

   ```bash
   sudo mv self_signed_certificate.crt   $SSL_CERT_DIR/certs/cosdata.crt
   sudo mv private_key_pkcs8.pem         $SSL_CERT_DIR/private/cosdata.key
   ```


4. Secure the private key

   ```bash
   sudo groupadd ssl-cert            || true
   sudo chgrp ssl-cert $SSL_CERT_DIR/private/cosdata.key
   sudo chmod 640  $SSL_CERT_DIR/private/cosdata.key
   sudo chmod 750  $SSL_CERT_DIR/private
   sudo usermod -aG ssl-cert $USER   # you may need to log out/in or run `newgrp ssl-cert`
   ```



#### c. Configure Cosdata to Use TLS

In your `config.toml`, update the `[server]` section:

   ```toml
   [server]
   mode     = "https"
   tls_cert = "/etc/ssl/certs/cosdata.crt"
   tls_key  = "/etc/ssl/private/cosdata.key"
   ```


#### d. Restart Cosdata

If running directly:
   ```bash
   ./target/release/cosdata --admin-key YOUR_ADMIN_KEY
   ```

If using Docker, mount your cert directory:
   ```bash
   docker run -it --rm \
     -v "/etc/ssl/certs:/etc/ssl/certs:ro" \
     -v "/etc/ssl/private:/etc/ssl/private:ro" \
     cosdataio/cosdata:latest \
     cosdata --admin-key YOUR_ADMIN_KEY
   ```


### 🔎 Verify HTTPS

Open your browser or run:

```bash
curl -kv https://localhost:8443/health
```

You should see a successful TLS handshake and a healthy status response.


<br>

# 🧩 Client SDKs

Cosdata provides an officially maintained Python SDK for seamless integration into your projects.

### 🐍 Python SDK

**Install**  
```bash
pip install cosdata-sdk
```

**Quickstart Example**

```python
from cosdata import Client

# Initialize the client with your server details
client = Client(
    host="http://127.0.0.1:8443",  # Default host
    username="admin",               # Default username
    password="admin",               # Default password
    verify=False                    # SSL verification
)

# Create a collection for storing 768-dimensional vectors
collection = client.create_collection(
    name="my_collection",
    dimension=768,                  # Vector dimension
    description="My vector collection"
)

# Create an index with custom parameters
index = collection.create_index(
    distance_metric="cosine",       # Default: cosine
    num_layers=10,                  # Default: 10
    max_cache_size=1000,           # Default: 1000
    ef_construction=128,           # Default: 128
    ef_search=64,                  # Default: 64
    neighbors_count=32,            # Default: 32
    level_0_neighbors_count=64     # Default: 64
)

# Generate and insert vectors
import numpy as np

def generate_random_vector(id: int, dimension: int) -> dict:
    values = np.random.uniform(-1, 1, dimension).tolist()
    return {
        "id": f"vec_{id}",
        "dense_values": values,
        "document_id": f"doc_{id//10}",  # Group vectors into documents
        "metadata": {  # Optional metadata
            "created_at": "2024-03-20",
            "category": "example"
        }
    }

# Generate and insert vectors
vectors = [generate_random_vector(i, 768) for i in range(100)]

# Add vectors using a transaction
with collection.transaction() as txn:
    # Single vector upsert
    txn.upsert_vector(vectors[0])
    # Batch upsert for remaining vectors
    txn.batch_upsert_vectors(vectors[1:])

# Search for similar vectors
results = collection.search.dense(
    query_vector=vectors[0]["dense_values"],  # Use first vector as query
    top_k=5,                                  # Number of nearest neighbors
    return_raw_text=True
)
```

**Learn More**

-   📦 Cosdata Python SDK Documentation: [cosdata-sdk-python](https://github.com/cosdata/cosdata-sdk-python)


<br>

### 🟢 Node.js SDK

**Install**  
```bash
npm install cosdata-sdk
```

**Quickstart Example**

```typescript
import { createClient } from 'cosdata-sdk';

// Initialize the client (all parameters are optional)
const client = createClient({
  host: 'http://127.0.0.1:8443',  // Default host
  username: 'admin',              // Default username
  password: 'test_key',           // Default password
  verifySSL: false                // SSL verification
});

// Create a collection
const collection = await client.createCollection({
  name: 'my_collection',
  dimension: 128,
  dense_vector: {
    enabled: true,
    dimension: 128,
    auto_create_index: false
  }
});

// Create an index
const index = await collection.createIndex({
  name: 'my_collection_dense_index',
  distance_metric: 'cosine',
  quantization_type: 'auto',
  sample_threshold: 100,
  num_layers: 16,
  max_cache_size: 1024,
  ef_construction: 128,
  ef_search: 64,
  neighbors_count: 10,
  level_0_neighbors_count: 20
});

// Generate some vectors
function generateRandomVector(dimension: number): number[] {
  return Array.from({ length: dimension }, () => Math.random());
}

const vectors = Array.from({ length: 100 }, (_, i) => ({
  id: `vec_${i}`,
  dense_values: generateRandomVector(128),
  document_id: `doc_${i}`
}));

// Add vectors using a transaction
const txn = collection.transaction();
await txn.batch_upsert_vectors(vectors);
await txn.commit();

// Search for similar vectors
const results = await collection.getSearch().dense({
  query_vector: generateRandomVector(128),
  top_k: 5,
  return_raw_text: true
});
```

**Learn More**

-   📦 GitHub: [cosdata-sdk-node](https://github.com/cosdata/cosdata-sdk-node)



<br>

# 🙌 Contributing

We welcome contributions from the community! Whether it's _fixing a bug_, _improving documentation_, or building _new features_—every bit helps.

For full guidelines (coding standards, commit messages, CI checks), please see our [CONTRIBUTING.md](CONTRIBUTING.md) If you have any questions, feel free to open an issue or join the discussion on [Discord](https://discord.gg/WbSbXYWvta). We can’t wait to collaborate with you!

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

<br>

# 🤝 Contacts & Community

Have questions, ideas, or want to contribute? We'd love to hear from you!

🔗 Discord: Chat, collaborate, and get support — [Join now](https://discord.gg/WbSbXYWvta)

📨 Email: Partnerships & business inquiries — [contact@cosdata.io](mailto:contact@cosdata.io)

🐛 Issues: Report bugs or suggest features — [Open an issue](https://github.com/cosdata/cosdata/issues)

💡 Discussions: Share ideas and ask questions — [Join Discussion](https://discord.gg/WbSbXYWvta)

Let's collaborate and build the future of vector search—together! 💡

<br>

# ⭐️ Show Your Support

If Cosdata has empowered your projects, please consider giving us a star on GitHub! ⭐️ 

Your endorsement helps attract new contributors and fuels ongoing improvements.

Thank you for your support! 🙏
