<p align="center">
  <img src="org/logo.svg" alt="Cosdata" style="max-width: 100%; height: auto;">
</p>

<p align="center">
    <b>Next-gen vector database delivering lightning-fast performance at billion-vector scale.</b>
</p>
<p align="center">
  <a href="https://cosdata.io"><img src="https://flat.badgen.net/badge/www/cosdata.io/pink"></a>
  <a href="https://github.com/cosdata/cosdata/actions"><img src="https://flat.badgen.net/badge/build/passing/green"></a>
  <a href="https://www.rust-lang.org/"><img src="https://flat.badgen.net/badge/language/%F0%9F%A6%80%20Rust/yellow"></a>
  <a href="https://www.rust-lang.org/"><img src="https://flat.badgen.net/badge/language/%F0%9F%90%8D%20Python/black"></a>
  <br>
  <a href="https://discord.gg/WbSbXYWvta"><img src="https://flat.badgen.net/discord/members/WbSbXYWvta?icon=discord"></a>
  <a href="https://www.linkedin.com/company/cosdata/"><img alt="Profile" src="https://img.shields.io/badge/our_journey-LinkedIn-blue"/></a>
  <a href="https://github.com/cosdata/cosdata/blob/master/LICENSE"><img src="https://flat.badgen.net/static/license/Apache-2.0"></a> 
  <a href="https://github.com/cosdata/cosdata/pulls"><img src="https://flat.badgen.net/badge/PRs/open/pink"></a> 
</p>
</br>
<p></p>

## 📦 Table of Contents

- [Overview](#-overview)
- [Getting Started](#️-getting-started)
- [Client SDKs](#-client-sdks)
  - [Python SDK](#-python-sdk)
  - [Node.js SDK](#-nodejs-sdk)
- [Features](#-features)
- [Benchmarks](https://www.cosdata.io/resources/benchmarks)
- [Documentation](https://docs.cosdata.io/getting-started/introduction/)
- [Contributing](#-contributing)
- [Contacts & Community](#-contacts--community)
- [Show Your Support](#️-show-your-support)

<br>
<br>

# 🚀 Overview

Cosdata is cutting-edge AI data platform engineered for exceptional performance. Featuring immutability and version control, it's designed to elevate a wide range of AI and machine learning projects.

Cosdata is at the forefront of advancing search technology, excelling in critical areas that will define the next generation of search pipelines:
- **Semantic Search**: Leverage embedding-based hybrid search, seamlessly managing both dense and sparse vectors to deliver deep semantic analysis.
- **Structured Knowledge Graphs**: Sophisticated context retrieval by leveraging structured knowledge graphs.
- **Hybrid Search Capabilities**: Combine explicit relationship queries with vector similarity search.
- **Real-Time Search at Scale**: Execute real-time vector search with unmatched scalability, ensuring that your search pipeline performs flawlessly under any load.
- **ML Pipeline Integration**: Enjoy seamless integration with your existing machine learning pipeline, enhancing your search capabilities without disrupting your workflow.

Cosdata is designed to meet the demands of modern search applications, empowering businesses to harness the full potential of their data.

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
   docker pull cosdatateam/cosdata:latest
   ```

3. **Run the container**

   ```bash
   docker run -it \
   --name cosdata-server \
   -p 8443:8443 \
   -p 50051:50051 \
   cosdatateam/cosdata:latest
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

Cosdata includes a `test.py` script in the top level directory, to validate your server setup. This script will:

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

1. **Install dependencies**

   ```bash
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
     cosdatateam/cosdata:latest \
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
pip install cosdata-client
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

# ✨ Features

### Search Relevance
- **Hybrid Search**: Enhance search precision with our vector database, leveraging the power of combined sparse and dense vector searches to deliver highly relevant, context-rich results for complex queries.
- **Knowledge-Graph**: Improve the relevance of your search results by seamlessly combining structured information from knowledge graphs with the nuanced semantics of vector embedding, enabling our vector database to deliver richer and more relevant insights.

### High performance
- **Indexing**: Experience lightning-fast indexing with our vector database, optimized for handling high-dimensional data at scale. Our advanced indexing algorithms ensure that your sparse and dense vectors are always ready for instant querying, no matter how large or complex your dataset grows.
- **Latency**: Power your applications with lightning-fast search performance—our vector database is engineered to deliver exceptionally fast query responses.
- **Requests per second**: Achieve industry-leading concurrent requests per second (RPS) across different indices with an advanced architecture designed for optimal performance under heavy loads.

### Customizable
- **Configurability**: Gain precise control over your setup with manual configuration of all indexing and querying hyperparameters, enabling you to optimize performance, resource utilization and tailor results to your exact specifications.
- **Dense Vector indexing**: Achieve efficient and precise indexing with our vector database's optimized HNSW (Hierarchical Navigable Small World) algorithm, designed to enhance search performance and accuracy for large-scale data sets.
- **Sparse vectors**: Expertly designed to work seamlessly with SPLADE-generated sparse vectors, our solution offers superior performance compared to BM25 indices for more precise and meaningful insights.

### Scalability
- Unlock unbounded scalability with our vector database, engineered to grow alongside your data and query demands. Whether you're handling millions of records or scaling up to massive datasets, enjoy consistent, high-speed performance without compromise
- Achieve predictable and efficient query performance with our vector database, engineered for near-linear scalability that ensures fast results, even as your data expands.

### Efficient
- **Resource utilization**: Efficiency is at the core of our vector database, where ingenious provably efficient data structures and algorithms ensure outstanding performance while providing increasingly relevant search results.
- **Scalar quantization**: Configure finer quantization resolutions, including quarter-nary (2-bit) and octal (3-bit), for enhanced compression and improved recall trade-offs, giving you more control over data efficiency and performance.
- **Product quantization**: A pioneering product quantization approach to not only compress data more effectively but also enhance recall performance beyond what scalar quantization offers, optimizing both data efficiency and retrieval recall.

### Enterprise-grade
- **Data isolation**: Experience enterprise-grade privacy with our vector database, providing robust data isolation to ensure secure and consistent access.
- **Data security**: Achieve reliable data security with our vector database, designed with robust safeguards such as role-based-access-control to protect your information and maintain its integrity against unauthorized access and threats.
- **Multiple deployment modes**: Deploy our vector database in various environments—whether at the edge, in a private cloud, public cloud, or serverless setup—providing you with flexible, scalable solutions to meet your unique operational needs.
- **Reliability**: Cosdata delivers reliable performance and data integrity with fault-tolerance, backup, and recovery features, designed to meet enterprise demands and ensure uninterrupted operation.
- **Versioning**: Experience Git-like version control with our vector database, enabling you to compare search performance, use time travel to access past states, audit data, and effortlessly roll back when necessary.

### Easy to use
- **Auto-configuration of hyper-parameters**: Achieve peak performance with our vector database, utilizing insights-driven auto-configuration of hyperparameters to automatically fine-tune your system for the best results, no manual adjustments needed.
- **Intuitive API**: Elegantly crafted HTTP Restful APIs featuring _"Transactions as a resource"_. Manage all functions of our vector database effortlessly with intuitive HTTP RESTful APIs.
- **Client SDKs in your favourite language**: Access our vector database effortlessly with client SDKs available in multiple programming languages.
- **Powerful and expressive cosQuery language**: Leverage cosQuery, a powerful and expressive declarative query language, to seamlessly query data across vector embedding and knowledge graph, enabling deep and nuanced insights into your data.

### Manage Multi-modal data
- Supports real-time querying and dynamic index updates, ensuring that new multi-modal data (text, images, audio, etc.) is immediately searchable without downtime or delays.

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