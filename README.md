<p align="center">
  <img src="org/logo.svg" alt="Cosdata" style="max-width: 100%; height: auto;">
</p>


<p align="center">
    <b>The future ready AI data platform to power the next generation search pipelines.</b>
</p>
<p align="center">
  <a href="https://github.com/cosdata/cosdata/actions"><img src="https://flat.badgen.net/badge/build/passing/green"></a>
  <a href="https://github.com/cosdata/cosdata/blob/master/LICENSE"><img src="https://flat.badgen.net/static/license/Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://flat.badgen.net/badge/language/%F0%9F%A6%80%20Rust/yellow"></a>
  <a href="https://discord.gg/WbSbXYWvta"><img src="https://flat.badgen.net/discord/members/WbSbXYWvta?icon=discord"></a>
</p>
</br>
<p></p>

**Cosdata**: A cutting-edge AI data platform engineered for exceptional performance. Featuring immutability and version control, it's designed to elevate a wide range of AI and machine learning projects.

Cosdata is at the forefront of advancing search technology, excelling in critical areas that will define the next generation of search pipelines:
- **Semantic Search**: Leverage embedding-based hybrid search, seamlessly managing both dense and sparse vectors to deliver deep semantic analysis.
- **Structured Knowledge Graphs**: Sophisticated context retrieval by leveraging structured knowledge graphs.
- **Hybrid Search Capabilities**: Combine explicit relationship queries with vector similarity search.
- **Real-Time Search at Scale**: Execute real-time vector search with unmatched scalability, ensuring that your search pipeline performs flawlessly under any load.
- **ML Pipeline Integration**: Enjoy seamless integration with your existing machine learning pipeline, enhancing your search capabilities without disrupting your workflow.

Cosdata is designed to meet the demands of modern search applications, empowering businesses to harness the full potential of their data.

<p align="center">
<strong><a href="#features">Features</a> • <a href="#getting-started">Getting Started</a> • <a href="org/docs_index.md">Docs</a> • <a href="#contacts">Contact Us</a> • <a href="#show-your-support">Show Your Support</a>
</strong>
</p>

## Features

### Search Relevance
- **Hybrid Search**: Enhance search precision with our vector database, leveraging the power of combined sparse and dense vector searches to deliver highly relevant, context-rich results for complex queries.
- **Knowledge-Graph**: Improve the relevance of your search results by seamlessly combining structured information from knowledge graphs with the nuanced semantics of vector embedding, enabling our vector database to deliver richer and more relevant insights.

### High performance
- **Indexing**: Experience lightning-fast indexing with our vector database, optimized for handling high-dimensional data at scale. Our advanced indexing algorithms ensure that your sparse and dense vectors are always ready for instant querying, no matter how large or complex your dataset grows.
- **Latency**: Power your applications with lightning-fast search performance—our vector database is engineered to deliver exceptionally fast query responses.
- **Requests per second**: Achieve industry-leading concurrent requests per second (RPS) across different indices with an advanced architecture designed for optimal performance under heavy loads.

### Customizable
- **Configurability**: Gain precise control over your setup with manual configuration of all indexing and querying hyperparameters, enabling you to optimize performance, resource utilization and tailor results to your exact specifications.
- **Dense Vector indexing**: Achieve efficient and precise indexing with our vector database’s optimized HNSW (Hierarchical Navigable Small World) algorithm, designed to enhance search performance and accuracy for large-scale data sets.
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


## Getting Started

This guide provides step-by-step instructions for installing Cosdata on Linux systems.

### Prerequisites

Before installing Cosdata, ensure you have the following:

- Git
- Rust (version 1.81.0 or higher)
- Cargo (Rust's package manager, toolchain version 1.81.0 or higher)
- A C++ compiler (GCC 4.8+ or Clang 3.4+)

### Installation Steps

#### Building Cosdata as a developer

Clone the Cosdata repository:

```
git clone https://github.com/cosdata/cosdata.git
cd cosdata
```

Build the project:

```
cargo build --release
```

### Running Cosdata

You can build Cosdata as a user and simply run the Cosdata server, follow the following steps. Easily installable version of Cosdata will be available soon.

Clone the Cosdata repository:

```
git clone https://github.com/cosdata/cosdata.git
cd cosdata
```

Install the binaries:

```
# Compiles the latest code and installs the binary.
cargo install --path .
```

And run the binary
```
cosdata
```

This should start the server and will ask for the Server Key to be entered. You can enter any text key (something simple like 'foobar' is fine for basic testing). You should see log lines similar to the following lines -

```
[2025-02-21T02:30:29Z INFO  cosdata::web_server] starting HTTP server at http://127.0.0.1:8443
[2025-02-21T02:30:29Z INFO  actix_server::builder] starting 20 workers
[2025-02-21T02:30:29Z INFO  actix_server::server] Actix runtime found; starting in Actix runtime
[2025-02-21T02:30:29Z INFO  actix_server::server] starting service: "actix-web-service-127.0.0.1:8443", workers: 20, listening on: 127.0.0.1:8443
[2025-02-21T02:30:29Z INFO  cosdata::grpc::server] gRPC server listening on [::1]:50051
```

### Testing Cosdata Server

#### Testing Cosdata Server with `test.py`

To quickly get started with Cosdata, you can run the `test.py` file available in the top level directory. The `test.py` script -

1. Creates a test collection and a Dense HNSW Index.
2. In a transaction, submits batches of Random vectors to be stored on the server.
3. Uses about 10% of the vectors from the above set as query vectors by adding small perturbations.
4. Issues query to the server to perform the search against the query vectors and performs a brute force search locally using `cosine` distance.
5. Compares the results

In order to run the `test.py` file a few dependencies need to be added. The description here uses [`uv`](https://docs.astral.sh/uv/) for downloading and setting up required dependencies. Perform following steps.

```
# Run uv sync to setup the Python virtual Env and download and install dependencies using `pyproject.toml` file.
uv sync

# Run the test.py file
uv run test.py
```

The script will then perform the steps described above and display the summary of execution


#### Testing Cosdata Server with `test-dataset.py`

You can test Cosdata server using real world datasets. This is performed using `test-dataset.py`.  (TODO: Add details for this).


### Running Cosdata in in HTTPS mode

It's recommended to run Cosdata server in HTTPS mode i.e. with TLS
support. However, during development it might be easier to get it
running without TLS. To do so, set `server.mode=http` in the
[config.toml](config.toml) file.

Alternately, you may use self-signed certificates for testing the
APIs. The paths to the certificate and private key files are
configured in the [config.toml](config.toml) file.

This sections mentions how you can generate and setup the certificates.

#### Generate Certificates

Run the following commands in sequence to get the private key and certificate

```bash
openssl req -newkey rsa:2048 -nodes -keyout private_key.pem -x509 -days 365 -out self_signed_certificate.crt

# Convert the private key to PKCS#8 format
openssl pkcs8 -topk8 -inform PEM -outform PEM -in private_key.pem -out private_key_pkcs8.pem -nocrypt
```

#### Setup Certificates

Set the `SSL_CERT_DIR` environment variable to the folder where you're gonna store the certificates:

```bash
export SSL_CERT_DIR="/etc/ssl"
```

Move certificates to appropriate folders and set permissions:

```bash
# Create directories if don't exist
sudo mkdir -p $SSL_CERT_DIR/{certs,private}

# Move certificates
sudo mv self_signed_certificate.crt $SSL_CERT_DIR/certs/cosdata-ssl.crt
sudo mv private_key_pkcs8.pem $SSL_CERT_DIR/private/cosdata-ssl.key

# Create 'ssl-cert' group (if if doesn't exist already)
sudo groupadd ssl-cert

# Change private key file permissions
sudo chgrp ssl-cert $SSL_CERT_DIR/private/cosdata-ssl.key
sudo chmod 640 $SSL_CERT_DIR/private/cosdata-ssl.key
sudo usermod -aG ssl-cert $USER

# Change private key folder permissions
sudo chmod 750 $SSL_CERT_DIR/private
sudo chgrp ssl-cert $SSL_CERT_DIR/private

# Add yourself to ssl-cert group (you may need to re-login after this)
newgrp ssl-cert
```

## Contacts

- Want to learn more and/or contribute to the project? Join our [Discord channel](https://discord.gg/WbSbXYWvta)
- For business inquiries, please reach us at [contact@cosdata.io](mailto:contact@cosdata.io)

## Show Your Support

⭐️ If you find this project useful, please give it a star on GitHub! ⭐️

Your support helps us improve and continue developing. Thank you!

