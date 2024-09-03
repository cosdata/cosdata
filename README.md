<p align="center">
  <img height="160" src="org/logo.svg" alt="Cosdata">
</p>

<p align="center">
    <b>RAG 2.0 engine to power AI with hybrid search and knowledge graphs for next-level intelligence.</b>
</p>
<p align="center">
  <a href="https://github.com/cosdata/cosdata/actions"><img src="https://flat.badgen.net/badge/build/passing/green"></a>
  <a href="https://github.com/cosdata/cosdata/blob/master/LICENSE"><img src="https://flat.badgen.net/static/license/Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://flat.badgen.net/badge/language/%F0%9F%A6%80%20Rust/yellow"></a>
  <a href="https://discord.gg/WbSbXYWvta"><img src="https://flat.badgen.net/discord/members/WbSbXYWvta?icon=discord"></a>
</p>
</br>
<p></p>

Introducing **Cosdata**, an advanced AI data platform engineered for high performance, immutability, and versioning, designed to elevate a wide range of AI and machine learning applications. As a vector database, Cosdata excels in embedding-based and hybrid search, handling both dense and sparse vectors for comprehensive semantic search. It provides real-time vector search at scale and integrates seamlessly with existing ML pipelines. Leveraging structured knowledge graphs, Cosdata utilizes sophisticated context retrieval methods to query explicit relationships and apply vector similarity search, ensuring the delivery of the most relevant and actionable insights.

### Manage Multi-modal data

- Supports real-time querying and dynamic index updates, ensuring that new multi-modal data (text, images, audio, etc.) is immediately searchable without downtime or delays.

### High performance

- **Indexing** Experience lightning-fast indexing with our vector database, optimized for handling high-dimensional data at scale. Our advanced indexing algorithms ensure that your sparse and dense vectors are always ready for instant querying, no matter how large or complex your dataset grows.

- **Latency** Power your applications with lightning-fast search performance—our vector database is engineered to deliver exceptionally fast query responses.

- **Requests per second** Achieve industry-leading concurrent requests per second (RPS) across different indices with an advanced architecture designed for optimal performance under heavy loads.

### Scalability

- Unlock unbounded scalability with our vector database, engineered to grow alongside your data and query demands. Whether you're handling millions of records or scaling up to massive datasets, enjoy consistent, high-speed performance without compromise
  
- Achieve predictable and efficient query performance with our vector database, engineered for near-linear scalability that ensures fast results, even as your data expands.

### Search Relevance

- **Hybrid Search** Enhance search precision with our vector database, leveraging the power of combined sparse and dense vector searches to deliver highly relevant, context-rich results for complex queries.

- **Knowledge-Graph** Improve the relevance of your search results by seamlessly combining structured information from knowledge graphs with the nuanced semantics of vector embedding, enabling our vector database to deliver richer and more relevant insights.
  
## Linux Installation Guide

This guide provides step-by-step instructions for installing Cosdata on Linux systems.

### Prerequisites

Before installing Cosdata, ensure you have the following:

- Git
- Rust (latest stable version)
- Cargo (Rust's package manager)
- A C++ compiler (GCC 4.8+ or Clang 3.4+)
- CMake (3.10+)

### Installation Steps

#### Building Cosdata

Clone the Cosdata repository:

```
git clone https://github.com/cosdata/cosdata.git
cd cosdata
```

Build the project:

```
cargo build --release
```

### Self Signed Certificates

It's recommended to run Cosdata server in HTTPS mode i.e. with TLS
support. However, during development it might be easier to get it
running without TLS. To do so, set `server.mode=http` in the
[config.toml](config.toml) file.

Alternately, you may use self-signed certificates for testing the
APIs. The paths to the certificate and private key files are
configured in the [config.toml](config.toml) file.

This sections mentions how you can generate and setup the
certificates.

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
