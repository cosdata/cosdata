# Cosdata

Cosdata is a versatile vector database solution featuring high performance, immutability, and versioning capabilities, catering to a wide range of AI and machine learning applications. Cosdata supports embedding-based search and similarity queries, real-time vector search at scale, and seamless integration with existing ML pipelines.

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
