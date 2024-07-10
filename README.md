# Cosdata

Cosdata is a versatile vector database solution featuring high performance, immutability, and versioning capabilities, catering to a wide range of AI and machine learning applications. Cosdata supports embedding-based search and similarity queries, real-time vector search at scale, and seamless integration with existing ML pipelines.

# Linux Installation Guide

This guide provides step-by-step instructions for installing Cosdata on Linux systems.

## Prerequisites


Before installing Cosdata, ensure you have the following:

- Git
- Rust (latest stable version)
- Cargo (Rust's package manager)
- A C++ compiler (GCC 4.8+ or Clang 3.4+)
- CMake (3.10+)

## Installation Steps

### RocksDB dependencies:

```
sudo apt install libgflags-dev libsnappy-dev liblz4-dev libzstd-dev
```

### Install RocksDB:

```
sudo apt install librocksdb-dev
```

## Building Cosdata

Clone the Cosdata repository:
``` 
git clone https://github.com/cosdata/cosdata.git
cd cosdata
```

Build the project:
```
cargo build --release
```
