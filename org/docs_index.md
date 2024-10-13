# Index of Documentation


1. [Vector Database API Design](http-api.org)
2. [LazyItems Serialization Structure](lazy_items.org)
3. [RESTful Vector Database API Documentation](RESTful-api.org)
4. [Separation of Storage and Compute in Cosdata](Separation_of_Storage_and_Compute.org)
5. [Inverted Index Building and Querying](inverted_index_design.org)
6. [Inverted Index Statistics](inverted_index_stats.org)
7. [Inverted Index Implementation in Rust](inverted_index_ds.org)
8. [Hierarchical Version Control System Design](versioning.org)
9. [Quaternary Multiplication Using Bitwise Operations](quaternary_multiplication.org)
10. [Cosdata Vector Database: Index Creation Approaches](indexing.org)
11. [MergedNode Serialization Structure](node_serialization.org)
12. [Transaction as a Resource in RESTful APIs](transactions.org)
13. [LMDB Usage in Cosdata](lmdb.org)


## [Vector Database API Design](http-api.org)

This comprehensive guide outlines the API design for a vector database system. It covers various aspects such as collection creation, index creation, namespace creation, and vector structure. The document also provides detailed examples of metadata schemas, vector insertion APIs, and query APIs, including support for both dense and sparse vectors.

## [LazyItems Serialization Structure](lazy_items.org)

This document describes the serialization structure for LazyItems. It provides a detailed breakdown of the file layout, chunk structure, and LazyItem structure. The document also covers the serialization and deserialization processes, explaining how data is organized and accessed efficiently within the system.

## [RESTful Vector Database API Documentation](RESTful-api.org)

This documentation outlines a RESTful API for a vector database system. It provides detailed information about various endpoints, including those for managing collections, vectors, and transactions. The document covers authentication requirements and includes sample request and response structures for each endpoint, offering a comprehensive guide for interacting with the vector database through a RESTful interface.

## [Separation of Storage and Compute in Cosdata](Separation_of_Storage_and_Compute.org)

This document delves into the architectural approach of separating storage and compute in the Cosdata project. It explains the core components of the storage and compute layers, discusses key aspects of the separation, and outlines the benefits and challenges of this approach. The document also covers implementation details of various components and concludes with insights on how this architecture enables a highly scalable and efficient vector database system.

## [Inverted Index Building and Querying](inverted_index_design.org)

This document outlines the process of building an inverted index using a provided graph data structure, querying it efficiently using SIMD operations, and returning the top K candidates with their highest dot product scores. It covers the inverted index structure, building process, querying process, and includes code snippets for implementation.

## [Inverted Index Statistics](inverted_index_stats.org)

This document provides insights into the statistics of implicit vs explicit nodes in an inverted index implementation. It includes Rust code for generating and analyzing these statistics, explaining how the counting process works and what insights can be gained from the results.

## [Inverted Index Implementation in Rust](inverted_index_ds.org)

This document details the implementation of an inverted index data structure in Rust. It covers the main components such as InvertedIndexItem and InvertedIndex, helper functions, and the traversal logic. The document also discusses key concepts like thread-safe implementation, recursive tree traversal, and the use of powers of 4 for efficient representation.

## [Hierarchical Version Control System Design](versioning.org)

This document describes the design of a hierarchical version control system. It covers the implementation of a custom XOR-based hashing algorithm for generating unique, deterministic hashes for version numbers across different branches. The document includes details on core components, usage examples, key features, and considerations for the version control system.

## [Quaternary Multiplication Using Bitwise Operations](quaternary_multiplication.org)

This document explains the process of performing quaternary multiplication using bitwise operations and lookup tables. It covers the objective, partial products table, lookup tables for weights, and different methods for efficient computation, including SIMD lookup and accumulation approaches. The document also discusses optimization techniques for various scenarios.

## [Cosdata Vector Database: Index Creation Approaches](indexing.org)

This document outlines three flexible approaches to index creation in Cosdata: explicit pre-insertion, implicit, and explicit post-insertion. It compares these approaches with other vector database systems and discusses the advantages, use cases, and considerations for each method. The document also covers performance implications, resource utilization, and monitoring aspects of index creation.

## [MergedNode Serialization Structure](node_serialization.org)

This document describes the byte-wise serialized structure of a MergedNode in the vector database system. It provides a detailed diagram of the serialization layout, explains each component of the structure, and discusses key features such as lazy loading and flexibility. The document also outlines implementation tasks for optimizing the serialization process.

## [Transaction as a Resource in RESTful APIs](transactions.org)

This document explores the concept of treating transactions as resources in RESTful APIs. It outlines the approach for creating, working with, committing, and aborting transactions using standard HTTP methods. The document provides examples of API endpoints for various transactional operations and discusses the benefits and considerations of this approach in the context of a vector database system.

## [LMDB Usage in Cosdata](lmdb.org)

This document details how Cosdata utilizes LMDB (Lightning Memory-Mapped Database) for storing various metadata. It describes the structure of different databases within LMDB, including metadata, embeddings, versions, and branches. The document explains the purpose of each database and provides insights into how vector embeddings and their offsets are stored and managed.
