pub mod collections;
pub mod error;
pub mod server;
pub mod vectors;

#[cfg(feature = "grpc-server")]
pub mod proto {
    tonic::include_proto!("vector_service");
}
