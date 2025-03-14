pub mod auth;
pub mod collections;
pub mod error;
pub mod indexes;
pub mod metadata;
pub mod server;
pub mod vectors;

#[cfg(feature = "grpc-server")]
pub mod proto {
    tonic::include_proto!("vector_service");

    pub const FILE_DESCRIPTOR_SET: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/vector_service.bin"));
}
