fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = tonic_build::configure()
        .build_server(true)
        .build_client(false);

    #[cfg(feature = "grpc-server")]
    {
        // Add descriptor set path only when grpc-server feature is enabled
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        config = config.file_descriptor_set_path(out_dir.join("vector_service.bin"));
    }

    config.compile(&["proto/vector_service.proto"], &["proto/"])?;

    Ok(())
}
