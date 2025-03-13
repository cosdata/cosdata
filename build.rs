fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "grpc-server")]
    {
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

        let config = tonic_build::configure()
            .build_server(true)
            .build_client(false)
            .file_descriptor_set_path(out_dir.join("vector_service.bin"));

        config.compile_protos(&["proto/vector_service.proto"], &["proto/"])?;
    }

    Ok(())
}
