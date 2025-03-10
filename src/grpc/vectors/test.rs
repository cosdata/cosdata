#[cfg(test)]
mod tests {
    use std::{fs, sync::Arc};
    use tokio::time::{timeout, Duration};
    use tonic::{Request, Response};

    use crate::app_context::AppContext;
    use crate::config_loader::Config;
    use crate::grpc::collections::CollectionsServiceImpl;
    use crate::grpc::proto::collections_service_server::CollectionsService;
    use crate::grpc::proto::vectors_service_server::VectorsService;
    use crate::grpc::proto::{
        CollectionConfig, CreateCollectionRequest, CreateVectorRequest, DeleteVectorRequest,
        DenseVectorOptions, FindSimilarVectorsRequest, GetVectorRequest, UpdateVectorRequest,
        Vector,
    };
    use crate::grpc::vectors::VectorsServiceImpl;

    fn create_test_config() {
        let test_config = r#"
    upload_threshold = 1000
    upload_process_batch_size = 500
    flush_eagerness_factor = 1.0

    [thread_pool]
    pool_size = 4

    [server]
    host = "127.0.0.1"
    port = 3000
    mode = "http"

    [server.ssl]
    cert_file = "cert.pem"
    key_file = "key.pem"

    [hnsw]
    default_neighbors_count = 8
    default_level_0_neighbors_count = 12
    default_ef_construction = 128
    default_ef_search = 64
    default_num_layer = 16
    default_max_cache_size = 1000

    [indexing]
    mode = "sequential"
    clamp_margin_percent = 0.5

    [search]
    shortlist_size = 100
    "#;
        fs::write("config.toml", test_config).expect("Failed to write test config");
    }

    async fn setup_with_timeout(
    ) -> Result<(VectorsServiceImpl, Arc<AppContext>, String), tokio::time::error::Elapsed> {
        timeout(Duration::from_secs(5), async {
            create_test_config();
            let config = crate::config_loader::load_config()?;
            let context = AppContext::new(config).unwrap();
            let context = Arc::new(context);

            let collection_name = "test_vectors".to_string();
            let collections_service = CollectionsServiceImpl {
                context: context.clone(),
            };

            let create_collection_req = CreateCollectionRequest {
                name: collection_name.clone(),
                description: Some("Test Collection".to_string()),
                dense_vector: Some(DenseVectorOptions { dimension: 4 }),
                sparse_vector: None,
                metadata_schema: None,
                config: Some(CollectionConfig {
                    on_disk_payload: false,
                }),
            };

            collections_service
                .create_collection(Request::new(create_collection_req))
                .await
                .unwrap();

            let vectors_service = VectorsServiceImpl {
                context: context.clone(),
            };

            Ok((vectors_service, context, collection_name))
        })
        .await?
    }

    fn cleanup() {
        let _ = fs::remove_file("config.toml");
    }

    #[tokio::test]
    async fn test_basic_vector_operations() {
        let result = setup_with_timeout().await;
        assert!(result.is_ok(), "Setup timed out");
        let (service, _context, collection_id) = result.unwrap();

        let create_result = timeout(Duration::from_secs(5), async {
            let create_req = CreateVectorRequest {
                collection_id: collection_id.clone(),
                vector: Some(Vector {
                    id: 1,
                    values: vec![0.1, 0.2, 0.3, 0.4],
                }),
            };
            service.create_vector(Request::new(create_req)).await
        })
        .await;

        assert!(create_result.is_ok(), "Create operation timed out");
        let create_response = create_result.unwrap().unwrap();
        assert_eq!(create_response.get_ref().id, 1);

        cleanup();
    }

    #[tokio::test]
    async fn test_vector_update() {
        let result = setup_with_timeout().await;
        assert!(result.is_ok(), "Setup timed out");
        let (service, _context, collection_id) = result.unwrap();

        timeout(Duration::from_secs(5), async {
            let create_req = CreateVectorRequest {
                collection_id: collection_id.clone(),
                vector: Some(Vector {
                    id: 1,
                    values: vec![0.1, 0.2, 0.3, 0.4],
                }),
            };
            service
                .create_vector(Request::new(create_req))
                .await
                .unwrap();
        })
        .await
        .unwrap();

        let update_result = timeout(Duration::from_secs(5), async {
            let update_req = UpdateVectorRequest {
                collection_id: collection_id.clone(),
                vector_id: 1,
                values: vec![0.5, 0.6, 0.7, 0.8],
            };
            service.update_vector(Request::new(update_req)).await
        })
        .await;

        assert!(update_result.is_ok(), "Update operation timed out");
        let update_response = update_result.unwrap().unwrap();
        assert_eq!(update_response.get_ref().values, vec![0.5, 0.6, 0.7, 0.8]);

        cleanup();
    }

    #[tokio::test]
    async fn test_find_similar_vectors() {
        let result = setup_with_timeout().await;
        assert!(result.is_ok(), "Setup timed out");
        let (service, _context, collection_id) = result.unwrap();

        let vectors = vec![
            (1, vec![0.1, 0.2, 0.3, 0.4]),
            (2, vec![0.2, 0.3, 0.4, 0.5]),
            (3, vec![0.3, 0.4, 0.5, 0.6]),
        ];

        for (id, values) in vectors {
            timeout(Duration::from_secs(5), async {
                let create_req = CreateVectorRequest {
                    collection_id: collection_id.clone(),
                    vector: Some(Vector { id, values }),
                };
                service
                    .create_vector(Request::new(create_req))
                    .await
                    .unwrap();
            })
            .await
            .unwrap();
        }

        let find_result = timeout(Duration::from_secs(5), async {
            let find_req = FindSimilarVectorsRequest {
                vector: vec![0.15, 0.25, 0.35, 0.45],
                k: 2,
            };
            service.find_similar_vectors(Request::new(find_req)).await
        })
        .await;

        assert!(
            find_result.is_ok(),
            "Find similar vectors operation timed out"
        );
        let find_response = find_result.unwrap().unwrap();
        assert!(!find_response.get_ref().results.is_empty());
        assert!(find_response.get_ref().results.len() <= 2);

        cleanup();
    }

    #[tokio::test]
    async fn test_delete_vector() {
        let result = setup_with_timeout().await;
        assert!(result.is_ok(), "Setup timed out");
        let (service, _context, collection_id) = result.unwrap();

        timeout(Duration::from_secs(5), async {
            let create_req = CreateVectorRequest {
                collection_id: collection_id.clone(),
                vector: Some(Vector {
                    id: 1,
                    values: vec![0.1, 0.2, 0.3, 0.4],
                }),
            };
            service
                .create_vector(Request::new(create_req))
                .await
                .unwrap();
        })
        .await
        .unwrap();

        let delete_result = timeout(Duration::from_secs(5), async {
            let delete_req = DeleteVectorRequest {
                collection_id,
                vector_id: 1,
            };
            service.delete_vector(Request::new(delete_req)).await
        })
        .await;

        assert!(
            delete_result.is_err(),
            "Delete operation should fail (Unimplemented)"
        );
        assert_eq!(
            delete_result.unwrap_err().code(),
            tonic::Code::Unimplemented
        );

        cleanup();
    }
}
