use std::{path::Path, sync::Arc};

use actix_web::{web, HttpResponse};

use crate::{
    api_service::ann_vector_query,
    convert_option_vec,
    models::{
        buffered_io::BufferManagerFactory,
        cache_loader::NodeRegistry,
        rpc::{RPCResponseBody, VectorANN},
        types::get_app_env,
    },
};

// Route: `/vectordb/search`
pub(crate) async fn search(web::Json(body): web::Json<VectorANN>) -> HttpResponse {
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => return HttpResponse::InternalServerError().body("Env initialization error"),
    };
    // Try to get the vector store from the environment
    let vec_store = match env.vector_store_map.get(&body.vector_db_name) {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    };

    let result = match ann_vector_query(
        vec_store.clone(),
        // TODO: use global cache
        Arc::new(NodeRegistry::new(
            1000,
            Arc::new(BufferManagerFactory::new(
                Path::new(".").into(),
                |root, ver| root.join(format!("{}.index", **ver)),
            )),
        )),
        body.vector,
    )
    .await
    {
        Ok(result) => result,
        Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
    };

    let response_data = RPCResponseBody::RespVectorKNN {
        knn: convert_option_vec(result),
    };
    HttpResponse::Ok().json(response_data)
}
