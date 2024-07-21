use actix_web::{web, HttpResponse};

use crate::{
    api_service::ann_vector_query,
    convert_option_vec,
    models::rpc::{RPCResponseBody, VectorANN},
    web_server::AppEnv,
};

// Route: `/vectordb/search`
pub(crate) async fn search(
    env: web::Data<AppEnv>,
    web::Json(body): web::Json<VectorANN>,
) -> HttpResponse {
    // Try to get the vector store from the environment
    let vec_store = match env.vector_store_map.get(&body.vector_db_name) {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    };

    let result = match ann_vector_query(vec_store.clone(), body.vector).await {
        Ok(result) => result,
        Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
    };

    let response_data = RPCResponseBody::RespVectorKNN {
        knn: convert_option_vec(result),
    };
    HttpResponse::Ok().json(response_data)
}
