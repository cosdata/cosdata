use actix_web::{web, HttpResponse};

use crate::{
    api_service::ann_vector_query,
    app_context::AppContext,
    convert_option_vec,
    models::{
        rpc::{RPCResponseBody, VectorANN},
        types::get_app_env,
    },
};

// Route: `/vectordb/search`
pub(crate) async fn search(
    web::Json(body): web::Json<VectorANN>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
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

    let result =
        match ann_vector_query(ctx.node_registry.clone(), vec_store.clone(), body.vector).await {
            Ok(result) => result,
            Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
        };

    let response_data = RPCResponseBody::RespVectorKNN {
        knn: convert_option_vec(result),
    };
    HttpResponse::Ok().json(response_data)
}
