use actix_web::{web, HttpResponse};

use crate::{
    api_service::{ann_vector_query, batch_ann_vector_query},
    app_context::AppContext,
    models::rpc::{BatchVectorANN, RPCResponseBody, VectorANN},
};

// Route: `/vectordb/search`
pub(crate) async fn search(
    web::Json(body): web::Json<VectorANN>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
    // Try to get the vector store from the environment
    let vec_store = match ctx.ain_env.collections_map.get(&body.vector_db_name) {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    };

    let result = match ann_vector_query(ctx.into_inner(), vec_store.clone(), body.vector).await {
        Ok(result) => result,
        Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
    };

    let response_data = RPCResponseBody::RespVectorKNN {
        knn: result.into_iter().map(|(id, dist)| (id.0, dist)).collect(),
    };
    HttpResponse::Ok().json(response_data)
}

// Route: `/vectordb/batch-search`
pub(crate) async fn batch_search(
    web::Json(body): web::Json<BatchVectorANN>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
    // Try to get the vector store from the environment
    let vec_store = match ctx.ain_env.collections_map.get(&body.vector_db_name) {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    };

    let results =
        match batch_ann_vector_query(ctx.into_inner(), vec_store.clone(), body.vectors).await {
            Ok(result) => result,
            Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
        };

    let response_data: Vec<_> = results
        .into_iter()
        .map(|result| RPCResponseBody::RespVectorKNN {
            knn: result.into_iter().map(|(id, dist)| (id.0, dist)).collect(),
        })
        .collect();
    HttpResponse::Ok().json(response_data)
}
