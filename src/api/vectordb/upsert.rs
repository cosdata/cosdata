use std::sync::atomic::Ordering;

use actix_web::{web, HttpResponse};

use crate::{
    api_service::run_upload_dense_vectors,
    app_context::AppContext,
    indexes::hnsw::types::DenseInputVector,
    models::rpc::{RPCResponseBody, UpsertVectors},
};

// Route: `/vectordb/upsert`
pub(crate) async fn upsert(
    web::Json(body): web::Json<UpsertVectors>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
    // Try to get the vector store from the environment
    let hnsw_index = match ctx
        .ain_env
        .collections_map
        .get_hnsw_index(&body.vector_db_name)
    {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    }
    .clone();

    if !hnsw_index
        .current_open_transaction
        .load(Ordering::SeqCst)
        .is_null()
    {
        return HttpResponse::Conflict()
            .body("Cannot upsert while there's an on-going transaction");
    }

    // Call run_upload with the extracted parameters
    let res = web::block(move || {
        run_upload_dense_vectors(
            ctx.into_inner(),
            hnsw_index,
            body.vectors
                .into_iter()
                // @TODO(vineet): Add support for optional metadata fields
                .map(|vec| DenseInputVector::new(vec.id, vec.values, None))
                .collect(),
        )
    })
    .await
    .unwrap();

    match res {
        Ok(_) => HttpResponse::Ok().json(RPCResponseBody::RespUpsertVectors { insert_stats: None }),
        Err(err) => {
            HttpResponse::InternalServerError().body(format!("error upserting vectors: {}", err))
        }
    }
}
