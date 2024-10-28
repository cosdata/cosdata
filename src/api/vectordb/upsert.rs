use actix_web::{web, HttpResponse};

use crate::{
    api_service::run_upload,
    app_context::AppContext,
    convert_vectors,
    models::rpc::{RPCResponseBody, UpsertVectors},
};

// Route: `/vectordb/upsert`
pub(crate) async fn upsert(
    web::Json(body): web::Json<UpsertVectors>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
    // Try to get the vector store from the environment
    let collection = match ctx.ain_env.collections_map.get(&body.vector_db_name) {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    }
    .clone();

    if collection.current_open_transaction.clone().get().is_some() {
        return HttpResponse::Conflict()
            .body("Cannot upsert while there's an on-going transaction");
    }

    if !collection.get_auto_config_flag() && !collection.get_configured_flag() {
        return HttpResponse::BadRequest()
            .body("Vector store is set to manual indexing but an index is not created");
    }

    // Call run_upload with the extracted parameters
    let res = web::block(move || {
        run_upload(ctx.into_inner(), collection, convert_vectors(body.vectors))
    })
    .await
    .unwrap();

    match res {
        Ok(_) => HttpResponse::Ok().json(RPCResponseBody::RespUpsertVectors { insert_stats: None }),
        Err(err) => HttpResponse::InternalServerError().body(format!("error upserting vectors: {}", err)),
    }
}
