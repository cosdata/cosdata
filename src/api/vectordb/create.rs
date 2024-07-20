use actix_web::{web, HttpResponse};

use crate::{
    api_service::init_vector_store,
    models::rpc::{CreateVectorDb, RPCResponseBody},
    web_server::AppEnv,
};

// Route: `/vectordb/createdb`
pub(crate) async fn create(
    env: web::Data<AppEnv>,
    web::Json(body): web::Json<CreateVectorDb>,
) -> HttpResponse {
    // Define the parameters for init_vector_store
    let name = body.vector_db_name;
    let size = body.dimensions as usize;
    let lower_bound = body.min_val;
    let upper_bound = body.max_val;
    // ---------------------------
    // -- TODO Maximum cache level
    // ---------------------------
    let max_cache_level = 5;

    // Call init_vector_store using web::block
    let result =
        init_vector_store(&env, name, size, lower_bound, upper_bound, max_cache_level).await;

    match result {
        Ok(_) => HttpResponse::Ok().json(RPCResponseBody::RespCreateVectorDb { result: true }),
        Err(e) => HttpResponse::NotAcceptable().body(format!("Error: {}", e)),
    }
}
