use actix_web::{web, HttpResponse};

use crate::web_server::AppEnv;

// Route: `/vectordb/{database_name}/transactions`
pub(crate) async fn create(
    env: web::Data<AppEnv>,
    database_name: web::Path<String>,
) -> HttpResponse {
    let Some(vec_store) = env.vector_store_map.get(&database_name.into_inner()) else {
        return HttpResponse::NotFound().body("Vector store not found");
    };

    vec_store.levels_prob;

    todo!()
}
