use actix_web::{web, HttpResponse};

use crate::web_server::AppEnv;

// Route: `/vectordb/{database_name}/transactions/{transaction_id}/delete`
pub(crate) async fn delete(
    env: web::Data<AppEnv>,
    path_data: web::Path<(String, String)>,
) -> HttpResponse {
    let (database_name, transaction_id) = path_data.into_inner();
    let Some(vec_store) = env.vector_store_map.get(&database_name) else {
        return HttpResponse::NotFound().body("Vector store not found");
    };

    todo!()
}
