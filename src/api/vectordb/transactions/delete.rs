use crate::app_context::AppContext;
use actix_web::{web, HttpResponse};

// Route: `/vectordb/{database_name}/transactions/{transaction_id}/delete`
pub(crate) async fn delete(
    path_data: web::Path<(String, String)>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
    let (database_name, _transaction_id) = path_data.into_inner();
    let Some(_vec_store) = ctx.ain_env.collections_map.get(&database_name) else {
        return HttpResponse::NotFound().body("Vector store not found");
    };

    todo!()
}
