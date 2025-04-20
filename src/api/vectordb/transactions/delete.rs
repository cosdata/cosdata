use crate::app_context::AppContext;
use actix_web::{web, HttpResponse};

// Route: `/vectordb/{database_name}/transactions/{transaction_id}/delete`
pub(crate) async fn delete(
    path_data: web::Path<(String, String)>,
    _ctx: web::Data<AppContext>,
) -> HttpResponse {
    let (_database_name, _transaction_id) = path_data.into_inner();

    todo!()
}
