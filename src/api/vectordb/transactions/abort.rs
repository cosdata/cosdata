use crate::app_context::AppContext;
use actix_web::{web, HttpResponse};

// Route: `/vectordb/{database_name}/transactions/{transaction_id}/abort`
pub(crate) async fn abort(
    path_data: web::Path<(String, u32)>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
    let (database_name, transaction_id) = path_data.into_inner();
    let Some(vec_store) = ctx.ain_env.vector_store_map.get(&database_name) else {
        return HttpResponse::NotFound().body("Vector store not found");
    };

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();
    let current_open_transaction = current_open_transaction_arc.get();
    let Some(current_transaction_id) = current_open_transaction else {
        return HttpResponse::NotFound().body("Transaction not found");
    };

    if **current_transaction_id != transaction_id {
        return HttpResponse::NotFound().body("Transaction not found");
    }

    current_open_transaction_arc.update(None);

    HttpResponse::Ok().finish()
}
