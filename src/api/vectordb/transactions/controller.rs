use actix_web::{web, HttpResponse};

use super::{error::TransactionError, service};

pub(crate) async fn create_transaction(
    collection_id: web::Path<String>,
) -> Result<HttpResponse, TransactionError> {
    let collection_id = collection_id.into_inner();
    let transaction = service::create_transaction(&collection_id).await?;
    Ok(HttpResponse::Ok().json(transaction))
}

pub(crate) async fn commit_transaction(
    params: web::Path<(String, String)>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();
    let _ = service::commit_transaction(&collection_id, &transaction_id).await?;
    Ok(HttpResponse::NoContent().finish())
}
