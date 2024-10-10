use actix_web::{web, HttpResponse};

use crate::{api::vectordb::vectors::dtos::CreateVectorDto, app_context::AppContext};

use super::{error::TransactionError, service};

pub(crate) async fn create_transaction(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let collection_id = collection_id.into_inner();
    let transaction = service::create_transaction(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(transaction))
}

pub(crate) async fn commit_transaction(
    params: web::Path<(String, u32)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();
    let _ = service::commit_transaction(ctx.into_inner(), &collection_id, transaction_id.into()).await?;
    Ok(HttpResponse::NoContent().finish())
}

pub(crate) async fn create_vector_in_transaction(
    params: web::Path<(String, u32)>,
    web::Json(create_vector_dto): web::Json<CreateVectorDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();
    let vector = service::create_vector_in_transaction(
        ctx.into_inner(),
        &collection_id,
        transaction_id.into(),
        create_vector_dto,
    )
    .await?;
    Ok(HttpResponse::Ok().json(vector))
}
