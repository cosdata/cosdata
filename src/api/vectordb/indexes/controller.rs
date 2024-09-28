use actix_web::{web, HttpResponse, Result};

use super::{dtos::CreateIndexDto, service};

pub(crate) async fn create_index(
    web::Json(create_index_dto): web::Json<CreateIndexDto>,
) -> Result<HttpResponse> {
    service::create_index(create_index_dto).await?;
    Ok(HttpResponse::Ok().json(serde_json::json!({})))
}
