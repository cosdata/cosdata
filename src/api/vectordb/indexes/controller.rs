use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;

use super::{dtos::CreateIndexDto, service};

pub(crate) async fn create_index(
    web::Json(create_index_dto): web::Json<CreateIndexDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    service::create_index(create_index_dto, ctx.into_inner()).await?;
    Ok(HttpResponse::Ok().json(serde_json::json!({})))
}
