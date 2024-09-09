use actix_web::{web, HttpResponse};

use super::{dtos::CreateIndexDto, service};

pub(crate) async fn create_index(
    web::Json(create_index_dto): web::Json<CreateIndexDto>,
) -> HttpResponse {
    let index = service::create_index(create_index_dto).await;
    HttpResponse::Ok().json(index)
}
