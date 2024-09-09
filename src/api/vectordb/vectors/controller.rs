use crate::config_loader::Config;
use actix_web::{web, HttpResponse, Result};

use super::{dtos::CreateVectorDto, service};

pub(crate) async fn create_vector(
    collection_id: web::Path<String>,
    web::Json(create_vector_dto): web::Json<CreateVectorDto>,
    config: web::Data<Config>,
) -> Result<HttpResponse> {
    let vector =
        service::create_vector(&collection_id, create_vector_dto, config.into_inner()).await?;
    Ok(HttpResponse::Ok().json(vector))
}

pub(crate) async fn get_vector_by_id(path: web::Path<(String, String)>) -> Result<HttpResponse> {
    let (collection_id, vector_id) = path.into_inner();
    let vector = service::get_vector_by_id(&collection_id, &vector_id).await?;
    Ok(HttpResponse::Ok().json(vector))
}
