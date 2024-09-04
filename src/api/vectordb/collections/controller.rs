use actix_web::{
    web::{self},
    HttpResponse, Result,
};

use super::{dtos::FindCollectionDto, service};

pub(crate) async fn get_collection_by_id(collection_id: web::Path<String>) -> Result<HttpResponse> {
    let collection = service::get_collection_by_id(&collection_id)?;
    Ok(HttpResponse::Ok().json(FindCollectionDto {
        id: collection.database_name.clone(),
        dimensions: collection.quant_dim,
        vector_db_name: collection.database_name.clone(),
    }))
}
