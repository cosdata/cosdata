use actix_web::{web, HttpResponse, Scope};
use dtos::CreateIndexDto;

mod dtos;

async fn index(body: web::Json<CreateIndexDto>) -> HttpResponse {
    HttpResponse::Ok().json(body)
}

pub(crate) fn indexes_module() -> Scope {
    let indexes_module = web::scope("/indexes").route("", web::post().to(index));

    indexes_module
}
