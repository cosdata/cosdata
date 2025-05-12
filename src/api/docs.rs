use actix_web::{web, HttpResponse, Scope};
use crate::api::openapi::AuthApiDoc;
use utoipa::OpenApi;

pub(crate) fn api_docs_module() -> Scope {
    web::scope("/api-docs")
        .route("/openapi.json", web::get().to(openapi_json))
}

async fn openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(AuthApiDoc::openapi())
}