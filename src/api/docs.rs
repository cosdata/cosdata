use actix_web::{web, HttpResponse, Scope};
use crate::api::openapi::{AuthApiDoc, CollectionsApiDoc, CombinedApiDoc, IndexesApiDoc, SearchApiDoc};
use utoipa::OpenApi;

pub(crate) fn api_docs_module() -> Scope {
    web::scope("/api-docs")
        .route("/openapi.json", web::get().to(openapi_json))
        .route("/auth/openapi.json", web::get().to(auth_openapi_json))
        .route("/collections/openapi.json", web::get().to(collections_openapi_json))
        .route("/indexes/openapi.json", web::get().to(indexes_openapi_json))
        .route("/search/openapi.json", web::get().to(search_openapi_json))
}

async fn openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(CombinedApiDoc::openapi())
}

async fn auth_openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(AuthApiDoc::openapi())
}

async fn collections_openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(CollectionsApiDoc::openapi())
}

async fn indexes_openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(IndexesApiDoc::openapi())
}

async fn search_openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(SearchApiDoc::openapi())
}