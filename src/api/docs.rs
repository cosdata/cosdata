use crate::api::openapi::{
    AuthApiDoc, CollectionsApiDoc, CombinedApiDoc, IndexesApiDoc, SearchApiDoc, StreamingApiDoc,
    TransactionsApiDoc, VectorsApiDoc, VersionsApiDoc,
};
use actix_web::{web, HttpResponse, Scope};
use utoipa::OpenApi;

pub(crate) fn api_docs_module() -> Scope {
    web::scope("/api-docs")
        .route("/openapi.json", web::get().to(openapi_json))
        .route("/auth/openapi.json", web::get().to(auth_openapi_json))
        .route(
            "/collections/openapi.json",
            web::get().to(collections_openapi_json),
        )
        .route("/indexes/openapi.json", web::get().to(indexes_openapi_json))
        .route("/search/openapi.json", web::get().to(search_openapi_json))
        .route(
            "/transactions/openapi.json",
            web::get().to(transactions_openapi_json),
        )
        .route("/vectors/openapi.json", web::get().to(vectors_openapi_json))
        .route(
            "/versions/openapi.json",
            web::get().to(versions_openapi_json),
        )
        .route(
            "/streaming/openapi.json",
            web::get().to(streaming_openapi_json),
        )
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

async fn transactions_openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(TransactionsApiDoc::openapi())
}

async fn vectors_openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(VectorsApiDoc::openapi())
}

async fn versions_openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(VersionsApiDoc::openapi())
}

async fn streaming_openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(StreamingApiDoc::openapi())
}
