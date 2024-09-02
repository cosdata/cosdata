use actix_web::{web::Json, HttpResponse, Result};

use super::{
    dtos::{Claims, LoginCredentials},
    service,
};

pub(crate) async fn login(Json(credentials): Json<LoginCredentials>) -> Result<HttpResponse> {
    let res = service::login(credentials).await?;
    Ok(HttpResponse::Ok().body(res))
}

pub(crate) async fn protected_route(claims: Claims) -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(claims))
}
