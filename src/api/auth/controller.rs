use actix_web::{web::Json, HttpMessage, HttpRequest, HttpResponse, Result};

use super::{
    dtos::{Claims, LoginCredentials},
    error::AuthError,
    service,
};

pub(crate) async fn login(Json(credentials): Json<LoginCredentials>) -> Result<HttpResponse> {
    let res = service::login(credentials).await?;
    Ok(HttpResponse::Ok().body(res))
}

pub(crate) async fn protected_route(req: HttpRequest) -> Result<HttpResponse> {
    let claims = req.extensions();
    let claims = claims
        .get::<Claims>()
        .ok_or(AuthError::FailedToExtractTokenFromRequest)?;
    Ok(HttpResponse::Ok().json(claims))
}
