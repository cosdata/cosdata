use actix_web::{post, web::Json, HttpResponse, Result};

use super::{dtos::LoginCredentials, service};

#[post("/login")]
pub(crate) async fn login(Json(credentials): Json<LoginCredentials>) -> Result<HttpResponse> {
    let res = service::login(credentials).await?;
    Ok(HttpResponse::Ok().json(res))
}
