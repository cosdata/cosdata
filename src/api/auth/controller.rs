use actix_web::{
    web::{self},
    HttpResponse, Result,
};

use crate::app_context::AppContext;

use super::{
    dtos::{CreateSessionDTO, Session},
    service,
};

/// Create a new user session (login)
#[utoipa::path(
    post,
    path = "/auth/create-session",
    request_body = CreateSessionDTO,
    responses(
        (status = 200, description = "Session created successfully", body = Session),
        (status = 400, description = "Wrong credentials"),
        (status = 401, description = "Invalid authentication token"),
        (status = 500, description = "Internal server error")
    ),
    tag = "auth"
)]
pub(crate) async fn create_session(
    web::Json(create_session_dto): web::Json<CreateSessionDTO>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let res = service::create_session(create_session_dto, ctx.into_inner()).await?;
    Ok(HttpResponse::Ok().json(res))
}
