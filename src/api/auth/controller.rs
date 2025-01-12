use actix_web::{
    web::{self},
    HttpResponse, Result,
};

use crate::app_context::AppContext;

use super::{dtos::CreateSessionDTO, service};

pub(crate) async fn create_session(
    web::Json(create_session_dto): web::Json<CreateSessionDTO>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let res = service::create_session(create_session_dto, ctx.into_inner()).await?;
    Ok(HttpResponse::Ok().json(res))
}
