use crate::models::rpc::Authenticate;
use actix_web::{web, HttpResponse};

// Route: `/auth/gettoken`
pub(crate) async fn get_token(web::Json(body): web::Json<Authenticate>) -> HttpResponse {
    HttpResponse::Ok().json(body)
}
