use actix_web::HttpResponse;

pub(crate) async fn create_transaction() -> HttpResponse {
    HttpResponse::Ok().body("Hello")
}