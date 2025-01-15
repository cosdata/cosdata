use actix_web::{web, Scope};
pub(crate) mod authentication_middleware;
mod controller;
mod dtos;
mod error;
mod service;

pub(crate) fn auth_module() -> Scope {
    let auth_module = web::scope("/auth").route(
        "/create-session",
        web::post().to(controller::create_session),
    );

    auth_module
}
