use actix_web::{web, Scope};

mod controller;
mod dtos;
mod error;
mod service;

pub(crate) fn auth_module() -> Scope {
    let auth_module = web::scope("/auth").service(controller::login);
    auth_module
}
