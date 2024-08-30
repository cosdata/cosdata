use actix_web::{web, Scope};

mod controller;
mod dtos;
mod get_token;
mod service;
mod error;

pub(crate) fn auth_module() -> Scope {
    let auth_module = web::scope("/auth").service(controller::login);
    auth_module
}

pub(crate) use get_token::get_token;
