use actix_web::{web, Scope};

mod controller;
pub(crate) mod dtos;
mod error;
mod service;

pub(crate) fn version_module() -> Scope {
    let version_module = web::scope("/versions");
	version_module
}
