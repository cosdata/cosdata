use crate::rbac::guards::require_manage_permissions;
use actix_web::web;

mod controller;
mod dtos;

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/rbac")
            .wrap(require_manage_permissions())
            .service(
                web::scope("/users")
                    .route("", web::get().to(controller::list_users))
                    .route("", web::post().to(controller::create_user))
                    .route("/{username}", web::get().to(controller::get_user))
                    .route("/{username}", web::put().to(controller::update_user))
                    .route("/{username}", web::delete().to(controller::delete_user)),
            )
            .service(
                web::scope("/roles")
                    .route("", web::get().to(controller::list_roles))
                    .route("", web::post().to(controller::create_role))
                    .route("/{role_name}", web::get().to(controller::get_role))
                    .route("/{role_name}", web::put().to(controller::update_role))
                    .route("/{role_name}", web::delete().to(controller::delete_role)),
            )
            .service(
                web::scope("/collections")
                    .route("", web::get().to(controller::list_rbac_collections))
                    .route("", web::post().to(controller::create_rbac_collection))
                    .route(
                        "/{collection_id_or_name}",
                        web::delete().to(controller::delete_rbac_collection),
                    )
                    .route(
                        "/{collection_id_or_name}/users/{username}/roles/{role_name}",
                        web::put().to(controller::assign_role),
                    )
                    .route(
                        "/{collection_id_or_name}/users/{username}/roles",
                        web::delete().to(controller::remove_user_roles),
                    ),
            ),
    );
}
