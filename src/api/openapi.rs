use utoipa::OpenApi;

/// API documentation for authentication endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::auth::controller::create_session
    ),
    components(
        schemas(
            crate::api::auth::dtos::CreateSessionDTO,
            crate::api::auth::dtos::Session,
            crate::api::auth::dtos::Claims
        )
    ),
    tags(
        (name = "auth", description = "Authentication endpoints")
    ),
    info(
        title = "Cosdata API",
        version = env!("CARGO_PKG_VERSION"),
        description = "Cosdata Vector Database API",
        license(
            name = "Apache 2.0",
            url = "https://www.apache.org/licenses/LICENSE-2.0"
        )
    )
)]
pub struct AuthApiDoc;