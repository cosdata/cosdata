use std::sync::Arc;

use crate::{
    app_context::AppContext,
    models::{
        crypto::{self, SingleSHA256Hash},
        types::SessionDetails,
    },
};

use super::{
    dtos::{CreateSessionDTO, Session},
    error::AuthError,
};

const TOKEN_LIFETIME: u64 = 900; // 15 minutes

pub(crate) async fn create_session(
    create_session_dto: CreateSessionDTO,
    ctx: Arc<AppContext>,
) -> Result<Session, AuthError> {
    let user = ctx
        .ain_env
        .users_map
        .get_user(&create_session_dto.username)
        .ok_or_else(|| AuthError::WrongCredentials)?;
    let password_hash = SingleSHA256Hash::from_str(&create_session_dto.password);
    let password_double_hash = password_hash.hash_again();
    // check if passwords match, in constant time to prevents timing attacks
    if !password_double_hash.verify_eq(&user.password_hash) {
        return Err(AuthError::WrongCredentials)?;
    }

    let (access_token, timestamp) = crypto::create_session(
        &create_session_dto.username,
        &ctx.ain_env.server_key,
        &password_hash,
    );

    let created_at = timestamp;
    let expires_at = timestamp + TOKEN_LIFETIME;

    ctx.ain_env.active_sessions.insert(
        access_token.clone(),
        SessionDetails {
            created_at,
            expires_at,
            user,
        },
    );

    Ok(Session {
        access_token,
        created_at,
        expires_at,
    })
}
