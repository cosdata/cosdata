use super::{dtos::LoginCredentials, error::AuthError};

pub(crate) async fn login(credentials: LoginCredentials) -> Result<LoginCredentials, AuthError> {
    const USERNAME: &str = "admin";
    const PASSWORD: &str = "admin";

    if credentials.username == USERNAME && credentials.password == PASSWORD {
        Ok(credentials)
    } else {
        Err(AuthError::WrongCredentials)
    }
}
