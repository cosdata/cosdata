use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct LoginCredentials {
    pub username: String,
    pub password: String,
    pub pretty_print: bool,
}
