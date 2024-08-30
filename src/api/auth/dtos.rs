use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct LoginCredentials {
    pub username: String,
    pub password: String,
    pub pretty_print: bool,
}

#[derive(Serialize, Deserialize, Debug)]
// a structure for holding claims data used in JWT tokens
// resembles payload in NodeJS world
pub struct Claims {
    pub exp: i64,         // Expiry time of the token
    pub iat: i64,         // Issued at time of the token
    pub username: String, // Email associated with the token
}
