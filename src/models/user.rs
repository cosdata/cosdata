use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AuthResp {
    pub session_key: Option<String>,
    pub calls_used: i32,
    pub calls_remaining: i32,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AddUserResp {
    pub aur_user: User,
    pub aur_password: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct User {
    pub u_username: String,
    pub u_hashed_password: String,
    pub u_first_name: String,
    pub u_last_name: String,
    pub u_email: String,
    pub u_roles: Vec<String>,
    pub u_api_quota: i32,
    pub u_api_used: i32,
    pub u_api_expiry_time: String,
    pub u_session_key: String,
    pub u_session_key_expiry: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Statistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min_val: i32,
    pub max_val: i32,
    pub count: i32,
}

pub fn login(user: &str, pass: &str) -> AuthResp {
    // Placeholder for login function
    AuthResp {
        session_key: None,
        calls_used: 0,
        calls_remaining: 0,
    }
}

pub fn lookup_user_data(session_key: &str) -> Option<(String, i32, i32, String, Vec<String>)> {
    // Placeholder for looking up user data
    None
}

pub fn update_user_data(
    session_key: &str,
    name: String,
    quota: i32,
    used: i32,
    exp: String,
    roles: &Vec<String>,
) {
    // Placeholder for updating user data
}

pub fn delete_user_data(session_key: &str) {
    // Placeholder for deleting user data
}
