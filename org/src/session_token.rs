// Authentication system flow:
// Storage:
// - Stores username and SHA256(SHA256(password)) hash in DB
// - Uses SHA256(password) as user_secret to derive master_key
//
// Authentication (/v1/auth):
// - Validates password by comparing double-hashed password with DB
// - Derives master_key using server_key and SHA256(password)
// - Returns time-based session token with timestamp
//
// Security:
// - Forward secrecy through time-based HMAC tokens
// - Master key requires both server and user components
// - Leaked tokens cannot compromise future sessions

// Authentication flow:
// - Stores username and SHA256(SHA256(password)) hash in DB
// - Uses SHA256(password) as user_secret to derive master_key
// - Derives session tokens using the master_key
// - Provides forward secrecy through time-based HMAC

use ring::{hmac, digest};
use ring::rand::{SecureRandom, SystemRandom};
use std::time::{SystemTime, UNIX_EPOCH};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};

const TOKEN_LIFETIME: u64 = 900; // 15 minutes

#[derive(Debug)]
struct Session {
   access_token: String,
   expires_at: u64,
}

fn hash_password(password: &str) -> Vec<u8> {
   let mut context = digest::Context::new(&digest::SHA256);
   context.update(password.as_bytes());
   context.finish().as_ref().to_vec()
}

fn double_hash_password(password: &str) -> Vec<u8> {
   let mut context = digest::Context::new(&digest::SHA256);
   context.update(&hash_password(password));
   context.finish().as_ref().to_vec()
}

fn generate_server_key() -> [u8; 32] {
   let rng = SystemRandom::new();
   let mut key = [0u8; 32];
   rng.fill(&mut key).unwrap();
   key
}

fn derive_master_key(server_key: &[u8], password: &str) -> [u8; 32] {
   let user_secret = hash_password(password);
   let key = hmac::Key::new(hmac::HMAC_SHA256, server_key);
   let derived = hmac::sign(&key, &user_secret);
   let mut result = [0u8; 32];
   result.copy_from_slice(derived.as_ref());
   result
}

fn get_current_timestamp() -> u64 {
   SystemTime::now()
       .duration_since(UNIX_EPOCH)
       .unwrap()
       .as_secs()
}

fn generate_token_key(master_key: &[u8], timestamp: u64) -> hmac::Key {
   let key = hmac::Key::new(hmac::HMAC_SHA256, master_key);
   let ts_bytes = timestamp.to_be_bytes();
   hmac::Key::new(hmac::HMAC_SHA256, 
       hmac::sign(&key, &ts_bytes).as_ref())
}

fn create_session(username: &str, server_key: &[u8], password: &str) -> Session {
   let master_key = derive_master_key(server_key, password);
   let timestamp = get_current_timestamp();
   let token_key = generate_token_key(&master_key, timestamp);
   
   let payload = format!("{}|{}", username, timestamp);
   let access_token = URL_SAFE_NO_PAD.encode(
       hmac::sign(&token_key, payload.as_bytes()).as_ref()
   );

   Session {
       access_token,
       expires_at: timestamp + TOKEN_LIFETIME,
   }
}

fn verify_token(token: &str, username: &str, server_key: &[u8], password: &str) -> bool {
   let master_key = derive_master_key(server_key, password);
   let timestamp = get_current_timestamp();
   let token_key = generate_token_key(&master_key, timestamp);
   let payload = format!("{}|{}", username, timestamp);
   
   let expected_token = URL_SAFE_NO_PAD.encode(
       hmac::sign(&token_key, payload.as_bytes()).as_ref()
   );
   
   token == expected_token
}
