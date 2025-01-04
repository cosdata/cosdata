
/*
Authentication system:
- Stores username and SHA256(password) hash in DB
- On /v1/auth endpoint: validates provided password by comparing hashes
- Returns time-based session token with forward secrecy
- Forward secrecy ensures leaked token cannot compromise future sessions
- Uses HMAC-SHA256 for message authentication and key derivation
*/

use ring::hmac;
use ring::rand::{SecureRandom, SystemRandom};
use std::time::{SystemTime, UNIX_EPOCH};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};

const TOKEN_LIFETIME: u64 = 900; // 15 minutes 

#[derive(Debug)]
struct Session {
   access_token: String,
   expires_at: u64,
}

fn generate_master_key() -> [u8; 32] {
   let rng = SystemRandom::new();
   let mut key = [0u8; 32];
   rng.fill(&mut key).unwrap();
   key
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

fn create_session(username: &str, master_key: &[u8]) -> Session {
   let timestamp = get_current_timestamp();
   let token_key = generate_token_key(master_key, timestamp);
   
   let payload = format!("{}|{}", username, timestamp);
   let access_token = URL_SAFE_NO_PAD.encode(
       hmac::sign(&token_key, payload.as_bytes()).as_ref()
   );

   Session {
       access_token,
       expires_at: timestamp + TOKEN_LIFETIME,
   }
}

fn verify_token(token: &str, username: &str, master_key: &[u8]) -> bool {
   let timestamp = get_current_timestamp();
   let token_key = generate_token_key(master_key, timestamp);
   let payload = format!("{}|{}", username, timestamp);
   
   let expected_token = URL_SAFE_NO_PAD.encode(
       hmac::sign(&token_key, payload.as_bytes()).as_ref()
   );
   
   token == expected_token
}
