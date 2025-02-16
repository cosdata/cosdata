// Cryptographic utility functions and data types

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ring::{constant_time::verify_slices_are_equal, digest, hmac};
use std::time::{SystemTime, UNIX_EPOCH};

// Single SHA256 hash
// Flow: Input -> |SHA256| -> 32-byte hash
#[derive(Clone)]
pub struct SingleSHA256Hash(pub [u8; 32]);

// Double SHA256 hash
// Flow: Input -> |SHA256| -> Hash1 -> |SHA256| -> Final 32-byte hash
#[derive(Clone)]
pub struct DoubleSHA256Hash(pub [u8; 32]);

// Master key derived from user password and server key
pub struct MasterKey(pub [u8; 32]);

impl SingleSHA256Hash {
    // Computes single-pass SHA256 hash of input data
    pub fn new(input: &[u8]) -> Self {
        // Perform SHA256 hashing of input data
        let hash = digest::digest(&digest::SHA256, input);

        // Convert dynamic-sized digest to fixed-size array
        let mut result = [0u8; 32];
        result.copy_from_slice(hash.as_ref());
        Self(result)
    }

    pub fn from_str(str: &str) -> Self {
        Self::new(str.as_bytes())
    }

    // Computes double-pass SHA256 hash by hashing this single-pass SHA256 again
    pub fn hash_again(&self) -> DoubleSHA256Hash {
        // Second pass: hash the first hash result
        let double_hash = digest::digest(&digest::SHA256, &self.0);

        // Convert to fixed-size array
        let mut result = [0u8; 32];
        result.copy_from_slice(double_hash.as_ref());
        DoubleSHA256Hash(result)
    }

    pub fn verify_eq(&self, other: &Self) -> bool {
        verify_slices_are_equal(&self.0, &other.0).is_ok()
    }
}

impl DoubleSHA256Hash {
    // Computes double-pass SHA256 hash
    pub fn new(input: &[u8]) -> Self {
        // First pass: hash the input
        let hash = digest::digest(&digest::SHA256, input);

        // Second pass: hash the first hash result
        let double_hash = digest::digest(&digest::SHA256, hash.as_ref());

        // Convert to fixed-size array
        let mut result = [0u8; 32];
        result.copy_from_slice(double_hash.as_ref());
        DoubleSHA256Hash(result)
    }

    pub fn from_str(str: &str) -> Self {
        Self::new(str.as_bytes())
    }

    pub fn verify_eq(&self, other: &Self) -> bool {
        verify_slices_are_equal(&self.0, &other.0).is_ok()
    }
}

impl MasterKey {
    // Derives master key from server key and user password
    // Flow:
    //
    //   User Key -> |HMAC-SHA256| -> Master Key
    //                      ^
    //                      |
    //                as signing key
    //                      |
    //                  Server Key
    pub fn new(server_key: &SingleSHA256Hash, user_key: &SingleSHA256Hash) -> Self {
        // Initialize HMAC with server key
        let key = hmac::Key::new(hmac::HMAC_SHA256, &server_key.0);

        // Sign user key with server key via HMAC
        let derived = hmac::sign(&key, &user_key.0);

        // Convert to fixed-size array
        let mut result = [0u8; 32];
        result.copy_from_slice(derived.as_ref());
        Self(result)
    }
}

// Generates time-based token key from master key
pub fn generate_token_key(master_key: &MasterKey, timestamp: u64) -> hmac::Key {
    // Create HMAC key from master key
    let key = hmac::Key::new(hmac::HMAC_SHA256, &master_key.0);

    // Convert timestamp to bytes and sign with master key
    let ts_bytes = timestamp.to_be_bytes();
    hmac::Key::new(hmac::HMAC_SHA256, hmac::sign(&key, &ts_bytes).as_ref())
}

// Retrieves current Unix timestamp for token generation
pub fn get_current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// Creates a new session with time-based access token
// Flow:
// 1. Derive master key from server key and user key
// 2. Generate token key using current timestamp
// 3. Create and sign payload (username + timestamp)
// 4. Encode final token in URL-safe base64
pub fn create_session(
    username: &str,
    server_key: &SingleSHA256Hash,
    user_key: &SingleSHA256Hash,
) -> (String, u64) {
    // Generate master key from user credentials
    let master_key = MasterKey::new(server_key, user_key);

    // Get current timestamp for token
    let timestamp = get_current_timestamp();

    // Generate time-based token key
    let token_key = generate_token_key(&master_key, timestamp);

    // Create payload with username and timestamp
    let payload = format!("{}|{}", username, timestamp);

    // Sign and encode final access token
    let access_token = URL_SAFE_NO_PAD.encode(hmac::sign(&token_key, payload.as_bytes()));

    (access_token, timestamp)
}

#[cfg(test)]
mod tests {
    use ring::rand::{SecureRandom, SystemRandom};

    use super::*;

    fn random_bytes() -> [u8; 32] {
        let rng = SystemRandom::new();
        let mut key = [0u8; 32];
        rng.fill(&mut key).unwrap();
        key
    }

    #[test]
    fn test_single_step_and_double_step_hashing_consistency() {
        let input = random_bytes();
        let single_hash = SingleSHA256Hash::new(&input);
        let two_step_double_hash = single_hash.hash_again();
        let single_step_double_hash = DoubleSHA256Hash::new(&input);
        assert_eq!(two_step_double_hash.0, single_step_double_hash.0);
    }
}
