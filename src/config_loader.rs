use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize, Clone)]
pub struct Config {
    pub server: Server,
    pub upload_threshold: u32,
    pub upload_process_batch_size: usize
}

#[derive(Deserialize, Clone)]
pub struct Ssl {
    pub cert_file: PathBuf,
    pub key_file: PathBuf,
}

#[derive(Deserialize, Clone)]
pub struct Server {
    pub host: String,
    pub port: u16,
    pub ssl: Ssl,
}

pub fn load_config() -> Config {
    let config_contents = fs::read_to_string("config.toml").expect("Failed to load config file");
    let config: Config = toml::from_str(&config_contents).expect("Failed to parse config file contents!");
    config
}

