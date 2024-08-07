use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
pub struct Config {
   pub server: Server,
}

#[derive(Deserialize)]
pub struct Server {
   pub host: String,
   pub port: String
}

pub fn load_config() -> Config {
    let config_contents = fs::read_to_string("config.toml").expect("Failed to config file");
    let config: Config = toml::from_str(&config_contents).expect("Failed to parse config file contents!");
    
    
    config
}

