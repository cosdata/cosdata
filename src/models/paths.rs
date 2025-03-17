use std::path::PathBuf;

pub fn get_data_path() -> PathBuf {
    use std::env;

    // Check if COSDATA_HOME is set (installed mode)
    if let Ok(home) = env::var("COSDATA_HOME") {
        return PathBuf::from(home).join("data");
    }

    // Check if running inside a Cargo-built target folder
    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            if parent.ends_with("debug") || parent.ends_with("release") {
                return PathBuf::from("./data"); // Local development mode (within repo)
            }
        }
    }

    // Default fallback (shouldn't happen often)
    PathBuf::from(env::var("HOME").unwrap()).join("cosdata/data")
}

pub fn get_config_path() -> PathBuf {
    use std::env;

    // Check if COSDATA_HOME is set (installed mode)
    if let Ok(home) = env::var("COSDATA_HOME") {
        return PathBuf::from(home).join("config/config.toml");
    }

    // Check if running inside a Cargo-built target folder
    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            if parent.ends_with("debug") || parent.ends_with("release") {
                return PathBuf::from("./config.toml"); // Local development mode
            }
        }
    }

    // Default fallback (shouldn't happen often)
    PathBuf::from(env::var("HOME").unwrap()).join("cosdata/config/config.toml")
}
