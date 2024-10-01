use crate::config_loader::Config;

pub struct AppContext {
    pub config: Config,
}

impl AppContext {

    pub fn new(config: Config) -> Self {
        Self {
            config,
        }
    }
}

