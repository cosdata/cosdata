use super::models::common::WaCustomError;
use super::models::paths::get_config_path;
use serde::{Deserialize, Deserializer};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::{io, vec};

#[derive(Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub thread_pool: ThreadPool,
    pub server: Server,
    pub hnsw: Hnsw,
    pub indexing: Indexing,
    pub search: Search,
    pub upload_threshold: u32,
    pub upload_process_batch_size: usize,
    pub num_regions_to_load_on_restart: usize,
    pub inverted_index_data_file_parts: u8,
    pub sparse_raw_values_reranking_factor: usize,
    pub rerank_sparse_with_raw_values: bool,
    #[serde(default)]
    pub cache: CacheConfig,
}

#[derive(Deserialize, Clone)]
pub struct Ssl {
    pub cert_file: PathBuf,
    pub key_file: PathBuf,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ServerMode {
    Http,
    Https,
}

impl ServerMode {
    pub fn protocol(&self) -> &str {
        match self {
            Self::Http => "http",
            Self::Https => "https",
        }
    }
}

// Custom type for Host
#[derive(Debug, Clone)]
pub enum Host {
    Ip(IpAddr),
    Hostname(String),
}

#[derive(Clone, Deserialize)]
pub struct ThreadPool {
    pub pool_size: usize,
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self {
            pool_size: num_cpus::get(),
        }
    }
}

impl std::fmt::Display for Host {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Host::Ip(ip) => ip.fmt(f),
            Host::Hostname(s) => write!(f, "{s}"),
        }
    }
}

impl<'de> Deserialize<'de> for Host {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        if let Ok(address) = &s.parse::<Ipv6Addr>() {
            return Ok(Host::Ip(IpAddr::V6(*address)));
        }
        if let Ok(address) = &s.parse::<Ipv4Addr>() {
            return Ok(Host::Ip(IpAddr::V4(*address)));
        }
        Ok(Host::Hostname(s))
    }
}

// Custom type for Port
#[derive(Debug, Clone, Copy)]
pub struct Port(u16);

impl std::fmt::Display for Port {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Conversion from/to raw port numbers
impl From<u16> for Port {
    fn from(port: u16) -> Self {
        Port(port)
    }
}

impl From<Port> for u16 {
    fn from(port: Port) -> Self {
        port.0
    }
}

impl<'de> Deserialize<'de> for Port {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let number = u16::deserialize(d)?;
        Ok(Self(number))
    }
}

// Custom type that wraps Host and Port
pub struct HostPort<'a>(&'a Host, &'a Port);

// Implement ToSocketAddrs for the HostPort type
impl ToSocketAddrs for HostPort<'_> {
    type Iter = std::vec::IntoIter<SocketAddr>;

    fn to_socket_addrs(&self) -> io::Result<Self::Iter> {
        match &self.0 {
            Host::Ip(ip) => {
                let socket = SocketAddr::new(*ip, self.1 .0);
                Ok(vec![socket].into_iter())
            }
            Host::Hostname(hostname) => {
                let addresses = (hostname.as_str(), self.1 .0).to_socket_addrs()?;
                Ok(addresses.collect::<Vec<_>>().into_iter())
            }
        }
    }
}

#[derive(Deserialize, Clone)]
pub struct Server {
    pub host: Host,
    pub port: Port,
    pub ssl: Ssl,
    pub mode: ServerMode,
}

impl Server {
    pub fn listen_address(&self) -> HostPort {
        HostPort(&self.host, &self.port)
    }
}

#[derive(Deserialize, Clone)]
pub struct Hnsw {
    pub default_neighbors_count: usize,
    pub default_level_0_neighbors_count: usize,
    pub default_ef_construction: u32,
    pub default_ef_search: u32,
    pub default_num_layer: u8,
    pub default_max_cache_size: usize,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum VectorsIndexingMode {
    Sequential,
    Batch { batch_size: usize },
}

#[derive(Deserialize, Clone)]
pub struct Indexing {
    pub clamp_margin_percent: f32,
    #[serde(flatten)]
    pub mode: VectorsIndexingMode,
}

#[derive(Deserialize, Clone)]
pub struct Search {
    pub shortlist_size: usize,
    pub early_terminate_threshold: f32,
}

pub fn load_config() -> Result<Config, WaCustomError> {
    let config_path = get_config_path();

    let config_contents = std::fs::read_to_string(&config_path)
        .map_err(|e| WaCustomError::ConfigError(format!("Failed to read config file: {}", e)))?;

    toml::from_str(&config_contents)
        .map_err(|e| WaCustomError::ConfigError(format!("Invalid config format: {}", e)))
}

#[derive(Debug, Deserialize, Clone)]
pub struct CacheConfig {
    // Maximum number of collections to keep in memory
    #[serde(default = "default_max_collections")]
    pub max_collections: usize,
    // Probability factor for eviction (0.0-1.0)
    #[serde(default = "default_eviction_probability")]
    pub eviction_probability: f32,
}

fn default_max_collections() -> usize {
    10
}

fn default_eviction_probability() -> f32 {
    0.03125 // Similar to the existing LRUCache default
}

// Implement Default for CacheConfig
impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_collections: default_max_collections(),
            eviction_probability: default_eviction_probability(),
        }
    }
}
