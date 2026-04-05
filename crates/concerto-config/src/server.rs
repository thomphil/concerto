//! The `[server]` section of the configuration.

use serde::{Deserialize, Serialize};

/// HTTP server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Interface to bind to. Defaults to `0.0.0.0` (all interfaces).
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to bind to. Defaults to `8000`.
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8000
}
