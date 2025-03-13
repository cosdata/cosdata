use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone)]
#[serde(rename_all = "UPPERCASE")]
pub enum EncodingFormat {
    Cbor,
    Json,
    Default,
}
