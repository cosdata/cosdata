use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone)]
pub enum EncodingFormat {
    CBOR,
    JSON,
    DEFAULT,
}
