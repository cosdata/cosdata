use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub(crate) struct CreateVectorDto {
    pub id: String,
    pub values: Vec<u8>,
}

#[derive(Serialize)]
pub(crate) struct VectorResponseDto {
    pub id: String,
    pub values: Vec<u8>,
    pub created_at: String,
}
