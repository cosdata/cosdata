use serde::{Deserialize, Serialize};

use crate::models::rpc::VectorIdValue;

#[derive(Deserialize)]
pub(crate) struct CreateVectorDto {
    pub id: VectorIdValue,
    pub values: Vec<f32>,
}


#[derive(Serialize)]
pub(crate) struct CreateVectorResponseDto {
    pub id: VectorIdValue,
    pub values: Vec<f32>,
    // pub created_at: String
}
