use serde::{Deserialize, Serialize};

use crate::models::rpc::Vector;

#[derive(Deserialize)]
pub(crate) struct CreateVectorDto {
    pub id: u64,
    pub values: Vec<f32>,
}

#[derive(Serialize)]
pub(crate) struct CreateVectorResponseDto {
    pub id: u64,
    pub values: Vec<f32>,
    // pub created_at: String
}

#[derive(Deserialize)]
pub(crate) struct UpdateVectorDto {
    pub values: Vec<f32>,
}

#[derive(Serialize)]
pub(crate) struct UpdateVectorResponseDto {
    pub id: u64,
    pub values: Vec<f32>,
    // pub created_at: String
}

#[derive(Deserialize)]
pub(crate) struct FindSimilarVectorsDto {
    pub vector: Vec<f32>,
    pub k: u64,
}

#[derive(Serialize)]
pub(crate) struct SimilarVector {
    pub id: u64,
    pub score: f32,
}

#[derive(Serialize)]
pub(crate) struct FindSimilarVectorsResponseDto {
    pub results: Vec<SimilarVector>,
}

#[derive(Deserialize)]
pub(crate) struct UpsertDto {
    pub vectors: Vec<Vector>,
}
