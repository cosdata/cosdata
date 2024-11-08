use serde::{Deserialize, Serialize};

use crate::models::rpc::{Vector, VectorIdValue};

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

#[derive(Deserialize)]
pub(crate) struct UpdateVectorDto {
    pub values: Vec<f32>,
}

#[derive(Serialize)]
pub(crate) struct UpdateVectorResponseDto {
    pub id: VectorIdValue,
    pub values: Vec<f32>,
    // pub created_at: String
}

#[derive(Deserialize)]
pub(crate) struct FindSimilarVectorsDto {
    pub vector: Vec<f32>,
    pub k: i32,
}

#[derive(Serialize)]
pub(crate) struct SimilarVector {
    pub id: VectorIdValue,
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
