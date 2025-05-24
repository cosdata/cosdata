use crate::models::types::DistanceMetric;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// Create a wrapper type for DistanceMetric that can derive ToSchema
#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetricSchema {
    Cosine,
    Euclidean,
    Hamming,
    DotProduct,
}

// Implement From and Into traits to convert between the wrapper and real type
impl From<DistanceMetric> for DistanceMetricSchema {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Cosine => DistanceMetricSchema::Cosine,
            DistanceMetric::Euclidean => DistanceMetricSchema::Euclidean,
            DistanceMetric::Hamming => DistanceMetricSchema::Hamming,
            DistanceMetric::DotProduct => DistanceMetricSchema::DotProduct,
        }
    }
}

impl From<DistanceMetricSchema> for DistanceMetric {
    fn from(schema: DistanceMetricSchema) -> Self {
        match schema {
            DistanceMetricSchema::Cosine => DistanceMetric::Cosine,
            DistanceMetricSchema::Euclidean => DistanceMetric::Euclidean,
            DistanceMetricSchema::Hamming => DistanceMetric::Hamming,
            DistanceMetricSchema::DotProduct => DistanceMetric::DotProduct,
        }
    }
}
