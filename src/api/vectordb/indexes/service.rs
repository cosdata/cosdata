use super::{dtos::CreateIndexDto, error::IndexesError, repo};

pub(crate) async fn create_index(create_index_dto: CreateIndexDto) -> Result<(), IndexesError> {
    repo::create_index(
        create_index_dto.collection_name,
        create_index_dto.name,
        create_index_dto.distance_metric_type,
        create_index_dto.quantization,
        create_index_dto.data_type,
        create_index_dto.index_params,
    )
    .await
}
