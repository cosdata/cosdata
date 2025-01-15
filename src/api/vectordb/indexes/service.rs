use std::sync::Arc;

use crate::app_context::AppContext;

use super::{dtos::CreateIndexDto, error::IndexesError, repo};

pub(crate) async fn create_index(
    create_index_dto: CreateIndexDto,
    ctx: Arc<AppContext>,
) -> Result<(), IndexesError> {
    repo::create_index(
        ctx,
        create_index_dto.collection_name,
        create_index_dto.name,
        create_index_dto.distance_metric_type,
        create_index_dto.quantization,
        create_index_dto.index,
    )
    .await
}
