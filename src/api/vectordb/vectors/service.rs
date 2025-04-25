use std::sync::Arc;

use crate::{
    app_context::AppContext,
    models::types::{DocumentId, VectorId},
};

use super::{
    dtos::{CreateVectorDto, SimilarVector},
    error::VectorsError,
    repo,
};

pub(crate) async fn query_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    document_id: DocumentId,
) -> Result<Vec<CreateVectorDto>, VectorsError> {
    repo::query_vectors(ctx, collection_id, document_id).await
}

pub(crate) async fn get_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<CreateVectorDto, VectorsError> {
    repo::get_vector_by_id(ctx, collection_id, vector_id).await
}

pub(crate) async fn check_vector_existence(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<bool, VectorsError> {
    repo::check_vector_existence(ctx, collection_id, vector_id).await
}

pub(crate) async fn fetch_vector_neighbors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<Vec<SimilarVector>, VectorsError> {
    repo::fetch_vector_neighbors(ctx, collection_id, vector_id).await
}
