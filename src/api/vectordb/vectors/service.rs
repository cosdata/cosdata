use super::{dtos::CreateVectorDto, error::VectorsError, repo};

pub(crate) async fn create_vector(
    collection_id: &str,
    create_vector_dto: CreateVectorDto,
) -> Result<String, VectorsError> {
    repo::create_vector(collection_id, create_vector_dto).await
}

pub(crate) async fn get_vector_by_id(
    collection_id: &str,
    vector_id: &str,
) -> Result<String, VectorsError> {
    repo::get_vector_by_id(collection_id, vector_id).await
}
