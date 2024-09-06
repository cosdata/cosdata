use super::{dtos::CreateVectorDto, error::VectorsError};

pub(crate) async fn create_vector(
    collection_id: &str,
    create_vector_dto: CreateVectorDto,
) -> Result<String, VectorsError> {
    Err(VectorsError::NotFound)?
}

pub(crate) async fn get_vector_by_id(
    collection_id: &str,
    vector_id: &str,
) -> Result<String, VectorsError> {
    Err(VectorsError::NotFound)?
}
