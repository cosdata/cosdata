use std::sync::Arc;

use crate::models::versioning::Hash;

use crate::{
    api::vectordb::collections,
    api_service::{run_upload, run_upload_in_transaction},
    app_context::AppContext,
    models::rpc::VectorIdValue,
};

use super::{
    dtos::{
        CreateVectorDto, CreateVectorResponseDto, FindSimilarVectorsDto, SimilarVector,
        UpdateVectorDto, UpdateVectorResponseDto,
    },
    error::VectorsError,
};

pub(crate) async fn create_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_vector_dto: CreateVectorDto,
) -> Result<CreateVectorResponseDto, VectorsError> {
    let collection = collections::service::get_collection_by_id(ctx.clone(), collection_id)
        .await
        .map_err(|e| VectorsError::FailedToCreateVector(e.to_string()))?;

    let mut current_open_transaction_arc = collection.current_open_transaction.clone();

    if current_open_transaction_arc.get().is_some() {
        return Err(VectorsError::FailedToCreateVector(
            "there is an ongoing transaction!".into(),
        ));
    }

    run_upload(
        ctx,
        collection,
        vec![(
            create_vector_dto.id.clone(),
            create_vector_dto.values.clone(),
        )],
    )
    .map_err(VectorsError::WaCustom)?;
    Ok(CreateVectorResponseDto {
        id: create_vector_dto.id,
        values: create_vector_dto.values,
    })
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    create_vector_dto: CreateVectorDto,
) -> Result<CreateVectorResponseDto, VectorsError> {
    let collection = collections::service::get_collection_by_id(ctx.clone(), collection_id)
        .await
        .map_err(|e| VectorsError::FailedToCreateVector(e.to_string()))?;
    run_upload_in_transaction(
        collection,
        transaction_id,
        vec![(
            create_vector_dto.id.clone(),
            create_vector_dto.values.clone(),
        )],
    )
    .map_err(VectorsError::WaCustom)?;
    Ok(CreateVectorResponseDto {
        id: create_vector_dto.id,
        values: create_vector_dto.values,
    })
}

pub(crate) async fn get_vector_by_id(
    _collection_id: &str,
    _vector_id: &str,
) -> Result<CreateVectorResponseDto, VectorsError> {
    Err(VectorsError::NotImplemented)?
}

pub(crate) async fn update_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorIdValue,
    update_vector_dto: UpdateVectorDto,
) -> Result<UpdateVectorResponseDto, VectorsError> {
    let collection = collections::service::get_collection_by_id(ctx.clone(), collection_id)
        .await
        .map_err(|e| VectorsError::FailedToUpdateVector(e.to_string()))?;

    let mut current_open_transaction_arc = collection.current_open_transaction.clone();

    if current_open_transaction_arc.get().is_some() {
        return Err(VectorsError::FailedToUpdateVector(
            "there is an ongoing transaction!".into(),
        ));
    }

    run_upload(
        ctx,
        collection,
        vec![(vector_id.clone(), update_vector_dto.values.clone())],
    )
    .map_err(VectorsError::WaCustom)?;
    Ok(UpdateVectorResponseDto {
        id: vector_id,
        values: update_vector_dto.values,
    })
}

pub(crate) async fn find_similar_vectors(
    find_similar_vectors: FindSimilarVectorsDto,
) -> Result<Vec<SimilarVector>, VectorsError> {
    if find_similar_vectors.vector.len() == 0 {
        return Err(VectorsError::FailedToFindSimilarVectors(
            "Vector shouldn't be empty".to_string(),
        ));
    }
    Ok(vec![SimilarVector {
        id: VectorIdValue::IntValue(find_similar_vectors.k),
        score: find_similar_vectors.vector[0],
    }])
}

pub(crate) async fn delete_vector_by_id(
    _collection_id: &str,
    _vector_id: VectorIdValue,
) -> Result<CreateVectorResponseDto, VectorsError> {
    Err(VectorsError::NotImplemented)?
}
