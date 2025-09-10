use std::sync::Arc;

use crate::{
    api::vectordb::{transactions::error::TransactionError, vectors::dtos::CreateVectorDto},
    app_context::AppContext,
    models::{collection::OmVectorEmbedding, indexing_manager::IndexingManager, types::VectorId},
};

pub(crate) async fn upsert_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vectors: Vec<CreateVectorDto>,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let txn = collection.current_implicit_transaction.read();

    IndexingManager::implicit_txn_upsert(
        &collection,
        &txn,
        &ctx.config,
        vectors.into_iter().map(Into::into).collect(),
    )
    .map_err(|err| TransactionError::FailedToCreateVector(err.to_string()))?;

    Ok(())
}

pub(crate) async fn upsert_om_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vectors: Vec<OmVectorEmbedding>,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let txn = collection.current_implicit_transaction.read();

    IndexingManager::implicit_txn_om_upsert(&collection, &txn, &ctx.config, vectors)
        .map_err(|err| TransactionError::FailedToCreateVector(err.to_string()))?;

    Ok(())
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let txn = collection.current_implicit_transaction.read();

    IndexingManager::implicit_txn_delete(&collection, &txn, &ctx.config, vector_id)
        .map_err(|err| TransactionError::FailedToDeleteVector(err.to_string()))?;

    Ok(())
}
