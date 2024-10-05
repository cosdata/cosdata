use std::sync::Arc;

use super::{dtos::CreateTransactionResponseDto, error::TransactionError};
use crate::{
    api::vectordb::vectors::{
        self,
        dtos::{CreateVectorDto, CreateVectorResponseDto},
    },
    app_context::AppContext,
    models::{types::get_app_env, versioning::Version},
};
use chrono::Utc;

// creates a transaction for a specific collection (vector store)
pub(crate) async fn create_transaction(
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    let env = get_app_env().map_err(|_| TransactionError::FailedToGetAppEnv)?;

    let vec_store = env
        .vector_store_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();

    if current_open_transaction_arc.get().is_some() {
        return Err(TransactionError::OnGoingTransaction);
    }

    let transaction_id = vec_store
        .vcs
        .generate_hash("new_transaction", Version::from(0))
        .map_err(|_| TransactionError::FailedToCreateTransaction("LMDB Error".to_string()))?;

    current_open_transaction_arc.update(Some(transaction_id));

    Ok(CreateTransactionResponseDto {
        transaction_id: transaction_id.to_string(),
        created_at: Utc::now(),
    })
}

// commits a transaction for a specific collection (vector store)
pub(crate) async fn commit_transaction(
    collection_id: &str,
    transaction_id: &str,
) -> Result<(), TransactionError> {
    // initializing environment
    let env = get_app_env().map_err(|_| TransactionError::FailedToGetAppEnv)?;

    let vec_store = env
        .vector_store_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();
    let current_open_transaction = current_open_transaction_arc.get();
    let current_transaction_id = current_open_transaction.ok_or(TransactionError::NotFound)?;

    if current_transaction_id.to_string() != transaction_id {
        return Err(TransactionError::NotFound);
    }

    vec_store
        .current_version
        .clone()
        .update(current_transaction_id);
    current_open_transaction_arc.update(None);

    Ok(())
}

pub(crate) async fn create_vector_without_committing(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: &str,
    create_vector_dto: CreateVectorDto,
) -> Result<CreateVectorResponseDto, TransactionError> {
    let env = get_app_env().map_err(|_| TransactionError::FailedToGetAppEnv)?;

    let vec_store = env
        .vector_store_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();

    if current_open_transaction_arc.get().is_none() {
        return Err(TransactionError::NotFound);
    }

    if current_open_transaction_arc.unwrap().to_string() != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    let vector =
        vectors::repo::create_vector_without_committing(ctx, collection_id, create_vector_dto)
            .await
            .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(vector)
}
