use std::sync::Arc;

use self::vectors::dtos::UpsertDto;

use super::{dtos::CreateTransactionResponseDto, error::TransactionError};
use crate::models::rpc::VectorIdValue;
use crate::models::types::DenseIndexTransaction;
use crate::models::versioning::Hash;
use crate::{
    api::vectordb::vectors::{
        self,
        dtos::{CreateVectorDto, CreateVectorResponseDto},
    },
    app_context::AppContext,
};
use chrono::Utc;

// creates a transaction for a specific collection (vector store)
pub(crate) async fn create_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    let vec_store = ctx
        .ain_env
        .collections_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();

    if current_open_transaction_arc.get().is_some() {
        return Err(TransactionError::OnGoingTransaction);
    }

    let transaction = DenseIndexTransaction::new(vec_store.clone())
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
    let transaction_id = transaction.id;

    current_open_transaction_arc.update(Some(transaction));

    Ok(CreateTransactionResponseDto {
        transaction_id: transaction_id.to_string(),
        created_at: Utc::now(),
    })
}

// commits a transaction for a specific collection (vector store)
pub(crate) async fn commit_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    let vec_store = ctx
        .ain_env
        .collections_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();
    let current_open_transaction = current_open_transaction_arc.get().clone();
    let current_transaction = current_open_transaction.ok_or(TransactionError::NotFound)?;
    let current_transaction_id = current_transaction.id;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    current_transaction
        .pre_commit()
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    vec_store
        .current_version
        .clone()
        .update(current_transaction_id);
    current_open_transaction_arc.update(None);

    Ok(())
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    create_vector_dto: CreateVectorDto,
) -> Result<CreateVectorResponseDto, TransactionError> {
    let vec_store = ctx
        .ain_env
        .collections_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();

    let current_open_transaction = current_open_transaction_arc
        .get()
        .clone()
        .ok_or(TransactionError::NotFound)?;

    if current_open_transaction.id != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    let vector = vectors::repo::create_vector_in_transaction(
        ctx,
        collection_id,
        current_open_transaction,
        create_vector_dto,
    )
    .await
    .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(vector)
}

// aborts the currently open transaction of a specific collection (vector store)
pub(crate) async fn abort_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    let vec_store = ctx
        .ain_env
        .collections_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();
    let current_open_transaction = current_open_transaction_arc.get();
    let current_transaction = current_open_transaction
        .as_ref()
        .ok_or(TransactionError::NotFound)?;

    if current_transaction.id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    current_open_transaction_arc.update(None);

    Ok(())
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    vector_id: VectorIdValue,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    // TODO(a-rustacean): uncomment
    // crate::vector_store::delete_vector_by_id_in_transaction(
    //     collection,
    //     convert_value(vector_id.clone()),
    //     transaction_id,
    // )
    // .map_err(|e| TransactionError::FailedToDeleteVector(e.to_string()))?;

    Ok(())
}

pub(crate) async fn upsert(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    upsert_dto: UpsertDto,
) -> Result<(), TransactionError> {
    let vec_store = ctx
        .ain_env
        .collections_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();

    let current_open_transaction = current_open_transaction_arc
        .get()
        .clone()
        .ok_or(TransactionError::NotFound)?;

    if current_open_transaction.id != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    vectors::repo::upsert_in_transaction(ctx, collection_id, current_open_transaction, upsert_dto)
        .await
        .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}
