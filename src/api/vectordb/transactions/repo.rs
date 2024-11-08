use std::sync::Arc;

use self::vectors::dtos::UpsertDto;

use super::{dtos::CreateTransactionResponseDto, error::TransactionError};
use crate::convert_value;
use crate::models::rpc::VectorIdValue;
use crate::models::versioning::Hash;
use crate::vector_store::auto_commit_transaction;
use crate::{
    api::vectordb::vectors::{
        self,
        dtos::{CreateVectorDto, CreateVectorResponseDto},
    },
    app_context::AppContext,
    models::versioning::Version,
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

    let branch_info = vec_store
        .vcs
        .get_branch_info("main")
        .map_err(|e| TransactionError::FailedToCreateTransaction(e.to_string()))?;

    let branch_info = branch_info.ok_or(TransactionError::FailedToCreateTransaction(
        "the main branch is not found!".into(),
    ))?;

    let current_version = branch_info.get_current_version();

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();

    if current_open_transaction_arc.get().is_some() {
        return Err(TransactionError::OnGoingTransaction);
    }

    let transaction_id = vec_store
        .vcs
        .generate_hash("main", Version::from(*current_version + 1))
        .map_err(|_| TransactionError::FailedToCreateTransaction("LMDB Error".to_string()))?;

    current_open_transaction_arc.update(Some(transaction_id));

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
    let current_open_transaction = current_open_transaction_arc.get();
    let current_transaction_id = current_open_transaction.ok_or(TransactionError::NotFound)?;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    auto_commit_transaction(vec_store.clone())
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

    let current_open_transaction_id = current_open_transaction_arc
        .get()
        .ok_or(TransactionError::NotFound)?;

    if current_open_transaction_id != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    let vector = vectors::repo::create_vector_in_transaction(
        ctx,
        collection_id,
        current_open_transaction_id,
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
    let current_transaction_id = current_open_transaction.ok_or(TransactionError::NotFound)?;

    if current_transaction_id != transaction_id {
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

    crate::vector_store::delete_vector_by_id_in_transaction(
        collection,
        convert_value(vector_id.clone()),
        transaction_id,
    )
    .map_err(|e| TransactionError::FailedToDeleteVector(e.to_string()))?;

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

    let current_open_transaction_id = current_open_transaction_arc
        .get()
        .ok_or(TransactionError::NotFound)?;

    if current_open_transaction_id != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    vectors::repo::upsert_in_transaction(
        ctx,
        collection_id,
        current_open_transaction_id,
        upsert_dto,
    )
    .await
    .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}
