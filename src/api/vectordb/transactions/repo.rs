use std::sync::Arc;

use self::vectors::dtos::CreateVectorDto;

use super::{dtos::CreateTransactionResponseDto, error::TransactionError};
use crate::models::collection_transaction::CollectionTransaction;
use crate::models::meta_persist::update_current_version;
use crate::models::versioning::Hash;
use crate::{api::vectordb::vectors, app_context::AppContext};
use chrono::Utc;

// creates a transaction for a specific collection
pub(crate) async fn create_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_guard = collection.current_open_transaction.write().unwrap();

    if current_open_transaction_guard.is_some() {
        return Err(TransactionError::OnGoingTransaction);
    }

    let transaction = CollectionTransaction::new(collection.clone())
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
    let transaction_id = transaction.id;

    *current_open_transaction_guard = Some(transaction);

    Ok(CreateTransactionResponseDto {
        transaction_id: transaction_id.to_string(),
        created_at: Utc::now(),
    })
}

// commits a transaction for a specific collection
pub(crate) async fn commit_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_version_guard = collection.current_version.write().unwrap();

    let mut current_open_transaction_guard = collection.current_open_transaction.write().unwrap();
    let Some(current_open_transaction) = current_open_transaction_guard.take() else {
        return Err(TransactionError::NotFound);
    };
    let current_transaction_id = current_open_transaction.id;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    let version_number = current_open_transaction.version_number;

    current_open_transaction
        .pre_commit(&collection, &ctx.config)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    *current_version_guard = current_transaction_id;
    collection
        .vcs
        .set_branch_version("main", version_number.into(), current_transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, current_transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    Ok(())
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    create_vector_dto: CreateVectorDto,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction_guard = collection.current_open_transaction.read().unwrap();
    let Some(current_open_transaction) = &*current_open_transaction_guard else {
        return Err(TransactionError::NotFound);
    };

    if current_open_transaction.id != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    vectors::repo::create_vector_in_transaction(
        ctx,
        &collection,
        current_open_transaction,
        create_vector_dto,
    )
    .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}

// aborts the currently open transaction of a ollection
pub(crate) async fn abort_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_guard = collection.current_open_transaction.write().unwrap();
    let Some(current_open_transaction) = current_open_transaction_guard.take() else {
        return Err(TransactionError::NotFound);
    };
    let current_transaction_id = current_open_transaction.id;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    current_open_transaction
        .pre_commit(&collection, &ctx.config)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    Ok(())
}

pub(crate) async fn delete_vector_by_id(
    _ctx: Arc<AppContext>,
    _collection_id: &str,
    _transaction_id: Hash,
    _vector_id: u32,
) -> Result<(), TransactionError> {
    unimplemented!();
}

pub(crate) async fn upsert_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    vectors: Vec<CreateVectorDto>,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction_guard = collection.current_open_transaction.read().unwrap();
    let Some(current_open_transaction) = &*current_open_transaction_guard else {
        return Err(TransactionError::NotFound);
    };

    if current_open_transaction.id != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    vectors::repo::upsert_vectors_in_transaction(
        ctx,
        &collection,
        current_open_transaction,
        vectors,
    )
    .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}
