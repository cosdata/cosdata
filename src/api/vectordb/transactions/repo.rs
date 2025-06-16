use std::sync::Arc;

use self::vectors::dtos::CreateVectorDto;

use super::{dtos::CreateTransactionResponseDto, error::TransactionError};
use crate::models::collection_transaction::{ExplicitTransaction, TransactionStatus};
use crate::models::meta_persist::update_current_version;
use crate::models::types::VectorId;
use crate::models::versioning::VersionNumber;
use crate::models::wal::VectorOp;
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

    let mut current_open_transaction_guard = collection.current_explicit_transaction.write();

    if current_open_transaction_guard.is_some() {
        return Err(TransactionError::OnGoingTransaction);
    }

    let transaction = ExplicitTransaction::new(&collection)
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
    let transaction_id = transaction.version;

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
    transaction_id: VersionNumber,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_version_guard = collection.current_version.write();

    let mut current_open_transaction_guard = collection.current_explicit_transaction.write();
    let Some(current_open_transaction) = current_open_transaction_guard.take() else {
        return Err(TransactionError::NotFound);
    };
    let current_transaction_id = current_open_transaction.version;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    current_open_transaction
        .pre_commit()
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    *current_version_guard = current_transaction_id;
    collection
        .vcs
        .set_current_version(current_transaction_id, false)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, current_transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    collection.trigger_indexing(current_transaction_id);

    Ok(())
}

pub(crate) async fn get_transaction_status(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
) -> Result<TransactionStatus, TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let status = collection
        .transaction_status_map
        .get_latest(&transaction_id)
        .ok_or(TransactionError::NotFound)?
        .read();

    Ok(status.clone())
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
    create_vector_dto: CreateVectorDto,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction_guard = collection.current_explicit_transaction.read();
    let Some(current_open_transaction) = &*current_open_transaction_guard else {
        return Err(TransactionError::NotFound);
    };

    if current_open_transaction.version != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    vectors::repo::create_vector_in_transaction(
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
    transaction_id: VersionNumber,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_open_transaction_guard = collection.current_explicit_transaction.write();
    let Some(current_open_transaction) = current_open_transaction_guard.take() else {
        return Err(TransactionError::NotFound);
    };
    let current_transaction_id = current_open_transaction.version;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    Ok(())
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
    vector_id: VectorId,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction_guard = collection.current_explicit_transaction.read();
    let Some(current_open_transaction) = &*current_open_transaction_guard else {
        return Err(TransactionError::NotFound);
    };

    if current_open_transaction.version != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    current_open_transaction
        .wal
        .append(VectorOp::Delete(vector_id))
        .map_err(|e| TransactionError::FailedToDeleteVector(e.to_string()))?;

    Ok(())
}

pub(crate) async fn upsert_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
    vectors: Vec<CreateVectorDto>,
) -> Result<(), TransactionError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction_guard = collection.current_explicit_transaction.read();
    let Some(current_open_transaction) = &*current_open_transaction_guard else {
        return Err(TransactionError::NotFound);
    };

    if current_open_transaction.version != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    vectors::repo::upsert_vectors_in_transaction(&collection, current_open_transaction, vectors)
        .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}
