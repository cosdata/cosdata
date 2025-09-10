use std::sync::Arc;

use self::vectors::dtos::CreateVectorDto;

use super::{dtos::CreateTransactionResponseDto, error::TransactionError};
use crate::models::collection::OmVectorEmbedding;
use crate::models::collection_transaction::{
    ExplicitTransaction, ExplicitTransactionID, TransactionStatus,
};
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

    let transaction = ExplicitTransaction::new(&collection, &ctx.config)
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
    let transaction_id = transaction.id;

    *current_open_transaction_guard = Some(transaction);

    Ok(CreateTransactionResponseDto {
        transaction_id,
        created_at: Utc::now(),
    })
}

// commits a transaction for a specific collection
pub(crate) async fn commit_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: ExplicitTransactionID,
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
    let current_transaction_id = current_open_transaction.id;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    let mut last_allotted_version = collection.last_allotted_version.write();
    *last_allotted_version = VersionNumber::from(**last_allotted_version + 1);

    let allotted_version = *last_allotted_version;

    let records_upserted = current_open_transaction.wal.records_upserted();
    let records_deleted = current_open_transaction.wal.records_deleted();
    let total_operations = current_open_transaction.wal.total_operations();

    current_open_transaction
        .pre_commit(&collection, allotted_version)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    *current_version_guard = allotted_version;

    collection
        .vcs
        .set_current_version_explicit(
            allotted_version,
            current_transaction_id,
            records_upserted,
            records_deleted,
            total_operations,
        )
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, allotted_version)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    collection.trigger_indexing(current_transaction_id, allotted_version);

    Ok(())
}

pub(crate) async fn get_transaction_status(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: ExplicitTransactionID,
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
    transaction_id: ExplicitTransactionID,
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

    if current_open_transaction.id != transaction_id {
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
    transaction_id: ExplicitTransactionID,
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
    let current_transaction_id = current_open_transaction.id;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    Ok(())
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: ExplicitTransactionID,
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

    if current_open_transaction.id != transaction_id {
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
    transaction_id: ExplicitTransactionID,
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

    if current_open_transaction.id != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    vectors::repo::upsert_vectors_in_transaction(&collection, current_open_transaction, vectors)
        .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}

pub(crate) async fn upsert_om_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: ExplicitTransactionID,
    vectors: Vec<OmVectorEmbedding>,
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

    if current_open_transaction.id != transaction_id {
        return Err(TransactionError::FailedToCreateVector(
            "This is not the currently open transaction!".into(),
        ));
    }

    vectors::repo::upsert_om_vectors_in_transaction(&collection, current_open_transaction, vectors)
        .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}
