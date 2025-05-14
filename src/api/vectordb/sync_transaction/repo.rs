use std::sync::Arc;

use crate::{
    api::vectordb::{transactions::error::TransactionError, vectors::dtos::CreateVectorDto},
    app_context::AppContext,
    models::{
        collection_transaction::{CollectionTransaction, TransactionStatus},
        indexing_manager::IndexingManager,
        meta_persist::update_current_version,
        types::VectorId,
        wal::VectorOp,
    },
};
use chrono::Utc;
use parking_lot::RwLock;

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

    let mut current_open_transaction_guard = collection.current_open_transaction.write();

    while current_open_transaction_guard.is_some() {
        drop(current_open_transaction_guard);
        current_open_transaction_guard = collection.current_open_transaction.write();
    }

    let mut current_version = collection.current_version.write();

    let transaction = CollectionTransaction::new(&collection, true)
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;

    transaction
        .wal
        .append(VectorOp::Upsert(
            vectors.into_iter().map(Into::into).collect(),
        ))
        .map_err(|err| TransactionError::FailedToCreateVector(err.to_string()))?;

    let version_number = transaction.version_number;
    let transaction_id = transaction.id;

    transaction
        .pre_commit()
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    *current_version = transaction_id;
    collection
        .vcs
        .set_branch_version("main", version_number, transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    collection.transaction_status_map.insert(
        transaction_id,
        &transaction_id,
        RwLock::new(TransactionStatus::NotStarted {
            last_updated: Utc::now(),
        }),
    );

    IndexingManager::index_version(&collection, &ctx.config, transaction_id)
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

    let mut current_open_transaction_guard = collection.current_open_transaction.write();

    while current_open_transaction_guard.is_some() {
        drop(current_open_transaction_guard);
        current_open_transaction_guard = collection.current_open_transaction.write();
    }

    let mut current_version = collection.current_version.write();

    let transaction = CollectionTransaction::new(&collection, true)
        .map_err(|err| TransactionError::FailedToCreateVector(err.to_string()))?;

    transaction
        .wal
        .append(VectorOp::Delete(vector_id))
        .map_err(|err| TransactionError::FailedToDeleteVector(err.to_string()))?;

    let version_number = transaction.version_number;
    let transaction_id = transaction.id;

    transaction
        .pre_commit()
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    *current_version = transaction_id;
    collection
        .vcs
        .set_branch_version("main", version_number, transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    collection.transaction_status_map.insert(
        transaction_id,
        &transaction_id,
        RwLock::new(TransactionStatus::NotStarted {
            last_updated: Utc::now(),
        }),
    );

    IndexingManager::index_version(&collection, &ctx.config, transaction_id)
        .map_err(|err| TransactionError::FailedToDeleteVector(err.to_string()))?;

    Ok(())
}
