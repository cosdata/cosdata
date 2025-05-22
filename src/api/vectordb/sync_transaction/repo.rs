use std::sync::Arc;

use crate::{
    api::vectordb::{transactions::error::TransactionError, vectors::dtos::CreateVectorDto},
    app_context::AppContext,
    models::{
        collection_transaction::CollectionTransaction, indexing_manager::IndexingManager,
        meta_persist::update_current_version, types::VectorId, wal::VectorOp,
    },
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

    let mut current_open_transaction_guard = collection.current_open_transaction.write();

    while current_open_transaction_guard.is_some() {
        drop(current_open_transaction_guard);
        current_open_transaction_guard = collection.current_open_transaction.write();
    }

    let mut current_version = collection.current_version.write();

    let transaction = CollectionTransaction::new(&collection)
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;

    transaction
        .wal
        .append(VectorOp::Upsert(
            vectors.into_iter().map(Into::into).collect(),
        ))
        .map_err(|err| TransactionError::FailedToCreateVector(err.to_string()))?;

    let version = transaction.version;

    transaction
        .pre_commit()
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    *current_version = version;
    collection
        .vcs
        .set_current_version(version, true)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, version)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    IndexingManager::index_version(&collection, &ctx.config, version)
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

    let transaction = CollectionTransaction::new(&collection)
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;

    transaction
        .wal
        .append(VectorOp::Delete(vector_id))
        .map_err(|err| TransactionError::FailedToCreateVector(err.to_string()))?;

    let version = transaction.version;

    transaction
        .pre_commit()
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    *current_version = version;
    collection
        .vcs
        .set_current_version(version, true)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, version)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    IndexingManager::index_version(&collection, &ctx.config, version)
        .map_err(|err| TransactionError::FailedToCreateVector(err.to_string()))?;

    Ok(())
}
