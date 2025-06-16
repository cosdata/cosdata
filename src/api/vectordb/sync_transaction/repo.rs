use std::sync::Arc;

use chrono::Utc;
use parking_lot::RwLock;

use crate::{
    api::vectordb::{transactions::error::TransactionError, vectors::dtos::CreateVectorDto},
    app_context::AppContext,
    models::{
        collection_transaction::TransactionStatus, indexing_manager::IndexingManager,
        meta_persist::update_current_version, types::VectorId, versioning::VersionNumber,
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

    let mut current_open_transaction_guard = collection.current_explicit_transaction.write();

    while current_open_transaction_guard.is_some() {
        drop(current_open_transaction_guard);
        current_open_transaction_guard = collection.current_explicit_transaction.write();
    }

    let mut current_version = collection.current_version.write();

    let current_version_number = collection
        .vcs
        .get_current_version()
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
    let version = VersionNumber::from(*current_version_number + 1);

    *current_version = version;
    collection
        .vcs
        .set_current_version(version, true)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, version)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    IndexingManager::explicit_txn_upsert(
        &collection,
        version,
        &ctx.config,
        vectors.into_iter().map(Into::into).collect(),
    )
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

    let mut current_open_transaction_guard = collection.current_explicit_transaction.write();

    while current_open_transaction_guard.is_some() {
        drop(current_open_transaction_guard);
        current_open_transaction_guard = collection.current_explicit_transaction.write();
    }

    let mut current_version = collection.current_version.write();

    let current_version_number = collection
        .vcs
        .get_current_version()
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
    let version = VersionNumber::from(*current_version_number + 1);

    *current_version = version;
    collection
        .vcs
        .set_current_version(version, true)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    update_current_version(&collection.lmdb, version)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    collection.transaction_status_map.insert(
        version,
        &version,
        RwLock::new(TransactionStatus::NotStarted {
            last_updated: Utc::now(),
        }),
    );

    IndexingManager::explicit_txn_delete(&collection, version, &ctx.config, vector_id)
        .map_err(|err| TransactionError::FailedToDeleteVector(err.to_string()))?;

    Ok(())
}
