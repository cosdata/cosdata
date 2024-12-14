use std::ptr;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use self::vectors::dtos::{CreateDenseVectorDto, UpsertDto};

use super::{dtos::CreateTransactionResponseDto, error::TransactionError};
use crate::models::meta_persist::update_current_version;
use crate::models::types::DenseIndexTransaction;
use crate::models::versioning::Hash;
use crate::{
    api::vectordb::vectors::{self, dtos::CreateVectorResponseDto},
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

    if !vec_store
        .current_open_transaction
        .load(Ordering::SeqCst)
        .is_null()
    {
        return Err(TransactionError::OnGoingTransaction);
    }

    let transaction = DenseIndexTransaction::new(vec_store.clone())
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
    let transaction_id = transaction.id;

    vec_store
        .current_open_transaction
        .store(Box::into_raw(Box::new(transaction)), Ordering::SeqCst);

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

    let current_open_transaction = unsafe {
        let ptr = vec_store.current_open_transaction.load(Ordering::SeqCst);

        if ptr.is_null() {
            return Err(TransactionError::NotFound);
        }

        ptr::read(ptr)
    };
    let current_transaction_id = current_open_transaction.id;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    current_open_transaction
        .pre_commit()
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    vec_store
        .current_version
        .clone()
        .update(current_transaction_id);
    vec_store
        .current_open_transaction
        .store(ptr::null_mut(), Ordering::SeqCst);
    update_current_version(&vec_store.lmdb, current_transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    Ok(())
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    create_vector_dto: CreateDenseVectorDto,
) -> Result<CreateVectorResponseDto, TransactionError> {
    let vec_store = ctx
        .ain_env
        .collections_map
        .get(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction = unsafe {
        vec_store
            .current_open_transaction
            .load(Ordering::SeqCst)
            .as_ref()
            .ok_or(TransactionError::NotFound)?
    };

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

    let current_open_transaction = unsafe {
        let ptr = vec_store.current_open_transaction.load(Ordering::SeqCst);

        if ptr.is_null() {
            return Err(TransactionError::NotFound);
        }

        ptr::read(ptr)
    };
    let current_transaction_id = current_open_transaction.id;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    current_open_transaction
        .pre_commit()
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    vec_store
        .current_open_transaction
        .store(ptr::null_mut(), Ordering::SeqCst);

    Ok(())
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    _transaction_id: Hash,
    _vector_id: u32,
) -> Result<(), TransactionError> {
    let _collection = ctx
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

    let current_open_transaction = unsafe {
        vec_store
            .current_open_transaction
            .load(Ordering::SeqCst)
            .as_ref()
            .ok_or(TransactionError::NotFound)?
    };

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
