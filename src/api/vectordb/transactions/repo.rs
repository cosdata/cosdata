use std::ptr;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use self::vectors::dtos::{
    CreateDenseVectorDto, CreateSparseIdfDocumentDto, CreateSparseVectorDto,
};

use super::{dtos::CreateTransactionResponseDto, error::TransactionError};
use crate::indexes::hnsw::transaction::HNSWIndexTransaction;
use crate::indexes::inverted::transaction::InvertedIndexTransaction;
use crate::indexes::inverted_idf::transaction::InvertedIndexIDFTransaction;
use crate::models::meta_persist::update_current_version;
use crate::models::rpc::DenseVector;
use crate::models::versioning::Hash;
use crate::{api::vectordb::vectors, app_context::AppContext};
use chrono::Utc;

// creates a transaction for a specific collection (vector store)
pub(crate) async fn create_dense_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    if !hnsw_index
        .current_open_transaction
        .load(Ordering::SeqCst)
        .is_null()
    {
        return Err(TransactionError::OnGoingTransaction);
    }

    let transaction = HNSWIndexTransaction::new(hnsw_index.clone())
        .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
    let transaction_id = transaction.id;

    hnsw_index
        .current_open_transaction
        .store(Box::into_raw(Box::new(transaction)), Ordering::SeqCst);

    Ok(CreateTransactionResponseDto {
        transaction_id: transaction_id.to_string(),
        created_at: Utc::now(),
    })
}

pub(crate) async fn create_inverted_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    let transaction_id = if let Some(inverted_index) = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
    {
        if !inverted_index
            .current_open_transaction
            .load(Ordering::SeqCst)
            .is_null()
        {
            return Err(TransactionError::OnGoingTransaction);
        }

        let transaction = InvertedIndexTransaction::new(inverted_index.clone())
            .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
        let transaction_id = transaction.id;

        inverted_index
            .current_open_transaction
            .store(Box::into_raw(Box::new(transaction)), Ordering::SeqCst);
        transaction_id
    } else {
        let idf_inverted_index = ctx
            .ain_env
            .collections_map
            .get_idf_inverted_index(collection_id)
            .ok_or(TransactionError::CollectionNotFound)?;

        if !idf_inverted_index
            .current_open_transaction
            .load(Ordering::SeqCst)
            .is_null()
        {
            return Err(TransactionError::OnGoingTransaction);
        }

        let transaction = InvertedIndexIDFTransaction::new(idf_inverted_index.clone())
            .map_err(|err| TransactionError::FailedToCreateTransaction(err.to_string()))?;
        let transaction_id = transaction.id;

        idf_inverted_index
            .current_open_transaction
            .store(Box::into_raw(Box::new(transaction)), Ordering::SeqCst);
        transaction_id
    };

    Ok(CreateTransactionResponseDto {
        transaction_id: transaction_id.to_string(),
        created_at: Utc::now(),
    })
}

// commits a transaction for a specific collection (vector store)
pub(crate) async fn commit_dense_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let mut current_version = hnsw_index.current_version.write().unwrap();

    let current_open_transaction = unsafe {
        let ptr = hnsw_index.current_open_transaction.load(Ordering::SeqCst);

        if ptr.is_null() {
            return Err(TransactionError::NotFound);
        }

        ptr::read(ptr)
    };
    let current_transaction_id = current_open_transaction.id;

    if current_transaction_id != transaction_id {
        return Err(TransactionError::NotFound);
    }

    let version_number = current_open_transaction.version_number as u32;

    current_open_transaction
        .pre_commit(hnsw_index.clone())
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    *current_version = current_transaction_id;
    hnsw_index
        .vcs
        .set_branch_version("main", version_number.into(), current_transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    hnsw_index
        .current_open_transaction
        .store(ptr::null_mut(), Ordering::SeqCst);
    update_current_version(&hnsw_index.lmdb, current_transaction_id)
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    Ok(())
}

// commits a transaction for a specific collection (vector store)
pub(crate) async fn commit_sparse_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    if let Some(inverted_index) = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
    {
        let mut current_version = inverted_index.current_version.write().unwrap();

        let current_open_transaction = unsafe {
            let ptr = inverted_index
                .current_open_transaction
                .load(Ordering::SeqCst);

            if ptr.is_null() {
                return Err(TransactionError::NotFound);
            }

            ptr::read(ptr)
        };
        let current_transaction_id = current_open_transaction.id;

        if current_transaction_id != transaction_id {
            return Err(TransactionError::NotFound);
        }

        let version_number = current_open_transaction.version_number as u32;

        current_open_transaction
            .pre_commit(&inverted_index)
            .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

        *current_version = current_transaction_id;
        inverted_index
            .vcs
            .set_branch_version("main", version_number.into(), current_transaction_id)
            .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
        inverted_index
            .current_open_transaction
            .store(ptr::null_mut(), Ordering::SeqCst);
        update_current_version(&inverted_index.lmdb, current_transaction_id)
            .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    } else {
        let idf_inverted_index = ctx
            .ain_env
            .collections_map
            .get_idf_inverted_index(collection_id)
            .ok_or(TransactionError::CollectionNotFound)?;
        let mut current_version = idf_inverted_index.current_version.write().unwrap();

        let current_open_transaction = unsafe {
            let ptr = idf_inverted_index
                .current_open_transaction
                .load(Ordering::SeqCst);

            if ptr.is_null() {
                return Err(TransactionError::NotFound);
            }

            ptr::read(ptr)
        };
        let current_transaction_id = current_open_transaction.id;

        if current_transaction_id != transaction_id {
            return Err(TransactionError::NotFound);
        }

        let version_number = current_open_transaction.version_number as u32;

        current_open_transaction
            .pre_commit(idf_inverted_index.clone())
            .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

        *current_version = current_transaction_id;
        idf_inverted_index
            .vcs
            .set_branch_version("main", version_number.into(), current_transaction_id)
            .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
        idf_inverted_index
            .current_open_transaction
            .store(ptr::null_mut(), Ordering::SeqCst);
        update_current_version(&idf_inverted_index.lmdb, current_transaction_id)
            .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;
    }

    Ok(())
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    create_vector_dto: CreateDenseVectorDto,
) -> Result<(), TransactionError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction = unsafe {
        hnsw_index
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

    vectors::repo::create_vector_in_transaction(
        ctx,
        collection_id,
        current_open_transaction,
        create_vector_dto,
    )
    .await
    .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}

// aborts the currently open transaction of a specific collection (vector store)
pub(crate) async fn abort_dense_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction = unsafe {
        let ptr = hnsw_index.current_open_transaction.load(Ordering::SeqCst);

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
        .pre_commit(hnsw_index.clone())
        .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

    hnsw_index
        .current_open_transaction
        .store(ptr::null_mut(), Ordering::SeqCst);

    Ok(())
}

// aborts the currently open transaction of a specific collection (vector store)
pub(crate) async fn abort_sparse_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    if let Some(inverted_index) = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
    {
        let current_open_transaction = unsafe {
            let ptr = inverted_index
                .current_open_transaction
                .load(Ordering::SeqCst);

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
            .pre_commit(&inverted_index)
            .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

        inverted_index
            .current_open_transaction
            .store(ptr::null_mut(), Ordering::SeqCst);
    } else {
        let idf_inverted_index = ctx
            .ain_env
            .collections_map
            .get_idf_inverted_index(collection_id)
            .ok_or(TransactionError::CollectionNotFound)?;

        let current_open_transaction = unsafe {
            let ptr = idf_inverted_index
                .current_open_transaction
                .load(Ordering::SeqCst);

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
            .pre_commit(idf_inverted_index.clone())
            .map_err(|err| TransactionError::FailedToCommitTransaction(err.to_string()))?;

        idf_inverted_index
            .current_open_transaction
            .store(ptr::null_mut(), Ordering::SeqCst);
    }

    Ok(())
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    _transaction_id: Hash,
    _vector_id: u32,
) -> Result<(), TransactionError> {
    let _hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    unimplemented!();
}

pub(crate) async fn upsert_dense_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    vectors: Vec<DenseVector>,
) -> Result<(), TransactionError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction = unsafe {
        hnsw_index
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

    vectors::repo::upsert_dense_vectors_in_transaction(
        ctx,
        hnsw_index,
        current_open_transaction,
        vectors,
    )
    .await
    .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}

pub(crate) async fn upsert_sparse_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    vectors: Vec<CreateSparseVectorDto>,
) -> Result<(), TransactionError> {
    let inverted_index = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction = unsafe {
        inverted_index
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

    vectors::repo::upsert_sparse_vectors_in_transaction(
        ctx,
        inverted_index,
        current_open_transaction,
        vectors,
    )
    .await
    .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}

pub(crate) async fn upsert_sparse_idf_documents(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    documents: Vec<CreateSparseIdfDocumentDto>,
) -> Result<(), TransactionError> {
    let idf_inverted_index = ctx
        .ain_env
        .collections_map
        .get_idf_inverted_index(collection_id)
        .ok_or(TransactionError::CollectionNotFound)?;

    let current_open_transaction = unsafe {
        idf_inverted_index
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

    vectors::repo::upsert_sparse_idf_documents_in_transaction(
        idf_inverted_index,
        current_open_transaction,
        documents,
    )
    .await
    .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;

    Ok(())
}
