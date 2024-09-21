use super::{dtos::CreateTransactionResponseDto, error::TransactionError, repo};

pub(crate) async fn create_transaction(
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    repo::create_transaction(collection_id).await
}

pub(crate) async fn abort_transaction(
    collection_id: &str,
    transaction_id: &str,
) -> Result<(), TransactionError> {
    repo::abort_transaction(collection_id, transaction_id).await
}
