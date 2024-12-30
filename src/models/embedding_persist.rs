use std::{io::SeekFrom, sync::Arc};

use crate::indexes::inverted_index_types::RawSparseVectorEmbedding;

use super::{
    buffered_io::BufferManager, common::WaCustomError, types::RawVectorEmbedding, versioning::Hash,
};

pub struct EmbeddingOffset {
    pub version: Hash,
    pub offset: u32,
}

impl EmbeddingOffset {
    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(8);

        result.extend_from_slice(&self.version.to_le_bytes());
        result.extend_from_slice(&self.offset.to_le_bytes());

        result
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != 8 {
            return Err("Input must be exactly 8 bytes");
        }

        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let offset = u32::from_le_bytes(bytes[4..8].try_into().unwrap());

        Ok(Self {
            version: Hash::from(version),
            offset,
        })
    }
}

pub fn write_embedding(
    bufman: Arc<BufferManager>,
    emb: &RawVectorEmbedding,
) -> Result<u32, WaCustomError> {
    // TODO: select a better value for `N` (number of bytes to pre-allocate)
    let serialized = rkyv::to_bytes::<_, 256>(emb)
        .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

    let len = serialized.len() as u32;
    let cursor = bufman.open_cursor()?;

    let start = bufman.seek_with_cursor(cursor, SeekFrom::End(0))? as u32;
    bufman.write_u32_with_cursor(cursor, len)?;
    bufman.write_with_cursor(cursor, &serialized)?;

    bufman.close_cursor(cursor)?;

    Ok(start)
}

/// Writes a sparse embedding to a buffer and returns the next offset.
///
/// This function is responsible for writing a given sparse vector embedding into
/// a buffer managed by the `BufferManager`. After successfully writing the embedding,
/// it returns the updated offset, which indicates the position after the newly written
/// data in the buffer, to allow subsequent writes or reads.
///
/// # Arguments
///
/// * `bufman` - A reference-counted (`Arc`) `BufferManager` that manages the buffer
///   where the sparse embedding will be written. The `BufferManager` takes care of
///   resource management and memory access.
/// * `emb` - A reference to the `RawSparseVectorEmbedding` to be written into the buffer.
///   The embedding is expected to be in a raw, sparse vector format.
///
/// # Returns
///
/// This function returns a `Result` containing the following:
/// - `u32`: The next offset (in bytes) after the embedding is written, indicating
///   where the next data can be written in the buffer.
///
/// If an error occurs during the write operation, a `WaCustomError` is returned.
///
/// # Errors
///
/// - Returns a `WaCustomError` if the write operation fails, which could be caused by
///   issues such as insufficient space in the buffer, invalid embedding data, or other
///   internal errors.
pub fn write_sparse_embedding(
    bufman: Arc<BufferManager>,
    emb: &RawSparseVectorEmbedding,
) -> Result<u32, WaCustomError> {
    // TODO: select a better value for `N` (number of bytes to pre-allocate)
    let serialized = rkyv::to_bytes::<_, 256>(emb)
        .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

    let len = serialized.len() as u32;
    let cursor = bufman.open_cursor()?;

    let start = bufman.seek_with_cursor(cursor, SeekFrom::End(0))? as u32;
    bufman.write_u32_with_cursor(cursor, len)?;
    bufman.write_with_cursor(cursor, &serialized)?;

    bufman.close_cursor(cursor)?;

    Ok(start)
}

/// Reads a sparse embedding from a buffer and returns it along with the next offset.
///
/// This function is designed to read a sparse vector embedding from a specified
/// buffer managed by the `BufferManager`. It retrieves the embedding located at
/// a given offset and returns it as a `RawSparseVectorEmbedding`, along with the
/// updated offset for further processing.
///
/// # Arguments
///
/// * `bufman` - A reference-counted (`Arc`) `BufferManager` that provides access
///   to the buffer from which the embedding is read. The `BufferManager` is expected
///   to handle the underlying buffer resources and manage memory efficiently.
/// * `offset` - The byte offset (of type `u32`) from which the embedding data is
///   to be read. This is used to locate the sparse embedding in the buffer.
///
/// # Returns
///
/// This function returns a `Result` containing a tuple on success:
/// - `RawSparseVectorEmbedding` - The read sparse vector embedding.
/// - `u32` - The updated offset that can be used to read further data if necessary.
///
/// If the operation encounters an error, it will return a `WaCustomError`.
///
/// # Errors
///
/// - Returns a `WaCustomError` if reading from the buffer fails, or if the data at
///   the specified offset cannot be parsed correctly into a sparse embedding.
pub fn read_sparse_embedding(
    bufman: Arc<BufferManager>,
    offset: u32,
) -> Result<(RawSparseVectorEmbedding, u32), WaCustomError> {
    // TODO (Question)
    // should this function modified more to suit sparse vectors?
    // now we are reading and deserializing directly as RawSparseVectorEmbedding
    // would this cause an issue ?

    let cursor = bufman.open_cursor()?;

    bufman
        .seek_with_cursor(cursor, SeekFrom::Start(offset as u64))
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let len = bufman
        .read_u32_with_cursor(cursor)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let mut buf = vec![0; len as usize];

    bufman
        .read_with_cursor(cursor, &mut buf)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let emb: RawSparseVectorEmbedding =
        unsafe { rkyv::from_bytes_unchecked(&buf) }.map_err(|e| {
            WaCustomError::DeserializationError(format!(
                "Failed to deserialize VectorEmbedding: {}",
                e
            ))
        })?;

    let next = bufman
        .cursor_position(cursor)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))? as u32;

    bufman.close_cursor(cursor)?;

    Ok((emb, next))
}

pub fn read_embedding(
    bufman: Arc<BufferManager>,
    offset: u32,
) -> Result<(RawVectorEmbedding, u32), WaCustomError> {
    let cursor = bufman.open_cursor()?;

    bufman
        .seek_with_cursor(cursor, SeekFrom::Start(offset as u64))
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let len = bufman
        .read_u32_with_cursor(cursor)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let mut buf = vec![0; len as usize];

    bufman
        .read_with_cursor(cursor, &mut buf)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;

    let emb = unsafe { rkyv::from_bytes_unchecked(&buf) }.map_err(|e| {
        WaCustomError::DeserializationError(format!("Failed to deserialize VectorEmbedding: {}", e))
    })?;

    let next = bufman
        .cursor_position(cursor)
        .map_err(|e| WaCustomError::DeserializationError(e.to_string()))? as u32;

    bufman.close_cursor(cursor)?;

    Ok((emb, next))
}

#[cfg(test)]
mod tests {
    use super::{read_embedding, write_embedding, RawVectorEmbedding};
    use crate::models::{buffered_io::BufferManager, types::VectorId};
    use rand::{distributions::Uniform, rngs::ThreadRng, thread_rng, Rng};
    use std::sync::Arc;
    use tempfile::tempfile;

    fn get_random_embedding(rng: &mut ThreadRng) -> RawVectorEmbedding {
        let range = Uniform::new(-1.0, 1.0);

        let raw_vec: Vec<f32> = (0..rng.gen_range(100..200))
            .into_iter()
            .map(|_| rng.sample(&range))
            .collect();

        RawVectorEmbedding {
            raw_vec: Arc::new(raw_vec),
            hash_vec: VectorId(rng.gen()),
        }
    }

    #[test]
    fn test_embedding_serialization() {
        let mut rng = thread_rng();
        let embedding = get_random_embedding(&mut rng);
        let tempfile = tempfile().unwrap();

        let bufman = Arc::new(BufferManager::new(tempfile).unwrap());
        let offset = write_embedding(bufman.clone(), &embedding).unwrap();

        let (deserialized, _) = read_embedding(bufman.clone(), offset).unwrap();

        assert_eq!(embedding, deserialized);
    }

    #[test]
    fn test_multiple_embedding_serialization() {
        let mut rng = thread_rng();
        let embeddings: Vec<_> = (0..20).map(|_| get_random_embedding(&mut rng)).collect();
        let tempfile = tempfile().unwrap();

        let bufman = Arc::new(BufferManager::new(tempfile).unwrap());

        for embedding in &embeddings {
            write_embedding(bufman.clone(), embedding).unwrap();
        }

        let mut offset = 0;

        for embedding in embeddings {
            let (deserialized, next) = read_embedding(bufman.clone(), offset).unwrap();
            offset = next;

            assert_eq!(embedding, deserialized);
        }
    }
}
