use std::sync::Arc;

use crate::indexes::hnsw::types::RawDenseVectorEmbedding;

use super::{buffered_io::BufferManager, common::WaCustomError, versioning::Hash};

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

pub fn write_dense_embedding(
    bufman: &BufferManager,
    emb: &RawDenseVectorEmbedding,
) -> Result<u32, WaCustomError> {
    // TODO: select a better value for `N` (number of bytes to pre-allocate)
    let serialized = rkyv::to_bytes::<_, 256>(emb)
        .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

    let len = serialized.len() as u32;
    let cursor = bufman.open_cursor()?;

    let mut buf = Vec::with_capacity(4 + serialized.len());
    buf.extend(len.to_le_bytes());
    buf.extend(&*serialized);

    let start = bufman.write_to_end_of_file(cursor, &buf)? as u32;

    bufman.close_cursor(cursor)?;

    Ok(start)
}

pub fn read_embedding(
    bufman: Arc<BufferManager>,
    offset: u32,
) -> Result<(RawDenseVectorEmbedding, u32), WaCustomError> {
    let cursor = bufman.open_cursor()?;

    bufman
        .seek_with_cursor(cursor, offset as u64)
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
    use super::{read_embedding, write_dense_embedding, RawDenseVectorEmbedding};
    use crate::{
        metadata,
        models::{buffered_io::BufferManager, types::VectorId},
    };
    use rand::{distributions::Uniform, rngs::ThreadRng, thread_rng, Rng};
    use std::{collections::HashMap, sync::Arc};
    use tempfile::tempfile;

    fn get_random_embedding(rng: &mut ThreadRng) -> RawDenseVectorEmbedding {
        let range = Uniform::new(-1.0, 1.0);

        let raw_vec: Vec<f32> = (0..rng.gen_range(100..200))
            .map(|_| rng.sample(range))
            .collect();

        RawDenseVectorEmbedding {
            raw_vec: Arc::new(raw_vec),
            hash_vec: VectorId(rng.gen()),
            raw_metadata: None,
            is_pseudo: false,
        }
    }

    #[test]
    fn test_embedding_serialization() {
        let mut rng = thread_rng();
        let embedding = get_random_embedding(&mut rng);
        let tempfile = tempfile().unwrap();

        let bufman = Arc::new(BufferManager::new(tempfile, 8192).unwrap());
        let offset = write_dense_embedding(&bufman, &embedding).unwrap();

        let (deserialized, _) = read_embedding(bufman.clone(), offset).unwrap();

        assert_eq!(embedding, deserialized);
    }

    #[test]
    fn test_multiple_embedding_serialization() {
        let mut rng = thread_rng();
        let embeddings: Vec<_> = (0..20).map(|_| get_random_embedding(&mut rng)).collect();
        let tempfile = tempfile().unwrap();

        let bufman = Arc::new(BufferManager::new(tempfile, 8192).unwrap());

        for embedding in &embeddings {
            write_dense_embedding(&bufman, embedding).unwrap();
        }

        let mut offset = 0;

        for embedding in embeddings {
            let (deserialized, next) = read_embedding(bufman.clone(), offset).unwrap();
            offset = next;

            assert_eq!(embedding, deserialized);
        }
    }

    #[test]
    fn test_embedding_with_metadata_serialization() {
        let mut rng = thread_rng();
        let mut embedding = get_random_embedding(&mut rng);
        let mut md: metadata::MetadataFields = HashMap::with_capacity(2);
        md.insert(
            "city".to_owned(),
            metadata::FieldValue::String("Bangalore".to_owned()),
        );
        md.insert("limit".to_owned(), metadata::FieldValue::Int(100));
        embedding.raw_metadata = Some(md);

        let tempfile = tempfile().unwrap();
        let bufman = Arc::new(BufferManager::new(tempfile, 8192).unwrap());
        let offset = write_dense_embedding(&bufman, &embedding).unwrap();
        let (deserialized, _) = read_embedding(bufman.clone(), offset).unwrap();
        assert_eq!(embedding, deserialized);
    }
}
