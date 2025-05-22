use std::{
    collections::HashMap,
    fs::OpenOptions,
    io,
    path::Path,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use parking_lot::Mutex;

use crate::{indexes::inverted::types::SparsePair, metadata::FieldValue};

use super::{
    buffered_io::{BufIoError, BufferManager},
    collection::RawVectorEmbedding,
    serializer::{read_len, read_opt_string, read_string, write_len},
    types::{DocumentId, VectorId},
    versioning::VersionNumber,
};

#[derive(Debug, Clone)]
pub enum VectorOp {
    Upsert(Vec<RawVectorEmbedding>),
    Delete(VectorId),
}

pub struct WALFile {
    bufman: BufferManager,
    cursor: u64,
    read_lock: Mutex<()>,
    vectors_count: AtomicU32,
}

impl WALFile {
    pub fn new(root_path: &Path, version: VersionNumber) -> Result<Self, BufIoError> {
        let file_path: Arc<Path> = root_path.join(format!("{}.wal", *version)).into();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)?;

        let bufman = BufferManager::new(file, 8192)?;
        let cursor = bufman.open_cursor()?;
        let vectors_count = bufman.read_u32_with_cursor(cursor)?;
        bufman.seek_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, vectors_count)?;

        Ok(Self {
            bufman,
            cursor,
            read_lock: Mutex::new(()),
            vectors_count: AtomicU32::new(vectors_count),
        })
    }

    pub fn vectors_count(&self) -> u32 {
        self.vectors_count.load(Ordering::Relaxed)
    }

    pub fn flush(&self) -> Result<(), BufIoError> {
        let cursor = self.bufman.open_cursor()?;
        self.bufman
            .update_u32_with_cursor(cursor, self.vectors_count.load(Ordering::Acquire))?;
        self.bufman.close_cursor(cursor)?;
        self.bufman.flush()
    }

    pub fn append(&self, op: VectorOp) -> Result<(), BufIoError> {
        let mut buf = Vec::new();

        buf.extend([u8::MAX; 4]);

        match op {
            VectorOp::Upsert(vectors) => {
                self.vectors_count
                    .fetch_add(vectors.len() as u32, Ordering::Relaxed);
                write_len(&mut buf, vectors.len() as u16);
                for vector in vectors {
                    write_len(&mut buf, vector.id.len() as u16);
                    buf.extend(vector.id.as_bytes());
                    if let Some(document_id) = &vector.document_id {
                        write_len(&mut buf, document_id.len() as u16);
                        buf.extend(document_id.as_bytes());
                    } else {
                        write_len(&mut buf, 0);
                    }
                    if let Some(dense_values) = &vector.dense_values {
                        write_len(&mut buf, dense_values.len() as u16);
                        for val in dense_values {
                            buf.extend(val.to_le_bytes());
                        }
                    } else {
                        write_len(&mut buf, 0);
                    }
                    if let Some(metadata) = &vector.metadata {
                        write_len(&mut buf, metadata.len() as u16);
                        for (field, val) in metadata {
                            write_len(&mut buf, field.len() as u16);
                            buf.extend(field.as_bytes());

                            match val {
                                FieldValue::Int(int) => {
                                    buf.push(0);
                                    buf.extend(int.to_le_bytes());
                                }
                                FieldValue::String(str) => {
                                    buf.push(1);
                                    write_len(&mut buf, str.len() as u16);
                                    buf.extend(str.as_bytes());
                                }
                            }
                        }
                    } else {
                        write_len(&mut buf, 0);
                    }

                    if let Some(sparse_values) = &vector.sparse_values {
                        write_len(&mut buf, sparse_values.len() as u16);
                        for pair in sparse_values {
                            buf.extend(pair.0.to_le_bytes());
                            buf.extend(pair.1.to_le_bytes());
                        }
                    } else {
                        write_len(&mut buf, 0);
                    }

                    if let Some(text) = &vector.text {
                        write_len(&mut buf, text.len() as u16);
                        buf.extend(text.as_bytes());
                    } else {
                        write_len(&mut buf, 0);
                    }
                }
                let len = buf.len() as u32 - 4;
                buf[0..4].copy_from_slice(&len.to_le_bytes());
            }
            VectorOp::Delete(id) => {
                write_len(&mut buf, id.len() as u16);
                buf.extend(id.as_bytes());
                let len = buf.len() as u32 - 4;
                buf[0..4].copy_from_slice(&(len | (1u32 << 31)).to_le_bytes());
            }
        }

        self.bufman.write_to_end_of_file(self.cursor, &buf)?;
        Ok(())
    }

    pub fn read(&self) -> Result<Option<VectorOp>, BufIoError> {
        let guard = self.read_lock.lock();

        let cursor_pos = self.bufman.cursor_position(self.cursor)?;
        let file_size = self.bufman.file_size();

        if cursor_pos >= file_size {
            return Ok(None);
        }

        let len_with_tag = self.bufman.read_u32_with_cursor(self.cursor)?;
        let (len, is_delete) = if len_with_tag & (1u32 << 31) != 0 {
            (0x7FFFFFFF & len_with_tag, true)
        } else {
            (len_with_tag, false)
        };
        self.bufman
            .seek_with_cursor(self.cursor, len as u64 + 4 + cursor_pos)?;

        drop(guard);

        let cursor = self.bufman.open_cursor()?;
        self.bufman.seek_with_cursor(cursor, cursor_pos + 4)?;
        let op = if is_delete {
            let id = read_string(&self.bufman, cursor)?;
            VectorOp::Delete(VectorId::from(id))
        } else {
            let len = read_len(&self.bufman, cursor)? as usize;
            let mut vectors = Vec::with_capacity(len);

            for _ in 0..len {
                let id = VectorId::from(read_string(&self.bufman, cursor)?);
                let document_id = read_opt_string(&self.bufman, cursor)?.map(DocumentId::from);
                let dense_values_len = read_len(&self.bufman, cursor)? as usize;
                let dense_values = if dense_values_len == 0 {
                    None
                } else {
                    let mut values = Vec::with_capacity(dense_values_len);
                    for _ in 0..dense_values_len {
                        values.push(self.bufman.read_f32_with_cursor(cursor)?);
                    }
                    Some(values)
                };
                let metadata_len = read_len(&self.bufman, cursor)? as usize;
                let metadata = if metadata_len == 0 {
                    None
                } else {
                    let mut metadata = HashMap::with_capacity(metadata_len);

                    for _ in 0..metadata_len {
                        let field = read_string(&self.bufman, cursor)?;
                        let val = {
                            let variant = self.bufman.read_u8_with_cursor(cursor)?;
                            match variant {
                                0 => FieldValue::Int(self.bufman.read_i32_with_cursor(cursor)?),
                                1 => FieldValue::String(read_string(&self.bufman, cursor)?),
                                other => {
                                    return Err(BufIoError::Io(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        format!("Invalid FieldValue variant `{}`", other),
                                    )))
                                }
                            }
                        };
                        metadata.insert(field, val);
                    }

                    Some(metadata)
                };

                let sparse_values_len = read_len(&self.bufman, cursor)? as usize;
                let sparse_values = if sparse_values_len == 0 {
                    None
                } else {
                    let mut sparse_values = Vec::with_capacity(sparse_values_len);

                    for _ in 0..sparse_values_len {
                        let index = self.bufman.read_u32_with_cursor(cursor)?;
                        let value = self.bufman.read_f32_with_cursor(cursor)?;
                        let pair = SparsePair(index, value);
                        sparse_values.push(pair);
                    }

                    Some(sparse_values)
                };

                let text = read_opt_string(&self.bufman, cursor)?;

                let vector = RawVectorEmbedding {
                    id,
                    document_id,
                    dense_values,
                    metadata,
                    sparse_values,
                    text,
                };
                vectors.push(vector);
            }

            VectorOp::Upsert(vectors)
        };

        Ok(Some(op))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexes::inverted::types::SparsePair;
    use crate::metadata::FieldValue;
    use rand::{distributions::Alphanumeric, thread_rng, Rng};
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn random_string(len: usize) -> String {
        thread_rng()
            .sample_iter(&Alphanumeric)
            .take(len)
            .map(char::from)
            .collect()
    }

    fn random_vector() -> RawVectorEmbedding {
        let mut rng = thread_rng();
        let dense_len = rng.gen_range(1..8);
        let metadata_len = rng.gen_range(1..4);
        let sparse_len = rng.gen_range(0..4);

        let mut metadata = HashMap::new();
        for _ in 0..metadata_len {
            let key = random_string(5);
            let val = if rng.gen_bool(0.5) {
                FieldValue::Int(rng.gen_range(-1000..1000))
            } else {
                FieldValue::String(random_string(6))
            };
            metadata.insert(key, val);
        }

        RawVectorEmbedding {
            id: random_string(8).into(),
            document_id: Some(random_string(12).into()),
            dense_values: Some((0..dense_len).map(|_| rng.gen()).collect()),
            metadata: Some(metadata),
            sparse_values: Some(
                (0..sparse_len)
                    .map(|_| SparsePair(rng.gen(), rng.gen()))
                    .collect(),
            ),
            text: Some(random_string(16)),
        }
    }

    fn reopen_wal(dir: &std::path::Path, version: u32) -> WALFile {
        WALFile::new(dir, VersionNumber::from(version)).unwrap()
    }

    #[test]
    fn test_random_upsert_persistence() {
        let dir = tempdir().unwrap();
        let version = 0;

        {
            let wal = WALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
            let vectors: Vec<_> = (0..3).map(|_| random_vector()).collect();
            wal.append(VectorOp::Upsert(vectors.clone())).unwrap();
            wal.flush().unwrap();
        }

        {
            let wal = reopen_wal(dir.path(), version);
            let result = wal.read().unwrap();
            match result {
                Some(VectorOp::Upsert(read_vecs)) => {
                    assert_eq!(read_vecs.len(), 3);
                }
                _ => panic!("Expected VectorOp::Upsert"),
            }
        }
    }

    #[test]
    fn test_random_delete_persistence() {
        let dir = tempdir().unwrap();
        let version = 0;
        let id = VectorId::from(random_string(10));

        {
            let wal = WALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
            wal.append(VectorOp::Delete(id.clone())).unwrap();
            wal.flush().unwrap();
        }

        {
            let wal = reopen_wal(dir.path(), version);
            match wal.read().unwrap() {
                Some(VectorOp::Delete(read_id)) => assert_eq!(read_id, id),
                _ => panic!("Expected VectorOp::Delete"),
            }
        }
    }

    #[test]
    fn test_mixed_ops_persistence() {
        let dir = tempdir().unwrap();
        let version = 0;

        let vecs: Vec<_> = (0..2).map(|_| random_vector()).collect();
        let del_id = VectorId::from(random_string(10));

        {
            let wal = WALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
            wal.append(VectorOp::Upsert(vecs.clone())).unwrap();
            wal.append(VectorOp::Delete(del_id.clone())).unwrap();
            wal.flush().unwrap();
        }

        {
            let wal = reopen_wal(dir.path(), version);
            match wal.read().unwrap() {
                Some(VectorOp::Upsert(read_vecs)) => assert_eq!(read_vecs.len(), 2),
                _ => panic!("Expected VectorOp::Upsert"),
            }
            match wal.read().unwrap() {
                Some(VectorOp::Delete(read_id)) => assert_eq!(read_id, del_id),
                _ => panic!("Expected VectorOp::Delete"),
            }
        }
    }

    #[test]
    fn test_multiple_consecutive_ops() {
        let dir = tempdir().unwrap();
        let version = 0;

        let entries: Vec<VectorOp> = (0..10)
            .map(|i| {
                if i % 2 == 0 {
                    VectorOp::Upsert(vec![random_vector()])
                } else {
                    VectorOp::Delete(VectorId::from(random_string(8)))
                }
            })
            .collect();

        {
            let wal = WALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
            for op in &entries {
                wal.append(op.clone()).unwrap();
            }
            wal.flush().unwrap();
        }

        {
            let wal = reopen_wal(dir.path(), version);
            for expected in &entries {
                let read = wal.read().unwrap().expect("Expected some op");
                match (expected, &read) {
                    (VectorOp::Upsert(ev), VectorOp::Upsert(rv)) => assert_eq!(ev[0].id, rv[0].id),
                    (VectorOp::Delete(eid), VectorOp::Delete(rid)) => assert_eq!(eid, rid),
                    _ => panic!("Mismatched operation types"),
                }
            }
        }
    }
}
