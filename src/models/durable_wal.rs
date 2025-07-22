use std::{fs::OpenOptions, path::Path, sync::Arc};

use crate::metadata::FieldValue;

use super::{
    buffered_io::{BufIoError, BufferManager},
    serializer::write_len,
    versioning::VersionNumber,
    wal::VectorOp,
};

pub struct DurableWALFile {
    bufman: BufferManager,
    cursor: u64,
    records_upserted: u32,
    records_deleted: u32,
    total_operations: u32,
}

impl DurableWALFile {
    pub fn new(root_path: &Path, version: VersionNumber) -> Result<Self, BufIoError> {
        let file_path: Arc<Path> = root_path.join(format!("{}.wal", *version)).into();

        let file = OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .truncate(false)
            .open(&file_path)?;
        let bufman = BufferManager::new(file, 8192)?;
        let cursor = bufman.open_cursor()?;
        bufman.update_u32_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, 0)?;

        Ok(Self {
            bufman,
            cursor,
            records_upserted: 0,
            records_deleted: 0,
            total_operations: 0,
        })
    }

    pub fn records_upserted(&self) -> u32 {
        self.records_upserted
    }

    pub fn records_deleted(&self) -> u32 {
        self.records_deleted
    }

    pub fn total_operations(&self) -> u32 {
        self.total_operations
    }

    pub fn flush(self) -> Result<(), BufIoError> {
        let cursor = self.bufman.open_cursor()?;
        self.bufman
            .update_u32_with_cursor(cursor, self.records_upserted)?;
        self.bufman
            .update_u32_with_cursor(cursor, self.records_deleted)?;
        self.bufman
            .update_u32_with_cursor(cursor, self.total_operations)?;
        self.bufman.close_cursor(cursor)?;
        self.bufman.flush()
    }

    pub fn append(&mut self, op: VectorOp) -> Result<(), BufIoError> {
        let mut buf = Vec::new();

        buf.extend([u8::MAX; 4]);

        match op {
            VectorOp::Upsert(vectors) => {
                self.records_upserted += vectors.len() as u32;
                write_len(&mut buf, vectors.len() as u32);
                for vector in &*vectors {
                    write_len(&mut buf, vector.id.len() as u32);
                    buf.extend(vector.id.as_bytes());
                    if let Some(document_id) = &vector.document_id {
                        write_len(&mut buf, document_id.len() as u32);
                        buf.extend(document_id.as_bytes());
                    } else {
                        write_len(&mut buf, 0);
                    }
                    if let Some(dense_values) = &vector.dense_values {
                        write_len(&mut buf, dense_values.len() as u32);
                        for val in dense_values {
                            buf.extend(val.to_le_bytes());
                        }
                    } else {
                        write_len(&mut buf, 0);
                    }
                    if let Some(metadata) = &vector.metadata {
                        write_len(&mut buf, metadata.len() as u32);
                        for (field, val) in metadata {
                            write_len(&mut buf, field.len() as u32);
                            buf.extend(field.as_bytes());

                            match val {
                                FieldValue::Int(int) => {
                                    buf.push(0);
                                    buf.extend(int.to_le_bytes());
                                }
                                FieldValue::String(str) => {
                                    buf.push(1);
                                    write_len(&mut buf, str.len() as u32);
                                    buf.extend(str.as_bytes());
                                }
                            }
                        }
                    } else {
                        write_len(&mut buf, 0);
                    }

                    if let Some(sparse_values) = &vector.sparse_values {
                        write_len(&mut buf, sparse_values.len() as u32);
                        for pair in sparse_values {
                            buf.extend(pair.0.to_le_bytes());
                            buf.extend(pair.1.to_le_bytes());
                        }
                    } else {
                        write_len(&mut buf, 0);
                    }

                    if let Some(text) = &vector.text {
                        write_len(&mut buf, text.len() as u32);
                        buf.extend(text.as_bytes());
                    } else {
                        write_len(&mut buf, 0);
                    }
                }
                let len = buf.len() as u32 - 4;
                buf[0..4].copy_from_slice(&len.to_le_bytes());
            }
            VectorOp::Delete(id) => {
                write_len(&mut buf, id.len() as u32);
                buf.extend(id.as_bytes());
                let len = buf.len() as u32 - 4;
                buf[0..4].copy_from_slice(&(len | (1u32 << 31)).to_le_bytes());
            }
        }

        self.bufman.write_to_end_of_file(self.cursor, &buf)?;
        let cursor = self.bufman.open_cursor()?;
        self.bufman
            .update_u32_with_cursor(cursor, self.records_upserted)?;
        self.bufman
            .update_u32_with_cursor(cursor, self.records_deleted)?;
        self.bufman
            .update_u32_with_cursor(cursor, self.total_operations)?;
        self.bufman.close_cursor(cursor)?;
        self.bufman.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::FieldValue;
    use crate::models::collection::RawVectorEmbedding;
    use crate::models::types::VectorId;
    use crate::{indexes::inverted::types::SparsePair, models::wal::WALFile};
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

    #[test]
    fn test_random_upsert_persistence() {
        let dir = tempdir().unwrap();
        let version = 0;

        let vectors: Vec<_> = (0..3).map(|_| random_vector()).collect();

        let mut wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
        wal.append(VectorOp::Upsert(vectors.clone())).unwrap();
        wal.flush().unwrap();

        let wal = WALFile::from_existing(dir.path(), VersionNumber::from(version)).unwrap();
        let op = wal.read().unwrap().unwrap();
        match &op {
            VectorOp::Upsert(read_vecs) => {
                assert_eq!(read_vecs.len(), 3);
            }
            _ => panic!("Expected single VectorOp::Upsert"),
        }
        assert!(wal.read().unwrap().is_none());
    }

    #[test]
    fn test_random_delete_persistence() {
        let dir = tempdir().unwrap();
        let version = 0;
        let id = VectorId::from(random_string(10));

        let mut wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
        wal.append(VectorOp::Delete(id.clone())).unwrap();
        wal.flush().unwrap();

        let wal = WALFile::from_existing(dir.path(), VersionNumber::from(version)).unwrap();
        let op = wal.read().unwrap().unwrap();
        match op {
            VectorOp::Delete(read_id) => assert_eq!(read_id, id),
            _ => panic!("Expected VectorOp::Delete"),
        }
        assert!(wal.read().unwrap().is_none());
    }

    #[test]
    fn test_mixed_ops_persistence() {
        let dir = tempdir().unwrap();
        let version = 0;

        let vecs: Vec<_> = (0..2).map(|_| random_vector()).collect();
        let del_id = VectorId::from(random_string(10));

        let mut wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
        wal.append(VectorOp::Upsert(vecs.clone())).unwrap();
        wal.append(VectorOp::Delete(del_id.clone())).unwrap();
        wal.flush().unwrap();

        let wal = WALFile::from_existing(dir.path(), VersionNumber::from(version)).unwrap();
        let mut ops = Vec::new();
        while let Some(op) = wal.read().unwrap() {
            ops.push(op);
        }
        assert_eq!(ops.len(), 2);

        match &ops[0] {
            VectorOp::Upsert(read_vecs) => assert_eq!(read_vecs.len(), 2),
            _ => panic!("Expected first op to be Upsert"),
        }

        match &ops[1] {
            VectorOp::Delete(read_id) => assert_eq!(read_id, &del_id),
            _ => panic!("Expected second op to be Delete"),
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

        let mut wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
        for op in &entries {
            wal.append(op.clone()).unwrap();
        }
        wal.flush().unwrap();

        let wal = WALFile::from_existing(dir.path(), VersionNumber::from(version)).unwrap();
        let mut ops = Vec::new();
        while let Some(op) = wal.read().unwrap() {
            ops.push(op);
        }
        assert_eq!(ops.len(), entries.len());

        for (expected, actual) in entries.iter().zip(ops.iter()) {
            match (expected, actual) {
                (VectorOp::Upsert(ev), VectorOp::Upsert(rv)) => assert_eq!(ev[0].id, rv[0].id),
                (VectorOp::Delete(eid), VectorOp::Delete(rid)) => assert_eq!(eid, rid),
                _ => panic!("Mismatched operation types"),
            }
        }
    }

    #[test]
    fn test_durability() {
        let dir = tempdir().unwrap();
        let version = 0;
        let id = VectorId::from("test_id".to_string());

        {
            let mut wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
            wal.append(VectorOp::Delete(id.clone())).unwrap();
            // No explicit flush, but append should flush
        }

        // Reopen should still see the data
        let wal = WALFile::from_existing(dir.path(), VersionNumber::from(version));
        assert!(wal.is_ok());
    }
}
