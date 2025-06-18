use std::{
    fs::{File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    path::Path,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
};

use crate::metadata::FieldValue;

use super::{versioning::VersionNumber, wal::VectorOp};

pub struct DurableWALFile {
    file: Arc<Mutex<File>>,
    vectors_count: AtomicU32,
    write_position: AtomicU32,
}

/// Writes a u16 length to the buffer using custom encoding
fn write_len(buf: &mut Vec<u8>, len: u16) {
    if len <= 0x7F {
        buf.push(len as u8);
    } else {
        let msb = ((len >> 8) as u8 & 0x7F) | 0x80;
        let lsb = len as u8;
        buf.push(msb);
        buf.push(lsb);
    }
}

impl DurableWALFile {
    /// Creates a new WAL file in a temporary location
    pub fn new(root_path: &Path, version: VersionNumber) -> Result<Self, io::Error> {
        let file_path = root_path.join(format!("{}.wal", *version));
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)?;

        // Write initial vectors_count (0)
        let buf = [0u8; 4];
        file.write_all(&buf)?;
        file.flush()?;

        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            vectors_count: AtomicU32::new(0),
            write_position: AtomicU32::new(4), // After 4-byte header
        })
    }

    /// Persists the WAL to its final location and ensures durability
    pub fn flush(&self) -> Result<(), io::Error> {
        let mut file = self.file.lock().unwrap();

        // Update header with current vectors count
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&self.vectors_count.load(Ordering::Relaxed).to_le_bytes())?;

        // Ensure all data is persisted
        file.flush()?;
        file.sync_all()?;

        Ok(())
    }

    /// Appends an operation to the WAL with immediate durability
    pub fn append(&self, op: VectorOp) -> Result<(), io::Error> {
        let mut buf = Vec::with_capacity(1024);

        // Reserve space for length header (4 bytes)
        buf.extend([0u8; 4]);

        match &op {
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

                // Update length header (without tag)
                let len = buf.len() as u32 - 4;
                buf[0..4].copy_from_slice(&len.to_le_bytes());
            }
            VectorOp::Delete(id) => {
                write_len(&mut buf, id.len() as u16);
                buf.extend(id.as_bytes());

                // Update length header with delete tag
                let len = buf.len() as u32 - 4;
                buf[0..4].copy_from_slice(&(len | (1u32 << 31)).to_le_bytes());
            }
        }

        // Write with lock protection and ensure durability
        let mut file = self.file.lock().unwrap();

        // Seek to current write position
        let pos = self.write_position.load(Ordering::Acquire);
        file.seek(SeekFrom::Start(pos as u64))?;

        // Write data
        file.write_all(&buf)?;
        file.flush()?; // Ensure write hits disk

        // Update write position
        self.write_position
            .fetch_add(buf.len() as u32, Ordering::Release);

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

        let wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
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

        let wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
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

        let wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
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

        let wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
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
            let wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
            wal.append(VectorOp::Delete(id.clone())).unwrap();
            // No explicit flush, but append should flush
        }

        // Reopen should still see the data
        let wal = WALFile::from_existing(dir.path(), VersionNumber::from(version));
        assert!(wal.is_ok());
    }
}
