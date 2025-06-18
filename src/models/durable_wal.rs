use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
};

use crate::{indexes::inverted::types::SparsePair, metadata::FieldValue};

use super::{
    collection::RawVectorEmbedding,
    types::{DocumentId, VectorId},
    versioning::VersionNumber,
    wal::VectorOp,
};

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

    /// Opens an existing WAL file for reading
    pub fn from_existing(root_path: &Path, version: VersionNumber) -> Result<Self, io::Error> {
        let file_path = root_path.join(format!("{}.wal", *version));
        let mut file = OpenOptions::new().read(true).open(&file_path)?;

        // Read vectors_count from header
        let mut count_buf = [0u8; 4];
        file.read_exact(&mut count_buf)?;
        let vectors_count = u32::from_le_bytes(count_buf);

        // Determine current write position
        let file_size = file.metadata()?.len() as u32;

        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            vectors_count: AtomicU32::new(vectors_count),
            write_position: AtomicU32::new(file_size),
        })
    }

    /// Gets the current vectors count
    pub fn vectors_count(&self) -> u32 {
        self.vectors_count.load(Ordering::Relaxed)
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

    /// Reads all operations from the WAL file
    pub fn read_all_ops(&self) -> Result<Vec<VectorOp>, io::Error> {
        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(4))?; // Skip header
        let mut ops = Vec::new();
        let mut buffer = Vec::new();

        // Read entire file after header
        file.read_to_end(&mut buffer)?;
        let mut cursor = 0;

        while cursor < buffer.len() {
            if cursor + 4 > buffer.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Incomplete WAL header",
                ));
            }

            let len_with_tag = u32::from_le_bytes([
                buffer[cursor],
                buffer[cursor + 1],
                buffer[cursor + 2],
                buffer[cursor + 3],
            ]);
            cursor += 4;

            let (len, is_delete) = if len_with_tag & (1u32 << 31) != 0 {
                (len_with_tag & !(1u32 << 31), true)
            } else {
                (len_with_tag, false)
            };

            let end_pos = cursor + len as usize;
            if end_pos > buffer.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Incomplete WAL record",
                ));
            }

            let record = &buffer[cursor..end_pos];
            cursor = end_pos;

            let op = if is_delete {
                let (id, _) = read_string_from_slice(record, 0)?;
                VectorOp::Delete(VectorId::from(id))
            } else {
                let vectors = decode_upsert(record)?;
                VectorOp::Upsert(vectors)
            };

            ops.push(op);
        }

        Ok(ops)
    }
}

// Helper functions for decoding
fn read_len_from_slice(data: &[u8], start: usize) -> Result<(u16, usize), io::Error> {
    if start >= data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "end of slice"));
    }

    let first = data[start];
    if first & 0x80 == 0 {
        Ok((first as u16, start + 1))
    } else {
        if start + 1 >= data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "end of slice"));
        }
        let msb = (first & 0x7F) as u16;
        let lsb = data[start + 1] as u16;
        Ok(((msb << 8) | lsb, start + 2))
    }
}

fn read_string_from_slice(data: &[u8], start: usize) -> Result<(String, usize), io::Error> {
    let (len, next_pos) = read_len_from_slice(data, start)?;
    let end_pos = next_pos + len as usize;

    if end_pos > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "end of slice"));
    }

    let s = String::from_utf8(data[next_pos..end_pos].to_vec())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok((s, end_pos))
}

fn read_opt_string_from_slice(
    data: &[u8],
    start: usize,
) -> Result<(Option<String>, usize), io::Error> {
    let (len, next_pos) = read_len_from_slice(data, start)?;
    if len == 0 {
        return Ok((None, next_pos));
    }

    let end_pos = next_pos + len as usize;
    if end_pos > data.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "end of slice"));
    }

    let s = String::from_utf8(data[next_pos..end_pos].to_vec())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok((Some(s), end_pos))
}

/// Decodes an upsert operation from a byte slice
fn decode_upsert(data: &[u8]) -> Result<Vec<RawVectorEmbedding>, io::Error> {
    let mut vectors = Vec::new();
    let mut cursor = 0;

    let (num_vectors, next_pos) = read_len_from_slice(data, cursor)?;
    cursor = next_pos;

    for _ in 0..num_vectors {
        let (id, next_pos) = read_string_from_slice(data, cursor)?;
        cursor = next_pos;
        let id = VectorId::from(id);

        let (document_id, next_pos) = match read_opt_string_from_slice(data, cursor)? {
            (Some(doc_id), pos) => (Some(DocumentId::from(doc_id)), pos),
            (None, pos) => (None, pos),
        };
        cursor = next_pos;

        // Decode dense values
        let (dense_len, next_pos) = read_len_from_slice(data, cursor)?;
        cursor = next_pos;
        let dense_values = if dense_len > 0 {
            let end_pos = cursor + dense_len as usize * 4;
            if end_pos > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "dense values"));
            }

            let mut values = Vec::with_capacity(dense_len as usize);
            for i in 0..dense_len as usize {
                let start = cursor + i * 4;
                values.push(f32::from_le_bytes([
                    data[start],
                    data[start + 1],
                    data[start + 2],
                    data[start + 3],
                ]));
            }
            cursor = end_pos;
            Some(values)
        } else {
            None
        };

        // Decode metadata
        let (metadata_len, next_pos) = read_len_from_slice(data, cursor)?;
        cursor = next_pos;
        let metadata = if metadata_len > 0 {
            let mut metadata = HashMap::with_capacity(metadata_len as usize);

            for _ in 0..metadata_len {
                let (field, next_pos) = read_string_from_slice(data, cursor)?;
                cursor = next_pos;

                if cursor >= data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "metadata variant",
                    ));
                }

                let variant = data[cursor];
                cursor += 1;

                let val = match variant {
                    0 => {
                        if cursor + 4 > data.len() {
                            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "int value"));
                        }
                        let val = i32::from_le_bytes([
                            data[cursor],
                            data[cursor + 1],
                            data[cursor + 2],
                            data[cursor + 3],
                        ]);
                        cursor += 4;
                        FieldValue::Int(val)
                    }
                    1 => {
                        let (str_val, next_pos) = read_string_from_slice(data, cursor)?;
                        cursor = next_pos;
                        FieldValue::String(str_val)
                    }
                    other => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Invalid metadata variant: {}", other),
                        ));
                    }
                };

                metadata.insert(field, val);
            }
            Some(metadata)
        } else {
            None
        };

        // Decode sparse values
        let (sparse_len, next_pos) = read_len_from_slice(data, cursor)?;
        cursor = next_pos;
        let sparse_values = if sparse_len > 0 {
            let mut sparse_values = Vec::with_capacity(sparse_len as usize);
            for _ in 0..sparse_len {
                if cursor + 8 > data.len() {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "sparse pair"));
                }

                let index = u32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]);
                let value = f32::from_le_bytes([
                    data[cursor + 4],
                    data[cursor + 5],
                    data[cursor + 6],
                    data[cursor + 7],
                ]);
                cursor += 8;

                sparse_values.push(SparsePair(index, value));
            }
            Some(sparse_values)
        } else {
            None
        };

        // Decode text
        let (text, next_pos) = read_opt_string_from_slice(data, cursor)?;
        cursor = next_pos;

        vectors.push(RawVectorEmbedding {
            id,
            document_id,
            dense_values,
            metadata,
            sparse_values,
            text,
        });
    }

    Ok(vectors)
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

    #[test]
    fn test_random_upsert_persistence() {
        let dir = tempdir().unwrap();
        let version = 0;

        let vectors: Vec<_> = (0..3).map(|_| random_vector()).collect();

        let wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
        wal.append(VectorOp::Upsert(vectors.clone())).unwrap();
        wal.flush().unwrap();

        let wal = DurableWALFile::from_existing(dir.path(), VersionNumber::from(version)).unwrap();
        let ops = wal.read_all_ops().unwrap();
        match &ops[..] {
            [VectorOp::Upsert(read_vecs)] => {
                assert_eq!(read_vecs.len(), 3);
            }
            _ => panic!("Expected single VectorOp::Upsert"),
        }
    }

    #[test]
    fn test_random_delete_persistence() {
        let dir = tempdir().unwrap();
        let version = 0;
        let id = VectorId::from(random_string(10));

        let wal = DurableWALFile::new(dir.path(), VersionNumber::from(version)).unwrap();
        wal.append(VectorOp::Delete(id.clone())).unwrap();
        wal.flush().unwrap();

        let wal = DurableWALFile::from_existing(dir.path(), VersionNumber::from(version)).unwrap();
        let ops = wal.read_all_ops().unwrap();
        match &ops[..] {
            [VectorOp::Delete(read_id)] => assert_eq!(read_id, &id),
            _ => panic!("Expected VectorOp::Delete"),
        }
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

        let wal = DurableWALFile::from_existing(dir.path(), VersionNumber::from(version)).unwrap();
        let ops = wal.read_all_ops().unwrap();
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

        let wal = DurableWALFile::from_existing(dir.path(), VersionNumber::from(version)).unwrap();
        let ops = wal.read_all_ops().unwrap();
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
        let wal = DurableWALFile::from_existing(dir.path(), VersionNumber::from(version));
        assert!(wal.is_ok());
    }
}
