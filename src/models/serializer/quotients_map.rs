use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, RwLock,
};

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    common::TSHashTable,
    tf_idf_index::UnsafeVersionedVec,
    tree_map::{Quotient, QuotientVec, QuotientsMap, QuotientsMapVec, UnsafeVersionedItem},
    types::FileOffset,
};

use super::SimpleSerialize;

const CHUNK_SIZE: usize = 4;

impl<T: SimpleSerialize> SimpleSerialize for QuotientsMap<T> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let mut list = self.map.to_list();
        list.sort_unstable_by_key(|(_, q)| q.sequence_idx);
        if list.is_empty() {
            return Ok(u32::MAX);
        }
        let len = self.len.load(Ordering::Relaxed);
        let offset_read_guard = self.offset.read().map_err(|_| BufIoError::Locking)?;
        let total_chunks = list.len().div_ceil(CHUNK_SIZE);
        if let Some(offset) = *offset_read_guard {
            let serialized_upto = self.serialized_upto.load(Ordering::Relaxed);
            bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            bufman.update_u64_with_cursor(cursor, len)?;

            let total_chunks_serialized = serialized_upto.div_ceil(CHUNK_SIZE);

            for i in 0..CHUNK_SIZE {
                let Some((k, v)) = list.get(i) else {
                    bufman.update_with_cursor(cursor, &[u8::MAX; 12])?;
                    continue;
                };
                let value_offset = v.value.serialize(bufman, cursor)?;
                bufman.seek_with_cursor(cursor, offset.0 as u64 + 8 + (i as u64 * 12))?;
                bufman.update_u64_with_cursor(cursor, *k)?;
                bufman.update_u32_with_cursor(cursor, value_offset)?;
            }

            let mut prev_chunk_offset = offset.0 as u64 + 8;

            for chunk_idx in 1..total_chunks_serialized {
                let current_chunk_offset = bufman.read_u32_with_cursor(cursor)?;
                bufman.seek_with_cursor(cursor, current_chunk_offset as u64)?;
                for i in 0..CHUNK_SIZE {
                    let Some((k, v)) = list.get(i + (chunk_idx * CHUNK_SIZE)) else {
                        bufman.update_with_cursor(cursor, &[u8::MAX; 12])?;
                        continue;
                    };
                    let value_offset = v.value.serialize(bufman, cursor)?;
                    bufman
                        .seek_with_cursor(cursor, current_chunk_offset as u64 + (i as u64 * 12))?;
                    bufman.update_u64_with_cursor(cursor, *k)?;
                    bufman.update_u32_with_cursor(cursor, value_offset)?;
                }

                prev_chunk_offset = current_chunk_offset as u64;
            }

            for chunk_idx in total_chunks_serialized..total_chunks {
                let mut buf = Vec::with_capacity(CHUNK_SIZE * 12 + 4);
                for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                    let Some((k, v)) = list.get(i) else {
                        buf.extend([u8::MAX; 12]);
                        continue;
                    };
                    buf.extend(k.to_le_bytes());
                    let offset = v.value.serialize(bufman, cursor)?;
                    buf.extend(offset.to_le_bytes());
                }
                buf.extend([u8::MAX; 4]);
                let chunk_offset = bufman.write_to_end_of_file(cursor, &buf)?;
                bufman.seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 12))?;
                bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
                prev_chunk_offset = chunk_offset;
            }

            self.serialized_upto.store(list.len(), Ordering::Relaxed);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.offset.write().map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *offset_write_guard {
            let serialized_upto = self.serialized_upto.load(Ordering::Relaxed);
            bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            bufman.update_u64_with_cursor(cursor, len)?;

            let total_chunks_serialized = serialized_upto.div_ceil(CHUNK_SIZE);

            for i in 0..CHUNK_SIZE {
                let Some((k, v)) = list.get(i) else {
                    bufman.update_with_cursor(cursor, &[u8::MAX; 12])?;
                    continue;
                };
                let value_offset = v.value.serialize(bufman, cursor)?;
                bufman.seek_with_cursor(cursor, offset.0 as u64 + 8 + (i as u64 * 12))?;
                bufman.update_u64_with_cursor(cursor, *k)?;
                bufman.update_u32_with_cursor(cursor, value_offset)?;
            }

            let mut prev_chunk_offset = offset.0 as u64 + 8;

            for chunk_idx in 1..total_chunks_serialized {
                let current_chunk_offset = bufman.read_u32_with_cursor(cursor)?;
                bufman.seek_with_cursor(cursor, current_chunk_offset as u64)?;
                for i in 0..CHUNK_SIZE {
                    let Some((k, v)) = list.get(i + (chunk_idx * CHUNK_SIZE)) else {
                        bufman.update_with_cursor(cursor, &[u8::MAX; 12])?;
                        continue;
                    };
                    let value_offset = v.value.serialize(bufman, cursor)?;
                    bufman
                        .seek_with_cursor(cursor, current_chunk_offset as u64 + (i as u64 * 12))?;
                    bufman.update_u64_with_cursor(cursor, *k)?;
                    bufman.update_u32_with_cursor(cursor, value_offset)?;
                }

                prev_chunk_offset = current_chunk_offset as u64;
            }

            for chunk_idx in total_chunks_serialized..total_chunks {
                let mut buf = Vec::with_capacity(CHUNK_SIZE * 12 + 4);
                for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                    let Some((k, v)) = list.get(i) else {
                        buf.extend([u8::MAX; 12]);
                        continue;
                    };
                    buf.extend(k.to_le_bytes());
                    let offset = v.value.serialize(bufman, cursor)?;
                    buf.extend(offset.to_le_bytes());
                }
                buf.extend([u8::MAX; 4]);
                let chunk_offset = bufman.write_to_end_of_file(cursor, &buf)?;
                bufman.seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 12))?;
                bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
                prev_chunk_offset = chunk_offset;
            }

            self.serialized_upto.store(list.len(), Ordering::Relaxed);
        }
        let mut chunk_buf = Vec::with_capacity(CHUNK_SIZE * 12 + 12);
        chunk_buf.extend(len.to_le_bytes());
        for i in 0..CHUNK_SIZE {
            let Some((k, v)) = list.get(i) else {
                chunk_buf.extend([u8::MAX; 12]);
                continue;
            };
            chunk_buf.extend(k.to_le_bytes());
            let offset = v.value.serialize(bufman, cursor)?;
            chunk_buf.extend(offset.to_le_bytes());
        }
        chunk_buf.extend([u8::MAX; 4]);
        let start = bufman.write_to_end_of_file(cursor, &chunk_buf)?;
        let mut prev_chunk_offset = start + 8;

        for chunk_idx in 1..total_chunks {
            chunk_buf = Vec::with_capacity(CHUNK_SIZE * 12 + 4);
            for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                let Some((k, v)) = list.get(i) else {
                    chunk_buf.extend([u8::MAX; 12]);
                    continue;
                };
                chunk_buf.extend(k.to_le_bytes());
                let offset = v.value.serialize(bufman, cursor)?;
                chunk_buf.extend(offset.to_le_bytes());
            }
            chunk_buf.extend([u8::MAX; 4]);
            let chunk_offset = bufman.write_to_end_of_file(cursor, &chunk_buf)?;
            bufman.seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 12))?;
            bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
            prev_chunk_offset = chunk_offset;
        }

        *offset_write_guard = Some(FileOffset(start as u32));
        self.serialized_upto.store(list.len(), Ordering::Relaxed);

        Ok(start as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        if offset.0 == u32::MAX {
            return Ok(Self {
                map: TSHashTable::new(16),
                offset: RwLock::new(Some(offset)),
                len: AtomicU64::new(0),
                serialized_upto: AtomicUsize::new(0),
            });
        }
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let len = bufman.read_u64_with_cursor(cursor)?;
        let map = TSHashTable::new(16);

        let total_chunks = (len as usize).div_ceil(CHUNK_SIZE);

        'outer: for chunk_idx in 0..total_chunks {
            for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                if i as u64 == len {
                    break 'outer;
                }
                let key = bufman.read_u64_with_cursor(cursor)?;
                let value_offset = bufman.read_u32_with_cursor(cursor)?;
                let value =
                    UnsafeVersionedItem::<T>::deserialize(bufman, FileOffset(value_offset))?;
                let q = Quotient {
                    value,
                    sequence_idx: i as u64,
                };
                map.insert(key, Arc::new(q));
            }
            let next_chunk_offset = bufman.read_u32_with_cursor(cursor)?;
            if next_chunk_offset == u32::MAX {
                break;
            }
            bufman.seek_with_cursor(cursor, next_chunk_offset as u64)?;
        }

        bufman.close_cursor(cursor)?;
        Ok(Self {
            offset: RwLock::new(Some(offset)),
            map,
            len: AtomicU64::new(len),
            serialized_upto: AtomicUsize::new(len as usize),
        })
    }
}

impl<T: SimpleSerialize> SimpleSerialize for QuotientsMapVec<T> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let mut list = self.map.to_list();
        list.sort_unstable_by_key(|(_, q)| q.sequence_idx);
        if list.is_empty() {
            return Ok(u32::MAX);
        }
        let len = self.len.load(Ordering::Relaxed);
        let offset_read_guard = self.offset.read().map_err(|_| BufIoError::Locking)?;
        let total_chunks = list.len().div_ceil(CHUNK_SIZE);
        if let Some(offset) = *offset_read_guard {
            let serialized_upto = self.serialized_upto.load(Ordering::Relaxed);
            bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            bufman.update_u64_with_cursor(cursor, len)?;

            let total_chunks_serialized = serialized_upto.div_ceil(CHUNK_SIZE);

            for i in 0..CHUNK_SIZE {
                let Some((k, v)) = list.get(i) else {
                    bufman.update_with_cursor(cursor, &[u8::MAX; 12])?;
                    continue;
                };
                let value_offset = v.value.serialize(bufman, cursor)?;
                bufman.seek_with_cursor(cursor, offset.0 as u64 + 8 + (i as u64 * 12))?;
                bufman.update_u64_with_cursor(cursor, *k)?;
                bufman.update_u32_with_cursor(cursor, value_offset)?;
            }

            let mut prev_chunk_offset = offset.0 as u64 + 8;

            for chunk_idx in 1..total_chunks_serialized {
                let current_chunk_offset = bufman.read_u32_with_cursor(cursor)?;
                bufman.seek_with_cursor(cursor, current_chunk_offset as u64)?;
                for i in 0..CHUNK_SIZE {
                    let Some((k, v)) = list.get(i + (chunk_idx * CHUNK_SIZE)) else {
                        bufman.update_with_cursor(cursor, &[u8::MAX; 12])?;
                        continue;
                    };
                    let value_offset = v.value.serialize(bufman, cursor)?;
                    bufman
                        .seek_with_cursor(cursor, current_chunk_offset as u64 + (i as u64 * 12))?;
                    bufman.update_u64_with_cursor(cursor, *k)?;
                    bufman.update_u32_with_cursor(cursor, value_offset)?;
                }

                prev_chunk_offset = current_chunk_offset as u64;
            }

            for chunk_idx in total_chunks_serialized..total_chunks {
                let mut buf = Vec::with_capacity(CHUNK_SIZE * 12 + 4);
                for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                    let Some((k, v)) = list.get(i) else {
                        buf.extend([u8::MAX; 12]);
                        continue;
                    };
                    buf.extend(k.to_le_bytes());
                    let offset = v.value.serialize(bufman, cursor)?;
                    buf.extend(offset.to_le_bytes());
                }
                buf.extend([u8::MAX; 4]);
                let chunk_offset = bufman.write_to_end_of_file(cursor, &buf)?;
                bufman.seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 12))?;
                bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
                prev_chunk_offset = chunk_offset;
            }

            self.serialized_upto.store(list.len(), Ordering::Relaxed);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.offset.write().map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *offset_write_guard {
            let serialized_upto = self.serialized_upto.load(Ordering::Relaxed);
            bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            bufman.update_u64_with_cursor(cursor, len)?;

            let total_chunks_serialized = serialized_upto.div_ceil(CHUNK_SIZE);

            for i in 0..CHUNK_SIZE {
                let Some((k, v)) = list.get(i) else {
                    bufman.update_with_cursor(cursor, &[u8::MAX; 12])?;
                    continue;
                };
                let value_offset = v.value.serialize(bufman, cursor)?;
                bufman.seek_with_cursor(cursor, offset.0 as u64 + 8 + (i as u64 * 12))?;
                bufman.update_u64_with_cursor(cursor, *k)?;
                bufman.update_u32_with_cursor(cursor, value_offset)?;
            }

            let mut prev_chunk_offset = offset.0 as u64 + 8;

            for chunk_idx in 1..total_chunks_serialized {
                let current_chunk_offset = bufman.read_u32_with_cursor(cursor)?;
                bufman.seek_with_cursor(cursor, current_chunk_offset as u64)?;
                for i in 0..CHUNK_SIZE {
                    let Some((k, v)) = list.get(i + (chunk_idx * CHUNK_SIZE)) else {
                        bufman.update_with_cursor(cursor, &[u8::MAX; 12])?;
                        continue;
                    };
                    let value_offset = v.value.serialize(bufman, cursor)?;
                    bufman
                        .seek_with_cursor(cursor, current_chunk_offset as u64 + (i as u64 * 12))?;
                    bufman.update_u64_with_cursor(cursor, *k)?;
                    bufman.update_u32_with_cursor(cursor, value_offset)?;
                }

                prev_chunk_offset = current_chunk_offset as u64;
            }

            for chunk_idx in total_chunks_serialized..total_chunks {
                let mut buf = Vec::with_capacity(CHUNK_SIZE * 12 + 4);
                for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                    let Some((k, v)) = list.get(i) else {
                        buf.extend([u8::MAX; 12]);
                        continue;
                    };
                    buf.extend(k.to_le_bytes());
                    let offset = v.value.serialize(bufman, cursor)?;
                    buf.extend(offset.to_le_bytes());
                }
                buf.extend([u8::MAX; 4]);
                let chunk_offset = bufman.write_to_end_of_file(cursor, &buf)?;
                bufman.seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 12))?;
                bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
                prev_chunk_offset = chunk_offset;
            }

            self.serialized_upto.store(list.len(), Ordering::Relaxed);
        }
        let mut chunk_buf = Vec::with_capacity(CHUNK_SIZE * 12 + 12);
        chunk_buf.extend(len.to_le_bytes());
        for i in 0..CHUNK_SIZE {
            let Some((k, v)) = list.get(i) else {
                chunk_buf.extend([u8::MAX; 12]);
                continue;
            };
            chunk_buf.extend(k.to_le_bytes());
            let offset = v.value.serialize(bufman, cursor)?;
            chunk_buf.extend(offset.to_le_bytes());
        }
        chunk_buf.extend([u8::MAX; 4]);
        let start = bufman.write_to_end_of_file(cursor, &chunk_buf)?;
        let mut prev_chunk_offset = start + 8;

        for chunk_idx in 1..total_chunks {
            chunk_buf = Vec::with_capacity(CHUNK_SIZE * 12 + 4);
            for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                let Some((k, v)) = list.get(i) else {
                    chunk_buf.extend([u8::MAX; 12]);
                    continue;
                };
                chunk_buf.extend(k.to_le_bytes());
                let offset = v.value.serialize(bufman, cursor)?;
                chunk_buf.extend(offset.to_le_bytes());
            }
            chunk_buf.extend([u8::MAX; 4]);
            let chunk_offset = bufman.write_to_end_of_file(cursor, &chunk_buf)?;
            bufman.seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 12))?;
            bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
            prev_chunk_offset = chunk_offset;
        }

        *offset_write_guard = Some(FileOffset(start as u32));
        self.serialized_upto.store(list.len(), Ordering::Relaxed);

        Ok(start as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        if offset.0 == u32::MAX {
            return Ok(Self {
                map: TSHashTable::new(16),
                offset: RwLock::new(Some(offset)),
                len: AtomicU64::new(0),
                serialized_upto: AtomicUsize::new(0),
            });
        }
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let len = bufman.read_u64_with_cursor(cursor)?;
        let map = TSHashTable::new(16);

        let total_chunks = (len as usize).div_ceil(CHUNK_SIZE);

        'outer: for chunk_idx in 0..total_chunks {
            for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                if i as u64 == len {
                    break 'outer;
                }
                let key = bufman.read_u64_with_cursor(cursor)?;
                let value_offset = bufman.read_u32_with_cursor(cursor)?;
                let value = UnsafeVersionedVec::<T>::deserialize(bufman, FileOffset(value_offset))?;
                let q = QuotientVec {
                    value,
                    sequence_idx: i as u64,
                };
                map.insert(key, Arc::new(q));
            }
            let next_chunk_offset = bufman.read_u32_with_cursor(cursor)?;
            if next_chunk_offset == u32::MAX {
                break;
            }
            bufman.seek_with_cursor(cursor, next_chunk_offset as u64)?;
        }

        bufman.close_cursor(cursor)?;
        Ok(Self {
            offset: RwLock::new(Some(offset)),
            map,
            len: AtomicU64::new(len),
            serialized_upto: AtomicUsize::new(len as usize),
        })
    }
}
