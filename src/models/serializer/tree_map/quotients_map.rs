use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};

use parking_lot::RwLock;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    common::TSHashTable,
    serializer::SimpleSerialize,
    tree_map::{Quotient, QuotientVec, QuotientsMap, QuotientsMapVec, VersionedItem},
    types::FileOffset,
    versioned_vec::VersionedVec,
    versioning::VersionNumber,
};

use super::TreeMapSerialize;

const CHUNK_SIZE: usize = 4;

impl<T: SimpleSerialize> TreeMapSerialize for QuotientsMap<T> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let mut list = self.map.to_list();
        list.sort_unstable_by_key(|(_, q)| q.sequence_idx);
        if list.is_empty() {
            return Ok(u32::MAX);
        }
        let len = self.len.load(Ordering::Relaxed);
        let offset_read_guard = self.offset.read();
        let total_chunks = list.len().div_ceil(CHUNK_SIZE);
        if let Some(offset) = *offset_read_guard {
            let serialized_upto = self.serialized_upto.load(Ordering::Relaxed);
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            dim_bufman.update_u64_with_cursor(cursor, len)?;

            let total_chunks_serialized = serialized_upto.div_ceil(CHUNK_SIZE);

            for i in 0..CHUNK_SIZE {
                let Some((k, v)) = list.get(i) else {
                    dim_bufman.update_with_cursor(cursor, &[u8::MAX; 16])?;
                    continue;
                };
                let val = v.value.read();
                let value_offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 8 + (i as u64 * 16))?;
                dim_bufman.update_u64_with_cursor(cursor, *k)?;
                dim_bufman.update_u32_with_cursor(cursor, value_offset)?;
                dim_bufman.update_u32_with_cursor(cursor, *val.version)?;
            }

            let mut prev_chunk_offset = offset.0 as u64 + 8;

            for chunk_idx in 1..total_chunks_serialized {
                let current_chunk_offset = dim_bufman.read_u32_with_cursor(cursor)?;
                dim_bufman.seek_with_cursor(cursor, current_chunk_offset as u64)?;
                for i in 0..CHUNK_SIZE {
                    let Some((k, v)) = list.get(i + (chunk_idx * CHUNK_SIZE)) else {
                        dim_bufman.update_with_cursor(cursor, &[u8::MAX; 16])?;
                        continue;
                    };
                    let val = v.value.read();
                    let value_offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                    dim_bufman
                        .seek_with_cursor(cursor, current_chunk_offset as u64 + (i as u64 * 16))?;
                    dim_bufman.update_u64_with_cursor(cursor, *k)?;
                    dim_bufman.update_u32_with_cursor(cursor, value_offset)?;
                    dim_bufman.update_u32_with_cursor(cursor, *val.version)?;
                }

                prev_chunk_offset = current_chunk_offset as u64;
            }

            for chunk_idx in total_chunks_serialized..total_chunks {
                let mut buf = Vec::with_capacity(CHUNK_SIZE * 16 + 4);
                for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                    let Some((k, v)) = list.get(i) else {
                        buf.extend([u8::MAX; 16]);
                        continue;
                    };
                    buf.extend(k.to_le_bytes());
                    let val = v.value.read();
                    let offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                    buf.extend(offset.to_le_bytes());
                    buf.extend(val.version.to_le_bytes());
                }
                buf.extend([u8::MAX; 4]);
                let chunk_offset = dim_bufman.write_to_end_of_file(cursor, &buf)?;
                dim_bufman
                    .seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 16))?;
                dim_bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
                prev_chunk_offset = chunk_offset;
            }

            self.serialized_upto.store(list.len(), Ordering::Relaxed);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.offset.write();
        if let Some(offset) = *offset_write_guard {
            let serialized_upto = self.serialized_upto.load(Ordering::Relaxed);
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            dim_bufman.update_u64_with_cursor(cursor, len)?;

            let total_chunks_serialized = serialized_upto.div_ceil(CHUNK_SIZE);

            for i in 0..CHUNK_SIZE {
                let Some((k, v)) = list.get(i) else {
                    dim_bufman.update_with_cursor(cursor, &[u8::MAX; 16])?;
                    continue;
                };
                let val = v.value.read();
                let value_offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 8 + (i as u64 * 16))?;
                dim_bufman.update_u64_with_cursor(cursor, *k)?;
                dim_bufman.update_u32_with_cursor(cursor, value_offset)?;
                dim_bufman.update_u32_with_cursor(cursor, *val.version)?;
            }

            let mut prev_chunk_offset = offset.0 as u64 + 8;

            for chunk_idx in 1..total_chunks_serialized {
                let current_chunk_offset = dim_bufman.read_u32_with_cursor(cursor)?;
                dim_bufman.seek_with_cursor(cursor, current_chunk_offset as u64)?;
                for i in 0..CHUNK_SIZE {
                    let Some((k, v)) = list.get(i + (chunk_idx * CHUNK_SIZE)) else {
                        dim_bufman.update_with_cursor(cursor, &[u8::MAX; 16])?;
                        continue;
                    };
                    let val = v.value.read();
                    let value_offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                    dim_bufman
                        .seek_with_cursor(cursor, current_chunk_offset as u64 + (i as u64 * 16))?;
                    dim_bufman.update_u64_with_cursor(cursor, *k)?;
                    dim_bufman.update_u32_with_cursor(cursor, value_offset)?;
                    dim_bufman.update_u32_with_cursor(cursor, *val.version)?;
                }

                prev_chunk_offset = current_chunk_offset as u64;
            }

            for chunk_idx in total_chunks_serialized..total_chunks {
                let mut buf = Vec::with_capacity(CHUNK_SIZE * 16 + 4);
                for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                    let Some((k, v)) = list.get(i) else {
                        buf.extend([u8::MAX; 16]);
                        continue;
                    };
                    buf.extend(k.to_le_bytes());
                    let val = v.value.read();
                    let offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                    buf.extend(offset.to_le_bytes());
                    buf.extend(val.version.to_le_bytes());
                }
                buf.extend([u8::MAX; 4]);
                let chunk_offset = dim_bufman.write_to_end_of_file(cursor, &buf)?;
                dim_bufman
                    .seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 16))?;
                dim_bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
                prev_chunk_offset = chunk_offset;
            }

            self.serialized_upto.store(list.len(), Ordering::Relaxed);
        }
        let mut chunk_buf = Vec::with_capacity(CHUNK_SIZE * 16 + 16);
        chunk_buf.extend(len.to_le_bytes());
        for i in 0..CHUNK_SIZE {
            let Some((k, v)) = list.get(i) else {
                chunk_buf.extend([u8::MAX; 16]);
                continue;
            };
            chunk_buf.extend(k.to_le_bytes());
            let val = v.value.read();
            let offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
            chunk_buf.extend(offset.to_le_bytes());
            chunk_buf.extend(val.version.to_le_bytes());
        }
        chunk_buf.extend([u8::MAX; 4]);
        let start = dim_bufman.write_to_end_of_file(cursor, &chunk_buf)?;
        let mut prev_chunk_offset = start + 8;

        for chunk_idx in 1..total_chunks {
            chunk_buf = Vec::with_capacity(CHUNK_SIZE * 16 + 4);
            for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                let Some((k, v)) = list.get(i) else {
                    chunk_buf.extend([u8::MAX; 16]);
                    continue;
                };
                chunk_buf.extend(k.to_le_bytes());
                let val = v.value.read();
                let offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                chunk_buf.extend(offset.to_le_bytes());
                chunk_buf.extend(val.version.to_le_bytes());
            }
            chunk_buf.extend([u8::MAX; 4]);
            let chunk_offset = dim_bufman.write_to_end_of_file(cursor, &chunk_buf)?;
            dim_bufman.seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 16))?;
            dim_bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
            prev_chunk_offset = chunk_offset;
        }

        *offset_write_guard = Some(FileOffset(start as u32));
        self.serialized_upto.store(list.len(), Ordering::Relaxed);

        Ok(start as u32)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        offset: FileOffset,
        _version: VersionNumber,
    ) -> Result<Self, BufIoError> {
        if offset.0 == u32::MAX {
            return Ok(Self {
                map: TSHashTable::new(16),
                offset: RwLock::new(None),
                len: AtomicU64::new(0),
                serialized_upto: AtomicUsize::new(0),
            });
        }
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let len = dim_bufman.read_u64_with_cursor(cursor)?;
        let map = TSHashTable::new(16);

        let total_chunks = (len as usize).div_ceil(CHUNK_SIZE);

        'outer: for chunk_idx in 0..total_chunks {
            for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                if i as u64 == len {
                    break 'outer;
                }
                let key = dim_bufman.read_u64_with_cursor(cursor)?;
                let value_offset = dim_bufman.read_u32_with_cursor(cursor)?;
                let value_version = dim_bufman.read_u32_with_cursor(cursor)?;
                let value = VersionedItem::<T>::deserialize(
                    dim_bufman,
                    data_bufmans,
                    FileOffset(value_offset),
                    VersionNumber::from(value_version),
                )?;
                let q = Quotient {
                    value: RwLock::new(value),
                    sequence_idx: i as u64,
                };
                map.insert(key, Arc::new(q));
            }
            let next_chunk_offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if next_chunk_offset == u32::MAX {
                break;
            }
            dim_bufman.seek_with_cursor(cursor, next_chunk_offset as u64)?;
        }

        dim_bufman.close_cursor(cursor)?;
        Ok(Self {
            offset: RwLock::new(Some(offset)),
            map,
            len: AtomicU64::new(len),
            serialized_upto: AtomicUsize::new(len as usize),
        })
    }
}

impl<T> TreeMapSerialize for QuotientsMapVec<T> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let mut list = self.map.to_list();
        list.sort_unstable_by_key(|(_, q)| q.sequence_idx);
        if list.is_empty() {
            return Ok(u32::MAX);
        }
        let len = self.len.load(Ordering::Relaxed);
        let offset_read_guard = self.offset.read();
        let total_chunks = list.len().div_ceil(CHUNK_SIZE);
        if let Some(offset) = *offset_read_guard {
            let serialized_upto = self.serialized_upto.load(Ordering::Relaxed);
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            dim_bufman.update_u64_with_cursor(cursor, len)?;

            let total_chunks_serialized = serialized_upto.div_ceil(CHUNK_SIZE);

            for i in 0..CHUNK_SIZE {
                let Some((k, v)) = list.get(i) else {
                    dim_bufman.update_with_cursor(cursor, &[u8::MAX; 16])?;
                    continue;
                };
                let val = v.value.read();
                let value_offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 8 + (i as u64 * 16))?;
                dim_bufman.update_u64_with_cursor(cursor, *k)?;
                dim_bufman.update_u32_with_cursor(cursor, value_offset)?;
                dim_bufman.update_u32_with_cursor(cursor, *val.version)?;
            }

            let mut prev_chunk_offset = offset.0 as u64 + 8;

            for chunk_idx in 1..total_chunks_serialized {
                let current_chunk_offset = dim_bufman.read_u32_with_cursor(cursor)?;
                dim_bufman.seek_with_cursor(cursor, current_chunk_offset as u64)?;
                for i in 0..CHUNK_SIZE {
                    let Some((k, v)) = list.get(i + (chunk_idx * CHUNK_SIZE)) else {
                        dim_bufman.update_with_cursor(cursor, &[u8::MAX; 16])?;
                        continue;
                    };
                    let val = v.value.read();
                    let value_offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                    dim_bufman
                        .seek_with_cursor(cursor, current_chunk_offset as u64 + (i as u64 * 16))?;
                    dim_bufman.update_u64_with_cursor(cursor, *k)?;
                    dim_bufman.update_u32_with_cursor(cursor, value_offset)?;
                    dim_bufman.update_u32_with_cursor(cursor, *val.version)?;
                }

                prev_chunk_offset = current_chunk_offset as u64;
            }

            for chunk_idx in total_chunks_serialized..total_chunks {
                let mut buf = Vec::with_capacity(CHUNK_SIZE * 16 + 4);
                for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                    let Some((k, v)) = list.get(i) else {
                        buf.extend([u8::MAX; 16]);
                        continue;
                    };
                    buf.extend(k.to_le_bytes());
                    let val = v.value.read();
                    let offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                    buf.extend(offset.to_le_bytes());
                    buf.extend(val.version.to_le_bytes());
                }
                buf.extend([u8::MAX; 4]);
                let chunk_offset = dim_bufman.write_to_end_of_file(cursor, &buf)?;
                dim_bufman
                    .seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 16))?;
                dim_bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
                prev_chunk_offset = chunk_offset;
            }

            self.serialized_upto.store(list.len(), Ordering::Relaxed);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.offset.write();
        if let Some(offset) = *offset_write_guard {
            let serialized_upto = self.serialized_upto.load(Ordering::Relaxed);
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            dim_bufman.update_u64_with_cursor(cursor, len)?;

            let total_chunks_serialized = serialized_upto.div_ceil(CHUNK_SIZE);

            for i in 0..CHUNK_SIZE {
                let Some((k, v)) = list.get(i) else {
                    dim_bufman.update_with_cursor(cursor, &[u8::MAX; 16])?;
                    continue;
                };
                let val = v.value.read();
                let value_offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 8 + (i as u64 * 16))?;
                dim_bufman.update_u64_with_cursor(cursor, *k)?;
                dim_bufman.update_u32_with_cursor(cursor, value_offset)?;
                dim_bufman.update_u32_with_cursor(cursor, *val.version)?;
            }

            let mut prev_chunk_offset = offset.0 as u64 + 8;

            for chunk_idx in 1..total_chunks_serialized {
                let current_chunk_offset = dim_bufman.read_u32_with_cursor(cursor)?;
                dim_bufman.seek_with_cursor(cursor, current_chunk_offset as u64)?;
                for i in 0..CHUNK_SIZE {
                    let Some((k, v)) = list.get(i + (chunk_idx * CHUNK_SIZE)) else {
                        dim_bufman.update_with_cursor(cursor, &[u8::MAX; 16])?;
                        continue;
                    };
                    let val = v.value.read();
                    let value_offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                    dim_bufman
                        .seek_with_cursor(cursor, current_chunk_offset as u64 + (i as u64 * 16))?;
                    dim_bufman.update_u64_with_cursor(cursor, *k)?;
                    dim_bufman.update_u32_with_cursor(cursor, value_offset)?;
                    dim_bufman.update_u32_with_cursor(cursor, *val.version)?;
                }

                prev_chunk_offset = current_chunk_offset as u64;
            }

            for chunk_idx in total_chunks_serialized..total_chunks {
                let mut buf = Vec::with_capacity(CHUNK_SIZE * 16 + 4);
                for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                    let Some((k, v)) = list.get(i) else {
                        buf.extend([u8::MAX; 16]);
                        continue;
                    };
                    buf.extend(k.to_le_bytes());
                    let val = v.value.read();
                    let offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                    buf.extend(offset.to_le_bytes());
                    buf.extend(val.version.to_le_bytes());
                }
                buf.extend([u8::MAX; 4]);
                let chunk_offset = dim_bufman.write_to_end_of_file(cursor, &buf)?;
                dim_bufman
                    .seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 16))?;
                dim_bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
                prev_chunk_offset = chunk_offset;
            }

            self.serialized_upto.store(list.len(), Ordering::Relaxed);
        }
        let mut chunk_buf = Vec::with_capacity(CHUNK_SIZE * 16 + 16);
        chunk_buf.extend(len.to_le_bytes());
        for i in 0..CHUNK_SIZE {
            let Some((k, v)) = list.get(i) else {
                chunk_buf.extend([u8::MAX; 16]);
                continue;
            };
            chunk_buf.extend(k.to_le_bytes());
            let val = v.value.read();
            let offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
            chunk_buf.extend(offset.to_le_bytes());
            chunk_buf.extend(val.version.to_le_bytes());
        }
        chunk_buf.extend([u8::MAX; 4]);
        let start = dim_bufman.write_to_end_of_file(cursor, &chunk_buf)?;
        let mut prev_chunk_offset = start + 8;

        for chunk_idx in 1..total_chunks {
            chunk_buf = Vec::with_capacity(CHUNK_SIZE * 16 + 4);
            for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                let Some((k, v)) = list.get(i) else {
                    chunk_buf.extend([u8::MAX; 16]);
                    continue;
                };
                chunk_buf.extend(k.to_le_bytes());
                let val = v.value.read();
                let offset = val.serialize(dim_bufman, data_bufmans, cursor)?;
                chunk_buf.extend(offset.to_le_bytes());
                chunk_buf.extend(val.version.to_le_bytes());
            }
            chunk_buf.extend([u8::MAX; 4]);
            let chunk_offset = dim_bufman.write_to_end_of_file(cursor, &chunk_buf)?;
            dim_bufman.seek_with_cursor(cursor, prev_chunk_offset + (CHUNK_SIZE as u64 * 16))?;
            dim_bufman.update_u32_with_cursor(cursor, chunk_offset as u32)?;
            prev_chunk_offset = chunk_offset;
        }

        *offset_write_guard = Some(FileOffset(start as u32));
        self.serialized_upto.store(list.len(), Ordering::Relaxed);

        Ok(start as u32)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        offset: FileOffset,
        _version: VersionNumber,
    ) -> Result<Self, BufIoError> {
        if offset.0 == u32::MAX {
            return Ok(Self {
                map: TSHashTable::new(16),
                offset: RwLock::new(None),
                len: AtomicU64::new(0),
                serialized_upto: AtomicUsize::new(0),
            });
        }
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let len = dim_bufman.read_u64_with_cursor(cursor)?;
        let map = TSHashTable::new(16);

        let total_chunks = (len as usize).div_ceil(CHUNK_SIZE);

        'outer: for chunk_idx in 0..total_chunks {
            for i in (chunk_idx * CHUNK_SIZE)..((chunk_idx + 1) * CHUNK_SIZE) {
                if i as u64 == len {
                    break 'outer;
                }
                let key = dim_bufman.read_u64_with_cursor(cursor)?;
                let value_offset = dim_bufman.read_u32_with_cursor(cursor)?;
                let value_version = dim_bufman.read_u32_with_cursor(cursor)?;
                let value = VersionedVec::<T>::deserialize(
                    dim_bufman,
                    data_bufmans,
                    FileOffset(value_offset),
                    VersionNumber::from(value_version),
                )?;
                let q = QuotientVec {
                    value: RwLock::new(value),
                    sequence_idx: i as u64,
                };
                map.insert(key, Arc::new(q));
            }
            let next_chunk_offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if next_chunk_offset == u32::MAX {
                break;
            }
            dim_bufman.seek_with_cursor(cursor, next_chunk_offset as u64)?;
        }

        dim_bufman.close_cursor(cursor)?;
        Ok(Self {
            offset: RwLock::new(Some(offset)),
            map,
            len: AtomicU64::new(len),
            serialized_upto: AtomicUsize::new(len as usize),
        })
    }
}
