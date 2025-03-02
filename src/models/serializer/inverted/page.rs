use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
        cache_loader::InvertedIndexCache,
        types::FileOffset,
    },
    storage::page::Page,
};

use super::InvertedIndexSerialize;

impl<const LEN: usize> InvertedIndexSerialize for Page<LEN> {
    fn serialize(
        &self,
        _dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = data_bufmans.get(data_file_idx)?;
        if let Some(offset) = *self.serialized_at.read().map_err(|_| BufIoError::Locking)? {
            if !self.dirty.swap(false, Ordering::AcqRel) {
                return Ok(offset.0);
            }
            bufman.seek_with_cursor(cursor, offset.0 as u64 + 4)?;
            bufman.update_u32_with_cursor(cursor, self.len as u32)?;
            for el in &self.data {
                bufman.update_u32_with_cursor(cursor, *el)?;
            }
            return Ok(offset.0);
        }
        let mut serialized_at = self
            .serialized_at
            .write()
            .map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *serialized_at {
            if !self.dirty.swap(false, Ordering::AcqRel) {
                return Ok(offset.0);
            }
            bufman.seek_with_cursor(cursor, offset.0 as u64 + 4)?;
            bufman.update_u32_with_cursor(cursor, self.len as u32)?;
            for el in &self.data {
                bufman.update_u32_with_cursor(cursor, *el)?;
            }
            return Ok(offset.0);
        }
        let mut data = Vec::with_capacity(8 + LEN * 4);
        data.extend([u8::MAX; 4]);
        data.extend((self.len as u32).to_le_bytes());
        for el in &self.data {
            data.extend(el.to_le_bytes());
        }
        let offset = bufman.write_to_end_of_file(cursor, &data)? as u32;
        *serialized_at = Some(FileOffset(offset));
        Ok(offset)
    }

    fn deserialize(
        _dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        _cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let bufman = data_bufmans.get(data_file_idx)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, file_offset.0 as u64 + 4)?;
        let len = bufman.read_u32_with_cursor(cursor)? as usize;
        let mut data = [u32::MAX; LEN];

        for i in 0..len {
            let el = bufman.read_u32_with_cursor(cursor)?;
            data[i] = el;
        }

        bufman.close_cursor(cursor)?;
        Ok(Self {
            data,
            len,
            serialized_at: Arc::new(RwLock::new(Some(file_offset))),
            dirty: Arc::new(AtomicBool::new(false)),
        })
    }
}
