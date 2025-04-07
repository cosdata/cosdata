use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    page::Page,
    types::FileOffset,
};

use super::SimpleSerialize;

impl<const LEN: usize> SimpleSerialize for Page<LEN> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
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
        self.dirty.store(false, Ordering::Release);
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

    fn deserialize(bufman: &BufferManager, file_offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, file_offset.0 as u64 + 4)?;
        let len = bufman.read_u32_with_cursor(cursor)? as usize;
        let mut data = [u32::MAX; LEN];

        for el in data.iter_mut().take(len) {
            *el = bufman.read_u32_with_cursor(cursor)?;
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
