use crate::models::serializer::CustomSerialize;
use crate::models::types::FileOffset;
use std::io::SeekFrom;

#[derive(Clone, Default)]
pub struct Pagepool<const LEN: usize> {
    inner: Vec<Page<LEN>>,
    current: usize,
}

impl<const LEN: usize> Pagepool<LEN> {
    pub fn push(&mut self, data: u32) {
        // If all the current list of chunks are full then create a new one
        if self.current == self.inner.len() - 1 {
            let mut page = Page::<LEN>::new();
            page.push(data);
        }
        // If the current chunk is full then go the next chunk
        else if self.inner[self.current].is_full() {
            self.current += 1;
            self.inner[self.current].push(data);
        }
        // Otherwise push the data to the current chunk
        else {
            self.inner[self.current].push(data);
        }
    }

    pub fn push_chunk(&mut self, chunk: [u32; LEN]) {
        self.inner.push(Page::<LEN>::from_data(chunk))
    }

    pub fn contains(&self, data: u32) -> bool {
        self.inner.iter().any(|p| p.data.contains(&data))
    }
}

impl<const LEN: usize> std::ops::Deref for Pagepool<LEN> {
    type Target = Vec<Page<LEN>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Clone, PartialEq)]
pub struct Page<const LEN: usize> {
    pub data: [u32; LEN],
    current: usize,
    is_serialized: bool,
}

impl<const LEN: usize> std::ops::Deref for Page<LEN> {
    type Target = [u32; LEN];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<const LEN: usize> Page<LEN> {
    fn new() -> Self {
        Self {
            data: [u32::MAX; LEN],
            is_serialized: false,
            current: 0,
        }
    }

    fn push(&mut self, data: u32) {
        self.current += 1;
        self.data[self.current] = data;
    }

    fn from_data(data: [u32; LEN]) -> Self {
        Self {
            data,
            is_serialized: false,
            current: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.current == LEN - 1
    }
}

impl<const LEN: usize> AsRef<[u32; LEN]> for Page<LEN> {
    fn as_ref(&self) -> &[u32; LEN] {
        &self.data
    }
}

impl<const LEN: usize> CustomSerialize for Pagepool<LEN> {
    fn serialize(
        &self,
        bufmans: std::sync::Arc<crate::models::buffered_io::BufferManagerFactory>,
        version: crate::models::versioning::Hash,
        cursor: u64,
    ) -> Result<u32, crate::models::buffered_io::BufIoError> {
        let bufman = bufmans.get(&version)?;
        let start_offset = bufman.cursor_position(cursor)? as u32;
        let total_items = self.inner.len();

        // Write total length in the memory
        bufman.write_u32_with_cursor(cursor, total_items as u32)?;

        for (pos, item) in self.inner.iter().enumerate() {
            Page::<LEN>::serialize(item, bufmans.clone(), version, cursor)?;

            if pos == total_items - 1 {
                bufman.write_u32_with_cursor(cursor, u32::MAX)?;
            } else {
                let current = bufman.cursor_position(cursor)?;
                bufman.write_u32_with_cursor(cursor, current as u32)?;
            }
        }

        Ok(start_offset)
    }

    fn deserialize(
        bufmans: std::sync::Arc<crate::models::buffered_io::BufferManagerFactory>,
        file_index: crate::models::lazy_load::FileIndex,
        cache: std::sync::Arc<crate::models::cache_loader::NodeRegistry>,
        max_loads: u16,
        skipm: &mut std::collections::HashSet<u64>,
    ) -> Result<Self, crate::models::buffered_io::BufIoError> {
        match file_index {
            crate::models::lazy_load::FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                ..
            } => {
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;

                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

                let total_length = bufman.read_u32_with_cursor(cursor)? as usize;

                let mut page_pool = Pagepool::<LEN>::default();
                for _ in 0..total_length {
                    let page = Page::<LEN>::deserialize(
                        bufmans.clone(),
                        file_index.clone(),
                        cache.clone(),
                        max_loads,
                        skipm,
                    )?;
                    page_pool.push_chunk(page.data);

                    let next_chunk = bufman.read_u32_with_cursor(cursor)?;
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(next_chunk as u64))?;
                }

                Ok(page_pool)
            }
            crate::models::lazy_load::FileIndex::Invalid => Ok(Pagepool::default()),
        }
    }
}

impl<const LEN: usize> CustomSerialize for Page<LEN> {
    fn deserialize(
        bufmans: std::sync::Arc<crate::models::buffered_io::BufferManagerFactory>,
        file_index: crate::models::lazy_load::FileIndex,
        _cache: std::sync::Arc<crate::models::cache_loader::NodeRegistry>,
        _max_loads: u16,
        _skipm: &mut std::collections::HashSet<u64>,
    ) -> Result<Self, crate::models::buffered_io::BufIoError> {
        match file_index {
            crate::models::lazy_load::FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                ..
            } => {
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

                let mut items = Page::<LEN>::new();

                for _ in 0..LEN {
                    items.push(bufman.read_u32_with_cursor(cursor)?)
                }

                Ok(items)
            }

            crate::models::lazy_load::FileIndex::Invalid => Ok(Page::<LEN>::new()),
        }
    }

    fn serialize(
        &self,
        bufmans: std::sync::Arc<crate::models::buffered_io::BufferManagerFactory>,
        version: crate::models::versioning::Hash,
        cursor: u64,
    ) -> Result<u32, crate::models::buffered_io::BufIoError> {
        let bufman = bufmans.get(&version)?;
        let start_offset = bufman.cursor_position(cursor)? as u32;

        // If the chunk is already serialised then do an early return
        if self.is_serialized {
            return Ok(start_offset);
        }

        for item in self.data {
            bufman.write_u32_with_cursor(cursor, item)?;
        }

        Ok(start_offset)
    }
}
