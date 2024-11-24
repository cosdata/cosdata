use crate::models::serializer::CustomSerialize;
use crate::models::types::FileOffset;
use std::io::SeekFrom;

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Pagepool<const LEN: usize> {
    inner: Vec<Page<LEN>>,
    current: usize,
}

impl<const LEN: usize> Pagepool<LEN> {
    pub fn push(&mut self, data: u32) {
        // If all the current list of chunks are full then create a new one
        if self.inner.is_empty() || self.current == self.inner.len() - 1 {
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

#[derive(Clone, PartialEq, Debug)]
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

        // Move the cursor to the end of the file and start writing from there
        bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

        let start_offset = bufman.cursor_position(cursor)? as u32;
        let total_items = self.inner.len();

        bufman.write_u32_with_cursor(cursor, total_items as u32)?;

        for (pos, item) in self.inner.iter().enumerate() {
            Page::<LEN>::serialize(item, bufmans.clone(), version, cursor)?;

            if pos == total_items - 1 {
                bufman.write_u32_with_cursor(cursor, u32::MAX)?;
            } else {
                // Move the cursor to the end of the file and get the position of the file
                // and write that in the current position
                let current = bufman.cursor_position(cursor)?;
                bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
                let next_chunk_start = bufman.cursor_position(cursor)?;

                // Move the cursor back to the current position
                bufman.seek_with_cursor(cursor, SeekFrom::Start(current))?;
                bufman.write_u32_with_cursor(cursor, next_chunk_start as u32)?;
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
                version_number,
            } => {
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;

                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

                let total_length = bufman.read_u32_with_cursor(cursor)? as usize;

                let mut page_pool = Pagepool::<LEN>::default();
                for _ in 0..total_length {
                    let current_offset = bufman.cursor_position(cursor)?;

                    let file_index = crate::models::lazy_load::FileIndex::Valid {
                        offset: FileOffset(current_offset as u32),
                        version_id,
                        version_number,
                    };

                    let page = Page::<LEN>::deserialize(
                        bufmans.clone(),
                        file_index,
                        cache.clone(),
                        max_loads,
                        skipm,
                    )?;

                    page_pool.push_chunk(page.data);

                    let next_chunk = bufman.read_u32_with_cursor(cursor)?;

                    if next_chunk == u32::MAX {
                        break;
                    }

                    bufman.seek_with_cursor(cursor, SeekFrom::Start(next_chunk as u64))?;
                }

                bufman.close_cursor(cursor)?;
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

                items.is_serialized = true;

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

#[cfg(test)]
mod page_tests {
    use super::*;

    use std::collections::HashSet;

    use crate::models::{
        buffered_io::{BufferManager, BufferManagerFactory},
        cache_loader::NodeRegistry,
        lazy_load::FileIndex,
        serializer::CustomSerialize,
        types::FileOffset,
        versioning::Hash,
    };
    use std::sync::Arc;

    use tempfile::{tempdir, TempDir};

    fn setup_test() -> (
        Arc<BufferManagerFactory>,
        Arc<BufferManager>,
        u64,
        TempDir,
        Arc<NodeRegistry>,
    ) {
        let root_version_id = Hash::from(0);

        let dir = tempdir().unwrap();
        let bufmans = Arc::new(BufferManagerFactory::new(
            dir.as_ref().into(),
            |root, ver| root.join(format!("{}.index", **ver)),
        ));

        let cache = Arc::new(NodeRegistry::new(1000, bufmans.clone()));
        let bufman = bufmans.get(&root_version_id).unwrap();
        let cursor = bufman.open_cursor().unwrap();
        (bufmans, bufman, cursor, dir, cache)
    }

    #[test]
    fn test_serialize_deserialize_page() {
        let mut page_pool = Pagepool::<10>::default();
        let mut skipm: HashSet<u64> = HashSet::new();

        for i in 0..10 * 10_u32 {
            page_pool.push(i);
        }

        let root_version_id = Hash::from(0);
        let root_version_number = 0;

        let (bufmgr_factory, bufmg, cursor, temp_dir, cache) = setup_test();
        let offset = page_pool.serialize(bufmgr_factory.clone(), root_version_id, cursor);

        assert!(offset.is_ok());

        let offset = offset.unwrap();
        bufmg.close_cursor(cursor).unwrap();

        let deser = Pagepool::<10>::deserialize(
            bufmgr_factory.clone(),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id: root_version_id,
                version_number: root_version_number,
            },
            cache.clone(),
            0_u16,
            &mut skipm,
        );
        assert!(deser.is_ok());
        let deser = deser.unwrap();

        assert_eq!(page_pool, deser);
    }
}
