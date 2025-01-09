use super::CustomSerialize;
use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::{Cacheable, NodeRegistry},
    lazy_load::{EagerLazyItem, FileIndex, LazyItem, SyncPersist},
    types::FileOffset,
    versioning::Hash,
};
use std::collections::HashSet;
use std::io::{self, SeekFrom};
use std::sync::Arc;

impl<T, E> CustomSerialize for EagerLazyItem<T, E>
where
    T: Cacheable + CustomSerialize + Clone + 'static,
    E: Clone + CustomSerialize + 'static,
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start = bufman.cursor_position(cursor)?;
        bufman.write_u32_with_cursor(cursor, 0)?;
        bufman.write_u32_with_cursor(cursor, 0)?;
        bufman.write_u16_with_cursor(cursor, self.1.get_current_version_number())?;
        bufman.write_u32_with_cursor(cursor, *self.1.get_current_version())?;
        let eager_data_offset = self.0.serialize(bufmans.clone(), version, cursor)?;
        let item_offset = self.1.serialize(bufmans.clone(), version, cursor)?;
        let end_position = bufman.cursor_position(cursor)?;

        bufman.seek_with_cursor(cursor, SeekFrom::Start(start))?;
        bufman.write_u32_with_cursor(cursor, eager_data_offset)?;
        bufman.write_u32_with_cursor(cursor, item_offset)?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(end_position))?;

        Ok(start as u32)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError>
    where
        Self: Sized,
    {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize EagerLazyItem with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                offset,
                version_number,
                version_id,
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset.0 as u64))?;
                let eager_data_offset = bufman.read_u32_with_cursor(cursor)?;
                let item_offset = bufman.read_u32_with_cursor(cursor)?;
                let item_version_number = bufman.read_u16_with_cursor(cursor)?;
                let item_version_id = bufman.read_u32_with_cursor(cursor)?.into();
                let eager_data_file_index = FileIndex::Valid {
                    offset: FileOffset(eager_data_offset),
                    version_number,
                    version_id,
                };
                let eager_data = E::deserialize(
                    bufmans.clone(),
                    eager_data_file_index,
                    cache.clone(),
                    max_loads,
                    skipm,
                )?;
                let item_file_index = FileIndex::Valid {
                    offset: FileOffset(item_offset),
                    version_number: item_version_number,
                    version_id: item_version_id,
                };
                let item =
                    LazyItem::deserialize(bufmans, item_file_index, cache, max_loads, skipm)?;
                Ok(Self(eager_data, item))
            }
        }
    }
}
