use std::{
    collections::HashSet,
    io::{self, SeekFrom},
    sync::{atomic::Ordering, Arc},
};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::{ProbCache, ProbCacheable},
    lazy_load::FileIndex,
    prob_lazy_load::{
        lazy_item::{ProbLazyItem, ProbLazyItemState},
        lazy_item_array::ProbLazyItemArray,
    },
    types::FileOffset,
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

pub fn lazy_item_deserialize_impl<
    T: ProbCacheable + UpdateSerialized + ProbSerialize,
    const N: usize,
>(
    bufmans: Arc<BufferManagerFactory>,
    file_index: FileIndex,
    cache: Arc<ProbCache>,
    max_loads: u16,
    skipm: &mut HashSet<u64>,
) -> Result<(T, ProbLazyItemArray<T, N>), BufIoError> {
    match file_index {
        FileIndex::Invalid => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Cannot deserialize ProbLazyItem with an invalid FileIndex",
        )
        .into()),
        FileIndex::Valid {
            offset,
            version_id,
            version_number,
        } => {
            let bufman = bufmans.get(&version_id)?;
            let cursor = bufman.open_cursor()?;
            bufman.seek_with_cursor(cursor, SeekFrom::Start(offset.0 as u64))?;
            let node_offset = bufman.read_u32_with_cursor(cursor)?;
            let versions_offset = bufman.read_u32_with_cursor(cursor)?;
            let data = T::deserialize(
                bufmans.clone(),
                FileIndex::Valid {
                    offset: FileOffset(node_offset),
                    version_id,
                    version_number,
                },
                cache.clone(),
                max_loads,
                skipm,
            )?;
            let versions = ProbLazyItemArray::deserialize(
                bufmans.clone(),
                FileIndex::Valid {
                    offset: FileOffset(versions_offset),
                    version_number,
                    version_id,
                },
                cache,
                max_loads,
                skipm,
            )?;

            Ok((data, versions))
        }
    }
}

impl<T: ProbCacheable + UpdateSerialized + ProbSerialize> ProbSerialize for *mut ProbLazyItem<T> {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        assert!(!self.is_null());
        unsafe {
            match &*(&**self).get_state() {
                ProbLazyItemState::Pending { file_index } => Ok(file_index.get_offset().unwrap().0),
                ProbLazyItemState::Ready {
                    data,
                    file_offset,
                    persist_flag,
                    version_id,
                    version_number,
                    versions,
                    ..
                } => {
                    let bufman = bufmans.get(version_id)?;

                    let offset = if let Some(file_offset) = file_offset.get() {
                        if !persist_flag.swap(false, Ordering::SeqCst) {
                            return Ok(file_offset.0);
                        }

                        let cursor = if version_id == &version {
                            cursor
                        } else {
                            bufman.open_cursor()?
                        };
                        bufman.seek_with_cursor(cursor, SeekFrom::Start(file_offset.0 as u64))?;

                        let data_offset = bufman.read_u32_with_cursor(cursor)?;
                        let versions_offset = bufman.read_u32_with_cursor(cursor)?;
                        let data_file_index = FileIndex::Valid {
                            offset: FileOffset(data_offset),
                            version_number: *version_number,
                            version_id: *version_id,
                        };
                        let versions_file_index = FileIndex::Valid {
                            offset: FileOffset(versions_offset),
                            version_number: *version_number,
                            version_id: *version_id,
                        };
                        data.update_serialized(bufmans.clone(), data_file_index)?;
                        versions.update_serialized(bufmans, versions_file_index)?;

                        if version_id != &version {
                            bufman.close_cursor(cursor)?;
                        }

                        file_offset.0
                    } else {
                        let cursor = if version_id == &version {
                            cursor
                        } else {
                            bufman.open_cursor()?
                        };

                        let offset = bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

                        file_offset.set(Some(FileOffset(u32::try_from(offset).unwrap())));
                        persist_flag.store(false, Ordering::SeqCst);

                        // 4 bytes for data offset + 4 bytes versions for offset
                        bufman.write_with_cursor(cursor, &[u8::MAX; 8])?;

                        let data_offset = data.serialize(bufmans.clone(), version, cursor)?;
                        let versions_offset =
                            versions.serialize(bufmans.clone(), version, cursor)?;
                        let end_offset = bufman.cursor_position(cursor)?;

                        bufman.seek_with_cursor(cursor, SeekFrom::Start(offset))?;
                        bufman.write_u32_with_cursor(cursor, data_offset)?;
                        bufman.write_u32_with_cursor(cursor, versions_offset)?;

                        bufman.seek_with_cursor(cursor, SeekFrom::Start(end_offset))?;

                        if version_id != &version {
                            bufman.close_cursor(cursor)?;
                        }

                        offset as u32
                    };

                    Ok(offset)
                }
            }
        }
    }

    fn deserialize(
        _bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<ProbCache>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        cache.get_lazy_object(file_index, max_loads, skipm)
    }
}
