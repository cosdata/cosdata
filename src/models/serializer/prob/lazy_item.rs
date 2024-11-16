use std::{
    collections::HashSet,
    io::{self, SeekFrom},
    sync::{atomic::Ordering, Arc},
};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::{Allocate, ProbCache, ProbCacheable},
    lazy_load::FileIndex,
    prob_lazy_load::{
        lazy_item::{ProbLazyItem, ProbLazyItemState},
        lazy_item_array::ProbLazyItemArray,
    },
    types::FileOffset,
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

fn lazy_item_serialize_impl<
    T: ProbCacheable + UpdateSerialized + ProbSerialize + Allocate,
    const N: usize,
>(
    data: Arc<T>,
    versions: &ProbLazyItemArray<T, N>,
    bufmans: Arc<BufferManagerFactory>,
    version: Hash,
    version_number: u16,
    cursor: u64,
    serialized_flag: bool,
    offset: u64,
) -> Result<u32, BufIoError> {
    let bufman = bufmans.get(&version)?;
    if serialized_flag {
        let data_offset = bufman.read_u32_with_cursor(cursor)?;
        let versions_offset = bufman.read_u32_with_cursor(cursor)?;
        let data_file_index = FileIndex::Valid {
            offset: FileOffset(data_offset),
            version_number,
            version_id: version,
        };
        let versions_file_index = FileIndex::Valid {
            offset: FileOffset(versions_offset),
            version_number,
            version_id: version,
        };
        data.update_serialized(bufmans.clone(), data_file_index)?;
        versions.update_serialized(bufmans, versions_file_index)?;
    } else {
        bufman.seek_with_cursor(cursor, SeekFrom::Start(offset + 8))?;
        let data_offset = data.serialize(bufmans.clone(), version, cursor)?;
        let versions_offset = versions.serialize(bufmans.clone(), version, cursor)?;
        let end_offset = bufman.cursor_position(cursor)?;

        bufman.seek_with_cursor(cursor, SeekFrom::Start(offset))?;
        bufman.write_u32_with_cursor(cursor, data_offset)?;
        bufman.write_u32_with_cursor(cursor, versions_offset)?;

        bufman.seek_with_cursor(cursor, SeekFrom::Start(end_offset))?;
    }

    Ok(u32::try_from(offset).unwrap())
}

pub fn lazy_item_deserialize_impl<
    T: ProbCacheable + UpdateSerialized + ProbSerialize + Allocate,
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

impl<T: ProbCacheable + UpdateSerialized + ProbSerialize + Allocate> ProbSerialize
    for *mut ProbLazyItem<T>
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        unsafe {
            match (&**self)
                .get_state()
                .load(Ordering::SeqCst)
                .as_ref()
                .unwrap()
            {
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

                        lazy_item_serialize_impl(
                            data.clone(),
                            versions,
                            bufmans,
                            *version_id,
                            *version_number,
                            cursor,
                            true,
                            file_offset.0 as u64
                        )?;

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

                        let (offset, _) = bufman.write_to_end_with_cursor(cursor, &[u8::MAX; 8])?;

                        file_offset.set(Some(FileOffset(u32::try_from(offset).unwrap())));
                        persist_flag.store(false, Ordering::SeqCst);

                        lazy_item_serialize_impl(
                            data.clone(),
                            versions,
                            bufmans,
                            *version_id,
                            *version_number,
                            cursor,
                            false,
                            offset
                        )?;

                        if version_id != &version {
                            bufman.close_cursor(cursor)?;
                        }

                        u32::try_from(offset).unwrap()
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
