use super::CustomSerialize;
use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::NodeRegistry,
    lazy_load::{
        FileIndex, IncrementalSerializableGrowableData, LazyItem, LazyItemVec, SyncPersist,
        VectorData,
    },
    types::{FileOffset, STM},
    versioning::Hash,
};
use std::collections::HashSet;
use std::{io::SeekFrom, sync::Arc};

impl CustomSerialize for IncrementalSerializableGrowableData {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)? as u32;

        // Store (data, version) pairs in a vector for serialization
        let items: Vec<_> = self
            .items
            .iter()
            .map(|item| {
                (
                    (*item.get_lazy_data().unwrap().get().clone().unwrap())
                        .clone()
                        .get()
                        .clone(),
                    item.get_current_version_number(),
                    item.get_current_version(),
                )
            })
            .collect();

        let total_items = items.len();

        // Serialize number of items in the vector
        // Each item is an array of 64 u32s
        bufman.write_u32_with_cursor(cursor, total_items as u32)?;

        // Serialize individual items
        // First store version, then the array of 64 u32s
        for (item, version_number, version_id) in items.into_iter() {
            let item_start_offset = bufman.cursor_position(cursor)? as u32;
            if item.is_serialized() {
                // If the array is already serialized, move the cursor forward by 6 + (64 * 4) bytes (6 bytes for version_id and version_number and 64 * 4 bytes for items) and serialize the next array
                bufman.seek_with_cursor(
                    cursor,
                    SeekFrom::Start(item_start_offset as u64 + 64 * 4 + 6),
                )?;
                continue;
            }

            // Serialize the version
            bufman.write_u32_with_cursor(cursor, *version_id)?;
            bufman.write_u16_with_cursor(cursor, version_number)?;

            // Serialize the array
            for i in 0..64 {
                bufman.write_u32_with_cursor(
                    cursor,
                    match item.get(i) {
                        Some(val) => val,
                        None => u32::MAX,
                    },
                )?;
            }
        }

        Ok(start_offset)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        _cache: Arc<NodeRegistry>,
        _max_loads: u16,
        _skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Ok(IncrementalSerializableGrowableData::new()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                ..
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                let items: LazyItemVec<STM<VectorData>> = LazyItemVec::new();

                // Deserialize the number of items in the vector
                let total_items = bufman.read_u32_with_cursor(cursor)? as usize;

                // Deserialize individual items
                for _ in 0..total_items {
                    let mut item = [u32::MAX; 64];
                    // Deserialize version
                    let version_id = bufman.read_u32_with_cursor(cursor)?;
                    let version_number = bufman.read_u16_with_cursor(cursor)?;

                    // Deserialize elements
                    for i in 0..64 {
                        let val = bufman.read_u32_with_cursor(cursor)?;
                        item[i] = val;
                    }
                    items.push(LazyItem::new(
                        Hash::from(version_id),
                        version_number,
                        STM::new(VectorData::from_array(item, true), 1, true),
                    ));
                }

                Ok(IncrementalSerializableGrowableData { items })
            }
        }
    }
}

#[allow(unused_variables)]
impl CustomSerialize for STM<VectorData> {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        //todo! Implement serialize

        Ok(0u32)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        //todo! Implement deserialize

        Ok(STM::new(VectorData::new(), 1, true))
    }
}
