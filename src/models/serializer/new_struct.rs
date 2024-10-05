use super::CustomSerialize;
use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::NodeRegistry,
    lazy_load::{FileIndex, NewStruct, VectorData},
    types::FileOffset,
    versioning::Hash,
};
use std::collections::HashSet;
use std::{io::SeekFrom, sync::Arc};

impl CustomSerialize for NewStruct {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(&version)?;
        let start_offset = bufman.cursor_position(cursor)? as u32;
        let mut items = self.items.clone();
        let total_items = items.len();

        // Serialize number of items in the vector
        // Each item is an array of 64 u32s
        bufman.write_u32_with_cursor(cursor, total_items as u32)?;

        // Serialize individual items
        for item in items.iter() {
            let item_start_offset = bufman.cursor_position(cursor)? as u32;
            if item.is_serialized() {
                // If the array is already serialized, move the cursor forward by 64 * 4 bytes and serialize the next array
                bufman
                    .seek_with_cursor(cursor, SeekFrom::Start(item_start_offset as u64 + 64 * 4))?;
                continue;
            }

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
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Ok(NewStruct::new()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version,
            } => {
                let bufman = bufmans.get(&version)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                let mut items = Vec::new();

                // Deserialize the number of items in the vector
                let total_items = bufman.read_u32_with_cursor(cursor)? as usize;

                // Deserialize individual items
                for _ in 0..total_items {
                    let mut item = [u32::MAX; 64];
                    for i in 0..64 {
                        let val = bufman.read_u32_with_cursor(cursor)?;
                        item[i] = val;
                    }
                    items.push(VectorData::from_array(item, true))
                }

                Ok(NewStruct { items })
            }
        }
    }
}
