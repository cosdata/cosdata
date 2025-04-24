use std::sync::{
    atomic::{AtomicBool, Ordering},
    RwLock,
};

use crate::models::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManagerFactory},
    tree_map::{QuotientsMap, QuotientsMapVec, TreeMapKey, TreeMapNode, TreeMapVecNode},
    types::FileOffset,
};

use super::{PartitionedSerialize, SimpleSerialize};

impl<K: TreeMapKey + SimpleSerialize + Clone, V: SimpleSerialize> PartitionedSerialize
    for TreeMapNode<K, V>
{
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<u8>,
        file_parts: u8,
        _file_idx: u8,
        _cursor: u64,
    ) -> Result<u32, BufIoError> {
        let file_idx = (self.node_idx % file_parts as u16) as u8;
        let bufman = bufmans.get(file_idx)?;
        let cursor = bufman.open_cursor()?;
        let offset_read_guard = self.offset.read().map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *offset_read_guard {
            bufman.seek_with_cursor(cursor, offset.0 as u64 + 2)?;
            for (idx, child) in self.children.items.iter().enumerate() {
                let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
                let Some(child) = opt_child else {
                    bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                    bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                    continue;
                };
                let child_offset = child.serialize(bufmans, file_parts, file_idx, cursor)?;
                bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                bufman.update_u32_with_cursor(cursor, child_offset)?;
            }

            if self.dirty.swap(true, Ordering::Relaxed) {
                let quotient_offset = self.quotients.serialize(&bufman, cursor)?;
                bufman.seek_with_cursor(cursor, offset.0 as u64 + 34)?;
                bufman.update_u32_with_cursor(cursor, quotient_offset)?;
            }

            return Ok(offset.0);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.offset.write().map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *offset_write_guard {
            bufman.seek_with_cursor(cursor, offset.0 as u64 + 2)?;
            for (idx, child) in self.children.items.iter().enumerate() {
                let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
                let Some(child) = opt_child else {
                    bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                    bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                    continue;
                };
                let child_offset = child.serialize(bufmans, file_parts, file_idx, cursor)?;
                bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                bufman.update_u32_with_cursor(cursor, child_offset)?;
            }

            if self.dirty.swap(true, Ordering::Relaxed) {
                let quotient_offset = self.quotients.serialize(&bufman, cursor)?;
                bufman.seek_with_cursor(cursor, offset.0 as u64 + 34)?;
                bufman.update_u32_with_cursor(cursor, quotient_offset)?;
            }

            return Ok(offset.0);
        }
        let mut buf = Vec::with_capacity(38);
        buf.extend(self.node_idx.to_le_bytes());
        for child in &self.children.items {
            let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
            let Some(child) = opt_child else {
                buf.extend([u8::MAX; 4]);
                continue;
            };
            let offset = child.serialize(bufmans, file_parts, file_idx, cursor)?;
            buf.extend(offset.to_le_bytes());
        }
        let quotient_offset = self.quotients.serialize(&bufman, cursor)?;
        buf.extend(quotient_offset.to_le_bytes());
        let start = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        bufman.close_cursor(cursor)?;
        *offset_write_guard = Some(FileOffset(start));
        Ok(start)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<u8>,
        file_parts: u8,
        file_idx: u8,
        file_offset: FileOffset,
    ) -> Result<Self, BufIoError> {
        let bufman = bufmans.get(file_idx)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let node_idx = bufman.read_u16_with_cursor(cursor)?;
        debug_assert_eq!((node_idx % file_parts as u16) as u8, file_idx);
        let children = AtomicArray::new();
        for i in 0..8 {
            let child_offset = bufman.read_u32_with_cursor(cursor)?;
            if child_offset == u32::MAX {
                continue;
            }
            let child_node_idx = node_idx + (1u16 << (i * 2));
            let file_idx = (child_node_idx % file_parts as u16) as u8;
            let child = Self::deserialize(bufmans, file_parts, file_idx, FileOffset(child_offset))?;
            children.insert(i, Box::into_raw(Box::new(child)));
        }
        let quotients_offset = bufman.read_u32_with_cursor(cursor)?;
        let quotients = QuotientsMap::deserialize(&bufman, FileOffset(quotients_offset))?;
        bufman.close_cursor(cursor)?;

        Ok(Self {
            node_idx,
            offset: RwLock::new(Some(file_offset)),
            quotients,
            children,
            dirty: AtomicBool::new(false),
        })
    }
}
