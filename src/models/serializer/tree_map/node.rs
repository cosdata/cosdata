use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use crate::models::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    serializer::SimpleSerialize,
    tree_map::{QuotientsMap, QuotientsMapVec, TreeMapNode, TreeMapVecNode},
    types::FileOffset,
    versioning::VersionNumber,
};

use super::TreeMapSerialize;

impl<T: SimpleSerialize> TreeMapSerialize for TreeMapNode<T> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let offset_read_guard = self.offset.read();
        if let Some(offset) = *offset_read_guard {
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2)?;
            for (idx, child) in self.children.items.iter().enumerate() {
                let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
                let Some(child) = opt_child else {
                    dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                    dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                    continue;
                };
                let child_offset = child.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                dim_bufman.update_u32_with_cursor(cursor, child_offset)?;
            }

            if self.dirty.swap(false, Ordering::Relaxed) {
                let quotient_offset = self.quotients.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 34)?;
                dim_bufman.update_u32_with_cursor(cursor, quotient_offset)?;
            }

            return Ok(offset.0);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.offset.write();
        if let Some(offset) = *offset_write_guard {
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2)?;
            for (idx, child) in self.children.items.iter().enumerate() {
                let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
                let Some(child) = opt_child else {
                    dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                    dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                    continue;
                };
                let child_offset = child.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                dim_bufman.update_u32_with_cursor(cursor, child_offset)?;
            }

            if self.dirty.swap(false, Ordering::Relaxed) {
                let quotient_offset = self.quotients.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 34)?;
                dim_bufman.update_u32_with_cursor(cursor, quotient_offset)?;
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
            let offset = child.serialize(dim_bufman, data_bufmans, cursor)?;
            buf.extend(offset.to_le_bytes());
        }
        let quotient_offset = self.quotients.serialize(dim_bufman, data_bufmans, cursor)?;
        buf.extend(quotient_offset.to_le_bytes());
        let start = dim_bufman.write_to_end_of_file(cursor, &buf)? as u32;
        self.dirty.store(false, Ordering::Release);
        *offset_write_guard = Some(FileOffset(start));
        Ok(start)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let node_idx = dim_bufman.read_u16_with_cursor(cursor)?;
        let children = AtomicArray::new();
        for i in 0..8 {
            let child_offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if child_offset == u32::MAX {
                continue;
            }
            let child =
                Self::deserialize(dim_bufman, data_bufmans, FileOffset(child_offset), version)?;
            children.insert(i, Box::into_raw(Box::new(child)));
        }
        let quotients_offset = dim_bufman.read_u32_with_cursor(cursor)?;
        let quotients = QuotientsMap::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(quotients_offset),
            version,
        )?;
        dim_bufman.close_cursor(cursor)?;

        Ok(Self {
            node_idx,
            offset: RwLock::new(Some(file_offset)),
            quotients,
            children,
            dirty: AtomicBool::new(false),
        })
    }
}

impl<T> TreeMapSerialize for TreeMapVecNode<T> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let offset_read_guard = self.offset.read();
        if let Some(offset) = *offset_read_guard {
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2)?;
            for (idx, child) in self.children.items.iter().enumerate() {
                let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
                let Some(child) = opt_child else {
                    dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                    dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                    continue;
                };
                let child_offset = child.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                dim_bufman.update_u32_with_cursor(cursor, child_offset)?;
            }

            if self.dirty.swap(false, Ordering::Relaxed) {
                let quotient_offset = self.quotients.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 34)?;
                dim_bufman.update_u32_with_cursor(cursor, quotient_offset)?;
            }

            return Ok(offset.0);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.offset.write();
        if let Some(offset) = *offset_write_guard {
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2)?;
            for (idx, child) in self.children.items.iter().enumerate() {
                let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
                let Some(child) = opt_child else {
                    dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                    dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                    continue;
                };
                let child_offset = child.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 2 + (idx as u64 * 4))?;
                dim_bufman.update_u32_with_cursor(cursor, child_offset)?;
            }

            if self.dirty.swap(false, Ordering::Relaxed) {
                let quotient_offset = self.quotients.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset.0 as u64 + 34)?;
                dim_bufman.update_u32_with_cursor(cursor, quotient_offset)?;
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
            let offset = child.serialize(dim_bufman, data_bufmans, cursor)?;
            buf.extend(offset.to_le_bytes());
        }
        let quotient_offset = self.quotients.serialize(dim_bufman, data_bufmans, cursor)?;
        buf.extend(quotient_offset.to_le_bytes());
        let start = dim_bufman.write_to_end_of_file(cursor, &buf)? as u32;
        self.dirty.store(false, Ordering::Release);
        *offset_write_guard = Some(FileOffset(start));
        Ok(start)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let node_idx = dim_bufman.read_u16_with_cursor(cursor)?;
        let children = AtomicArray::new();
        for i in 0..8 {
            let child_offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if child_offset == u32::MAX {
                continue;
            }
            let child =
                Self::deserialize(dim_bufman, data_bufmans, FileOffset(child_offset), version)?;
            children.insert(i, Box::into_raw(Box::new(child)));
        }
        let quotients_offset = dim_bufman.read_u32_with_cursor(cursor)?;
        let quotients = QuotientsMapVec::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(quotients_offset),
            version,
        )?;
        dim_bufman.close_cursor(cursor)?;

        Ok(Self {
            node_idx,
            offset: RwLock::new(Some(file_offset)),
            quotients,
            children,
            dirty: AtomicBool::new(false),
        })
    }
}
