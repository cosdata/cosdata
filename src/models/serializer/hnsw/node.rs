use std::{collections::HashSet, ptr, sync::atomic::AtomicPtr};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::HNSWIndexCache,
    prob_lazy_load::{lazy_item::FileIndex, lazy_item_array::ProbLazyItemArray},
    prob_node::{ProbNode, SharedNode},
    types::{BytesToRead, FileOffset, HNSWLevel, MetricResult},
    versioning::Hash,
};

use super::HNSWIndexSerialize;

// @SERIALIZED_SIZE:
//   Properties:
//     1 byte for HNSW level +             | 1
//     8 bytes for prop offset & length    | 1 + 8 = 9
//
//   Links:
//     10 bytes for parent offset & version +           | 10
//     10 bytes for child offset & version +            | 20
//     10 bytes for root version offset & version +     | 30
//     2 bytes for neighbors length +                   | 32
//     neighbors length * 19 bytes for neighbor link +  | nb * 19 + 32
//     8 * 10 bytes for version link                    | nb * 19 + 112
//
//   Total = nb * 19 + 121 (where `nb` is the neighbors count)
impl HNSWIndexSerialize for ProbNode {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let is_level_0 = self.hnsw_level.0 == 0;
        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)?;

        let neighbors = self.get_neighbors_raw();
        let size = Self::get_serialized_size(neighbors.len()) as u64;

        debug_assert_eq!(start_offset % size, 0, "offset: {}", start_offset);

        let mut buf = Vec::with_capacity(39);

        // Serialize basic fields
        buf.push(self.hnsw_level.0);

        // Serialize prop
        let (FileOffset(offset), BytesToRead(length)) = &self.prop.location;
        buf.extend(offset.to_le_bytes());
        buf.extend(length.to_le_bytes());

        let parent_ptr = self.get_parent();

        // Get parent file index
        let parent_file_index = if let Some(parent) = unsafe { parent_ptr.as_ref() } {
            debug_assert!(!parent.is_level_0);
            Some(parent.get_file_index())
        } else {
            None
        };

        let child_ptr = self.get_child();

        // Get child file index
        let child_file_index = if let Some(child) = unsafe { child_ptr.as_ref() } {
            debug_assert_eq!(child.is_level_0, self.hnsw_level.0 == 1);
            Some(child.get_file_index())
        } else {
            None
        };

        if let Some(FileIndex {
            offset,
            version_number,
            version_id,
        }) = parent_file_index
        {
            buf.extend(offset.0.to_le_bytes());
            buf.extend(version_number.to_le_bytes());
            buf.extend(version_id.to_le_bytes());
        } else {
            buf.extend([u8::MAX; 10]);
        }

        if let Some(FileIndex {
            offset,
            version_number,
            version_id,
        }) = child_file_index
        {
            buf.extend(offset.0.to_le_bytes());
            buf.extend(version_number.to_le_bytes());
            buf.extend(version_id.to_le_bytes());
        } else {
            buf.extend([u8::MAX; 10]);
        }

        if let Some(root) = unsafe { self.root_version.as_ref() } {
            debug_assert_eq!(root.is_level_0, is_level_0);
            let FileIndex {
                offset,
                version_number,
                version_id,
            } = root.get_file_index();
            buf.extend(offset.0.to_le_bytes());
            buf.extend(version_number.to_le_bytes());
            buf.extend(version_id.to_le_bytes());
        } else {
            buf.extend([u8::MAX; 10]);
        }

        bufman.update_with_cursor(cursor, &buf)?;

        #[cfg(debug_assertions)]
        {
            let current = bufman.cursor_position(cursor)?;

            assert_eq!(current, start_offset + 39);
        }

        {
            let _lock = self.lock_lowest_index();
            neighbors.serialize(bufmans, version, cursor)?;
        }
        self.versions.serialize(bufmans, version, cursor)?;

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError> {
        let FileIndex {
            version_id,
            version_number,
            offset: FileOffset(offset),
        } = file_index;
        let bufman = bufmans.get(version_id)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset as u64)?;
        // Read basic fields
        let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
        if is_level_0 {
            debug_assert_eq!(hnsw_level.0, 0);
        } else {
            debug_assert_ne!(hnsw_level.0, 0);
        }
        // Read prop
        let prop_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
        let prop_length = BytesToRead(bufman.read_u32_with_cursor(cursor)?);
        let prop = cache.get_prop(prop_offset, prop_length)?;

        let parent_offset = bufman.read_u32_with_cursor(cursor)?;
        let parent_version_number = bufman.read_u16_with_cursor(cursor)?;
        let parent_version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);

        let child_offset = bufman.read_u32_with_cursor(cursor)?;
        let child_version_number = bufman.read_u16_with_cursor(cursor)?;
        let child_version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);

        let root_version_offset = bufman.read_u32_with_cursor(cursor)?;
        let root_version_version_number = bufman.read_u16_with_cursor(cursor)?;
        let root_version_version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);
        bufman.close_cursor(cursor)?;
        // Deserialize parent
        let parent = if parent_offset != u32::MAX {
            SharedNode::deserialize(
                bufmans,
                FileIndex {
                    offset: FileOffset(parent_offset),
                    version_number: parent_version_number,
                    version_id: parent_version_id,
                },
                cache,
                max_loads,
                skipm,
                false,
            )?
        } else {
            ptr::null_mut()
        };
        // Deserialize child
        let child = if child_offset != u32::MAX {
            SharedNode::deserialize(
                bufmans,
                FileIndex {
                    offset: FileOffset(child_offset),
                    version_number: child_version_number,
                    version_id: child_version_id,
                },
                cache,
                max_loads,
                skipm,
                hnsw_level.0 == 1,
            )?
        } else {
            ptr::null_mut()
        };
        // Deserialize root_version
        let root_version = if root_version_offset != u32::MAX {
            SharedNode::deserialize(
                bufmans,
                FileIndex {
                    offset: FileOffset(root_version_offset),
                    version_number: root_version_version_number,
                    version_id: root_version_version_id,
                },
                cache,
                max_loads,
                skipm,
                hnsw_level.0 == 0,
            )?
        } else {
            ptr::null_mut()
        };

        let neighbors_file_index = FileIndex {
            offset: FileOffset(offset + 39),
            version_number,
            version_id,
        };

        let neighbors: Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> =
            HNSWIndexSerialize::deserialize(
                bufmans,
                neighbors_file_index,
                cache,
                max_loads,
                skipm,
                is_level_0,
            )?;

        let versions_file_index = FileIndex {
            offset: FileOffset(offset + 41 + neighbors.len() as u32 * 19),
            version_number,
            version_id,
        };

        let versions = ProbLazyItemArray::deserialize(
            bufmans,
            versions_file_index,
            cache,
            max_loads,
            skipm,
            is_level_0,
        )?;

        Ok(Self::new_with_neighbors_and_versions_and_root_version(
            hnsw_level,
            prop,
            neighbors,
            parent,
            child,
            versions,
            root_version,
            *cache.distance_metric.read().unwrap(),
        ))
    }
}
