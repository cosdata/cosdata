use std::{ptr, sync::Arc};

use rustc_hash::FxHashSet;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager},
        cache_loader::HNSWIndexCache,
        prob_lazy_load::lazy_item::FileIndex,
        prob_node::{Neighbors, ProbNode, SharedNode},
        types::{
            BytesToRead, FileOffset, HNSWLevel, InternalId, MetricResult, NodePropMetadata,
            NodePropValue,
        },
        versioning::VersionHash,
    },
};

use super::{HNSWIndexSerialize, RawDeserialize};

// @SERIALIZED_SIZE:
//   Properties:
//     1 byte for HNSW level +                           | 1
//     8 bytes for version +                             | 9
//     8 bytes for prop offset & length +                | 17
//     8 bytes for prop metadata offset & length         | 17 + 8 = 25
//
//   Links:
//     8 bytes for parent offset & file id +            | 8
//     8 bytes for child offset & file id +             | 16
//     1 byte for root version tag +                    | 17
//     8 bytes for root version offset & file id +      | 25
//     2 bytes for neighbors length +                   | 27
//     neighbors length * 17 bytes for neighbor link +  | nb * 17 + 27
//
//   Total = nb * 17 + 52 (where `nb` is the neighbors count)
impl HNSWIndexSerialize for ProbNode {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let start_offset = bufman.cursor_position(cursor)?;

        let neighbors = self.get_neighbors_raw();

        let mut buf = Vec::with_capacity(50);

        // Serialize basic fields
        buf.push(self.hnsw_level.0);

        buf.extend(self.version.to_le_bytes());

        // Serialize prop_value
        let (FileOffset(offset), BytesToRead(length)) = &self.prop_value.location;
        buf.extend(offset.to_le_bytes());
        buf.extend(length.to_le_bytes());

        // Serialize prop_metadata
        if let Some(prop_metadata) = &self.prop_metadata {
            let (FileOffset(offset), BytesToRead(length)) = prop_metadata.location;
            buf.extend(offset.to_le_bytes());
            buf.extend(length.to_le_bytes());
        } else {
            buf.extend([u8::MAX; 8]);
        }

        let parent_ptr = self.get_parent();

        // Get parent file index
        let parent_file_index = unsafe { parent_ptr.as_ref() }.map(|parent| parent.file_index);

        let child_ptr = self.get_child();

        // Get child file index
        let child_file_index = unsafe { child_ptr.as_ref() }.map(|child| child.file_index);

        if let Some(FileIndex { offset, file_id }) = parent_file_index {
            buf.extend(offset.0.to_le_bytes());
            buf.extend(file_id.to_le_bytes());
        } else {
            buf.extend([u8::MAX; 8]);
        }

        if let Some(FileIndex { offset, file_id }) = child_file_index {
            buf.extend(offset.0.to_le_bytes());
            buf.extend(file_id.to_le_bytes());
        } else {
            buf.extend([u8::MAX; 8]);
        }

        let (root, is_root) = *self.root_version.read();

        if let Some(root) = unsafe { root.as_ref() } {
            let FileIndex { offset, file_id } = root.file_index;
            buf.push(is_root as u8);
            buf.extend(offset.0.to_le_bytes());
            buf.extend(file_id.to_le_bytes());
        } else {
            buf.extend([u8::MAX; 9]);
        }

        bufman.update_with_cursor(cursor, &buf)?;

        #[cfg(debug_assertions)]
        {
            let current = bufman.cursor_position(cursor)?;
            assert_eq!(current, start_offset + 50);
        }

        {
            let _lock = self.freeze();
            neighbors.serialize(bufman, cursor)?;
        }

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufman: &BufferManager,
        file_index: FileIndex,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut FxHashSet<u64>,
    ) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        let offset = file_index.offset.0;
        bufman.seek_with_cursor(cursor, offset as u64)?;
        // Read basic fields
        let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
        let version = VersionHash::from(bufman.read_u64_with_cursor(cursor)?);
        // Read prop_value
        let prop_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
        let prop_length = BytesToRead(bufman.read_u32_with_cursor(cursor)?);
        let prop = cache.get_prop(prop_offset, prop_length)?;

        // Read prop_metadata
        let metadata_offset = bufman.read_u32_with_cursor(cursor)?;
        let metadata_length = bufman.read_u32_with_cursor(cursor)?;
        let metadata = if metadata_offset != u32::MAX {
            Some(
                cache
                    .get_prop_metadata(FileOffset(metadata_offset), BytesToRead(metadata_length))?,
            )
        } else {
            None
        };

        let parent_offset = bufman.read_u32_with_cursor(cursor)?;
        let parent_file_id = IndexFileId::from(bufman.read_u32_with_cursor(cursor)?);

        let child_offset = bufman.read_u32_with_cursor(cursor)?;
        let child_file_id = IndexFileId::from(bufman.read_u32_with_cursor(cursor)?);

        let root_tag = bufman.read_u8_with_cursor(cursor)? != 0;
        let root_offset = bufman.read_u32_with_cursor(cursor)?;
        let root_file_id = IndexFileId::from(bufman.read_u32_with_cursor(cursor)?);

        bufman.close_cursor(cursor)?;

        // Deserialize parent
        let parent = if parent_offset != u32::MAX {
            let parent_file_index = FileIndex {
                offset: FileOffset(parent_offset),
                file_id: parent_file_id,
            };
            SharedNode::deserialize(bufman, parent_file_index, cache, max_loads, skipm)?
        } else {
            ptr::null_mut()
        };
        // Deserialize child
        let child = if child_offset != u32::MAX {
            let child_file_index = FileIndex {
                offset: FileOffset(child_offset),
                file_id: child_file_id,
            };
            SharedNode::deserialize(bufman, child_file_index, cache, max_loads, skipm)?
        } else {
            ptr::null_mut()
        };
        // Deserialize root version
        let root_version = if root_offset != u32::MAX {
            let root_file_index = FileIndex {
                offset: FileOffset(root_offset),
                file_id: root_file_id,
            };
            SharedNode::deserialize(bufman, root_file_index, cache, max_loads, skipm)?
        } else {
            ptr::null_mut()
        };

        let neighbors_file_index = FileIndex {
            // @NOTE: 50 = 25 (properties) + 8 (parent) + 8 (child) + 9 (root)
            offset: FileOffset(offset + 50),
            file_id: file_index.file_id,
        };

        let neighbors =
            Neighbors::deserialize(bufman, neighbors_file_index, cache, max_loads, skipm)?;

        Ok(Self::new_with_neighbors_and_versions_and_root_version(
            hnsw_level,
            version,
            prop,
            metadata,
            neighbors,
            parent,
            child,
            root_version,
            root_tag,
            *cache.distance_metric.read().unwrap(),
        ))
    }
}

impl RawDeserialize for ProbNode {
    type Raw = (
        HNSWLevel,
        VersionHash,
        Arc<NodePropValue>,
        Option<Arc<NodePropMetadata>>,
        Vec<Option<(InternalId, FileIndex, MetricResult)>>,
        Option<FileIndex>,
        Option<FileIndex>,
        Option<FileIndex>,
        bool,
    );

    fn deserialize_raw(
        bufman: &BufferManager,
        cursor: u64,
        FileOffset(offset): FileOffset,
        file_id: IndexFileId,
        cache: &HNSWIndexCache,
    ) -> Result<Self::Raw, BufIoError> {
        bufman.seek_with_cursor(cursor, offset as u64)?;
        // Read basic fields
        let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
        let version = VersionHash::from(bufman.read_u64_with_cursor(cursor)?);
        // Read prop_value
        let prop_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
        let prop_length = BytesToRead(bufman.read_u32_with_cursor(cursor)?);
        let prop = cache.get_prop(prop_offset, prop_length)?;

        // Read prop_metadata
        let metadata_offset = bufman.read_u32_with_cursor(cursor)?;
        let metadata_length = bufman.read_u32_with_cursor(cursor)?;
        let metadata = if metadata_offset != u32::MAX {
            Some(
                cache
                    .get_prop_metadata(FileOffset(metadata_offset), BytesToRead(metadata_length))?,
            )
        } else {
            None
        };

        let parent_offset = bufman.read_u32_with_cursor(cursor)?;
        let parent_file_id = IndexFileId::from(bufman.read_u32_with_cursor(cursor)?);

        let child_offset = bufman.read_u32_with_cursor(cursor)?;
        let child_file_id = IndexFileId::from(bufman.read_u32_with_cursor(cursor)?);

        let root_tag = bufman.read_u8_with_cursor(cursor)? != 0;
        let root_offset = bufman.read_u32_with_cursor(cursor)?;
        let root_file_id = IndexFileId::from(bufman.read_u32_with_cursor(cursor)?);

        let parent = (parent_offset != u32::MAX).then_some(FileIndex {
            offset: FileOffset(parent_offset),
            file_id: parent_file_id,
        });

        let child = (child_offset != u32::MAX).then_some(FileIndex {
            offset: FileOffset(child_offset),
            file_id: child_file_id,
        });

        let root = (root_offset != u32::MAX).then_some(FileIndex {
            offset: FileOffset(root_offset),
            file_id: root_file_id,
        });

        let neighbors = Neighbors::deserialize_raw(
            bufman,
            cursor,
            // @NOTE: 50 = 25 (properties) + 8 (parent) + 8 (child) + 9 (root)
            FileOffset(offset + 50),
            file_id,
            cache,
        )?;

        Ok((
            hnsw_level, version, prop, metadata, neighbors, parent, child, root, root_tag,
        ))
    }
}
