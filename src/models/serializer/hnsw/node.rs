use std::sync::{atomic::Ordering, Arc};

use rustc_hash::FxHashSet;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager, FilelessBufferManager},
        cache_loader::HNSWIndexCache,
        lazy_item::FileIndex,
        prob_node::{Neighbors, ProbNode, SharedLatestNode},
        types::{BytesToRead, FileOffset, HNSWLevel, NodePropMetadata, NodePropValue},
        versioning::VersionNumber,
    },
};

use super::{HNSWIndexSerialize, RawDeserialize};

// @SERIALIZED_SIZE:
//   Properties:
//     1 byte for HNSW level +                           | 1
//     4 bytes for version +                             | 5
//     8 bytes for prop offset & length +                | 13
//     8 bytes for prop metadata offset & length         | 21
//
//   Links:
//     4 bytes for parent offset +                      | 4
//     4 bytes for child offset +                       | 8
//     2 bytes for neighbors length +                   | 10
//     neighbors length * 13 bytes for neighbor link +  | nb * 13 + 10
//
//   Total = nb * 13 + 31 (where `nb` is the neighbors count)
impl HNSWIndexSerialize for ProbNode {
    fn serialize(
        &self,
        bufman: &BufferManager,
        latest_version_links_bufman: &FilelessBufferManager,
        cursor: u64,
        latest_version_links_cursor: u64,
    ) -> Result<u32, BufIoError> {
        let start_offset = bufman.cursor_position(cursor)?;

        let neighbors = self.get_neighbors_raw();

        let mut buf = Vec::with_capacity(29);

        // Serialize basic fields
        buf.push(self.hnsw_level.0);

        buf.extend(self.version.load(Ordering::Relaxed).to_le_bytes());

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

        // Get parent offset
        let parent_offset =
            unsafe { parent_ptr.as_ref() }.map_or(u32::MAX, |parent| parent.file_offset.0);

        buf.extend(parent_offset.to_le_bytes());

        let child_ptr = self.get_child();

        // Get child offset
        let child_offset =
            unsafe { child_ptr.as_ref() }.map_or(u32::MAX, |child| child.file_offset.0);

        buf.extend(child_offset.to_le_bytes());

        bufman.update_with_cursor(cursor, &buf)?;

        #[cfg(debug_assertions)]
        {
            let current = bufman.cursor_position(cursor)?;
            assert_eq!(current, start_offset + 29);
        }

        {
            let _lock = self.freeze();
            neighbors.serialize(
                bufman,
                latest_version_links_bufman,
                cursor,
                latest_version_links_cursor,
            )?;
        }

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufman: &BufferManager,
        latest_version_links_bufman: &FilelessBufferManager,
        file_index: FileIndex<IndexFileId>,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut FxHashSet<u64>,
    ) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        let offset = file_index.offset.0;
        bufman.seek_with_cursor(cursor, offset as u64)?;
        // Read basic fields
        let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
        let version = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
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

        let parent_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
        let child_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);

        bufman.close_cursor(cursor)?;

        // Deserialize parent
        let parent_file_index = FileIndex {
            file_id: IndexFileId::invalid(),
            offset: parent_offset,
        };
        let parent = SharedLatestNode::deserialize(
            bufman,
            latest_version_links_bufman,
            parent_file_index,
            cache,
            max_loads,
            skipm,
        )?;
        // Deserialize child
        let child_file_index = FileIndex {
            file_id: IndexFileId::invalid(),
            offset: child_offset,
        };
        let child = SharedLatestNode::deserialize(
            bufman,
            latest_version_links_bufman,
            child_file_index,
            cache,
            max_loads,
            skipm,
        )?;

        let neighbors_file_index = FileIndex {
            // @NOTE: 29 = 21 (properties) + 4 (parent) + 4 (child)
            offset: FileOffset(offset + 29),
            file_id: file_index.file_id,
        };

        let neighbors = Neighbors::deserialize(
            bufman,
            latest_version_links_bufman,
            neighbors_file_index,
            cache,
            max_loads,
            skipm,
        )?;

        Ok(Self::new_with_neighbors_and_versions(
            hnsw_level,
            version,
            prop,
            metadata,
            neighbors,
            parent,
            child,
            *cache.distance_metric.read().unwrap(),
        ))
    }
}

impl RawDeserialize for ProbNode {
    type Raw = (
        HNSWLevel,
        VersionNumber,
        Arc<NodePropValue>,
        Option<Arc<NodePropMetadata>>,
        <Neighbors as RawDeserialize>::Raw,
        FileOffset,
        FileOffset,
    );

    fn deserialize_raw(
        bufman: &BufferManager,
        latest_version_links_bufman: &FilelessBufferManager,
        cursor: u64,
        latest_version_links_cursor: u64,
        FileOffset(offset): FileOffset,
        file_id: IndexFileId,
        cache: &HNSWIndexCache,
    ) -> Result<Self::Raw, BufIoError> {
        bufman.seek_with_cursor(cursor, offset as u64)?;
        // Read basic fields
        let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
        let version = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
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

        let parent_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
        let child_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);

        let neighbors = Neighbors::deserialize_raw(
            bufman,
            latest_version_links_bufman,
            cursor,
            latest_version_links_cursor,
            // @NOTE: 29 = 21 (properties) + 4 (parent) + 4 (child)
            FileOffset(offset + 29),
            file_id,
            cache,
        )?;

        Ok((
            hnsw_level,
            version,
            prop,
            metadata,
            neighbors,
            parent_offset,
            child_offset,
        ))
    }
}
