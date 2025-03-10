use crate::{
    models::{
        buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
        cache_loader::InvertedIndexCache,
        types::FileOffset,
    },
    storage::page::{Page, Pagepool},
};

use super::InvertedIndexSerialize;

impl<const LEN: usize> InvertedIndexSerialize for Pagepool<LEN> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let total_items = self.inner.len();
        if total_items == 0 {
            return Ok(u32::MAX);
        }
        let bufman = data_bufmans.get(data_file_idx)?;

        let start_offset = self.inner[0].serialize(
            dim_bufman,
            data_bufmans,
            data_file_idx,
            data_file_parts,
            cursor,
        )?;

        let mut prev_offset = start_offset;

        for item in self.inner.iter().skip(1) {
            let new_offset = item.serialize(
                dim_bufman,
                data_bufmans,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
            bufman.seek_with_cursor(cursor, prev_offset as u64)?;
            bufman.update_u32_with_cursor(cursor, new_offset)?;
            prev_offset = new_offset;
        }

        Ok(start_offset)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        data_file_parts: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        if file_offset.0 == u32::MAX {
            return Ok(Self::default());
        }
        let bufman = data_bufmans.get(data_file_idx)?;
        let cursor = bufman.open_cursor()?;

        let mut page_pool = Pagepool::<LEN>::default();
        let mut current_page_offset = file_offset.0;

        loop {
            bufman.seek_with_cursor(cursor, current_page_offset as u64)?;
            let next_page_offset = bufman.read_u32_with_cursor(cursor)?;
            let page = Page::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(current_page_offset),
                data_file_idx,
                data_file_parts,
                cache,
            )?;
            page_pool.inner.push(page);

            if next_page_offset == u32::MAX {
                break;
            }
            current_page_offset = next_page_offset;
        }
        bufman.close_cursor(cursor)?;
        Ok(page_pool)
    }
}
