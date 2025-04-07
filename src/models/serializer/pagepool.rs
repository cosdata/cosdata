use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    page::{Page, Pagepool},
    types::FileOffset,
};

use super::SimpleSerialize;

impl<const LEN: usize> SimpleSerialize for Pagepool<LEN> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let inner = self.inner.read().unwrap();
        let total_items = inner.len();
        if total_items == 0 {
            return Ok(u32::MAX);
        }

        let start_offset = inner[0].serialize(bufman, cursor)?;

        let mut prev_offset = start_offset;

        for item in inner.iter().skip(1) {
            let new_offset = item.serialize(bufman, cursor)?;
            bufman.seek_with_cursor(cursor, prev_offset as u64)?;
            bufman.update_u32_with_cursor(cursor, new_offset)?;
            prev_offset = new_offset;
        }

        Ok(start_offset)
    }

    fn deserialize(bufman: &BufferManager, file_offset: FileOffset) -> Result<Self, BufIoError> {
        if file_offset.0 == u32::MAX {
            return Ok(Self::default());
        }
        let cursor = bufman.open_cursor()?;

        let page_pool = Pagepool::<LEN>::default();
        let mut current_page_offset = file_offset.0;

        loop {
            bufman.seek_with_cursor(cursor, current_page_offset as u64)?;
            let next_page_offset = bufman.read_u32_with_cursor(cursor)?;
            let page = Page::deserialize(bufman, FileOffset(current_page_offset))?;
            page_pool.inner.write().unwrap().push(page);

            if next_page_offset == u32::MAX {
                break;
            }
            current_page_offset = next_page_offset;
        }
        bufman.close_cursor(cursor)?;
        Ok(page_pool)
    }
}
