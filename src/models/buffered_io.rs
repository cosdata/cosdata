use std::sync::{Arc, RwLock};
use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::fs::File;
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};

const BUFFER_SIZE: usize = 8192;
const FLUSH_THRESHOLD: usize = (BUFFER_SIZE as f32 * 0.7) as usize; // 70% of buffer size

struct BufferRegion {
    start: u64,
    buffer: RwLock<[u8; BUFFER_SIZE]>,
    dirty: AtomicBool,
    end: AtomicUsize,
}

impl BufferRegion {
    fn new(start: u64) -> Self {
        BufferRegion {
            start,
            buffer: RwLock::new([0; BUFFER_SIZE]),
            dirty: AtomicBool::new(false),
            end: AtomicUsize::new(0),
        }
    }

    fn should_flush(&self) -> bool {
        self.dirty.load(Ordering::SeqCst) && self.end.load(Ordering::SeqCst) >= FLUSH_THRESHOLD
    }
}

struct Cursor {
    position: u64,
}

impl Cursor {
    fn new() -> Self {
        Cursor { position: 0 }
    }
}

pub struct BufferManager {
    file: Arc<RwLock<File>>,
    regions: RwLock<BTreeMap<u64, Arc<BufferRegion>>>,
    cursors: RwLock<HashMap<u64, Cursor>>,
    next_cursor_id: AtomicU64,
    file_size: AtomicU64,
}

impl BufferManager {
    pub fn new(mut file: File) -> io::Result<Self> {
        let file_size = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;
        Ok(BufferManager {
            file: Arc::new(RwLock::new(file)),
            regions: RwLock::new(BTreeMap::new()),
            cursors: RwLock::new(HashMap::new()),
            next_cursor_id: AtomicU64::new(0),
            file_size: AtomicU64::new(file_size),
        })
    }

    pub fn open_cursor(&self) -> u64 {
        let cursor_id = self.next_cursor_id.fetch_add(1, Ordering::SeqCst);
        let mut cursors = self.cursors.write().unwrap();
        cursors.insert(cursor_id, Cursor::new());
        cursor_id
    }

    // @DOUBT: The caller will need to remember to call close_cursor,
    // other wise the cursors will keep accumulating. One way to
    // prevent that can be to implement Drop trait for the Cursor
    // struct. But for that, we'd need to have the cursors hashmap
    // inside an Arc so that a reference to it can be shared with the
    // Cursor struct. Then in the Cursor::drop method, the cursor can
    // be removed from the hashmap.
    pub fn close_cursor(&self, cursor_id: u64) {
        let mut cursors = self.cursors.write().unwrap();
        cursors.remove(&cursor_id);
    }

    fn get_or_create_region(&self, position: u64) -> io::Result<Arc<BufferRegion>> {
        let start = position - (position % BUFFER_SIZE as u64);
        let mut regions = self.regions.write().unwrap();

        if let Some(region) = regions.get(&start) {
            return Ok(Arc::clone(region));
        }

        // Create new region
        let mut region = BufferRegion::new(start);
        let mut file = self.file.write().unwrap();
        file.seek(SeekFrom::Start(start))?;
        let buffer = region.buffer.get_mut().unwrap();
        let bytes_read = file.read(&mut buffer[..])?;
        region.end.store(bytes_read, Ordering::SeqCst);

        let region = Arc::new(region);
        regions.insert(start, Arc::clone(&region));
        Ok(region)
    }

    fn flush_region(&self, region: &BufferRegion) -> io::Result<()> {
        let mut file = self.file.write().unwrap();
        file.seek(SeekFrom::Start(region.start))?;
        let buffer = region.buffer.read().unwrap();
        let end = region.end.load(Ordering::SeqCst);
        file.write_all(&buffer[..end])?;
        region.dirty.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn flush_region_if_needed(&self, region: &BufferRegion) -> io::Result<()> {
        if region.should_flush() {
            self.flush_region(region)?;
        }
        Ok(())
    }

    pub fn read_u32_with_cursor(&self, cursor_id: u64) -> io::Result<u32> {
        let mut buffer = [0u8; 4];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }

    pub fn read_u16_with_cursor(&self, cursor_id: u64) -> io::Result<u16> {
        let mut buffer = [0u8; 2];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(u16::from_le_bytes(buffer))
    }

    pub fn cursor_position(&self, cursor_id: u64) -> io::Result<u64> {
        let cursors = self.cursors.read().unwrap();
        cursors.get(&cursor_id)
            .map(|cursor| cursor.position)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid cursor"))
    }

    pub fn read_with_cursor(&self, cursor_id: u64, buf: &mut [u8]) -> io::Result<usize> {
        let mut curr_pos = {
            let cursors = self.cursors.read().unwrap();
            let cursor = cursors.get(&cursor_id)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid cursor"))?;
            cursor.position
        };

        let mut total_read = 0;
        while total_read < buf.len() {
            let region = self.get_or_create_region(curr_pos)?;
            let buffer = region.buffer.read().unwrap();
            let buffer_pos = (curr_pos - region.start) as usize;
            let available = region.end.load(Ordering::SeqCst) - buffer_pos;
            if available == 0 {
                if total_read == 0 && curr_pos >= self.file_size.load(Ordering::SeqCst) {
                    return Ok(0); // EOF
                }
                break;
            }
            let to_read = (buf.len() - total_read).min(available);
            buf[total_read..total_read + to_read].copy_from_slice(&buffer[buffer_pos..buffer_pos + to_read]);
            total_read += to_read;
            curr_pos += to_read as u64;
        }

        let mut cursors = self.cursors.write().unwrap();
        let cursor = cursors.get_mut(&cursor_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid cursor"))?;
        cursor.position = curr_pos;

        Ok(total_read)
    }

    pub fn write_with_cursor(&self, cursor_id: u64, buf: &[u8]) -> io::Result<usize> {
        let mut curr_pos = {
            let cursors = self.cursors.read().unwrap();
            let cursor = cursors.get(&cursor_id)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid cursor"))?;
            cursor.position
        };

        let mut total_written = 0;
        while total_written < buf.len() {
            let region = self.get_or_create_region(curr_pos)?;
            let mut buffer = region.buffer.write().unwrap();
            let buffer_pos = (curr_pos - region.start) as usize;
            let available = BUFFER_SIZE - buffer_pos;
            let to_write = (buf.len() - total_written).min(available);
            buffer[buffer_pos..buffer_pos + to_write].copy_from_slice(&buf[total_written..total_written + to_write]);
            region.end.store((buffer_pos + to_write).max(region.end.load(Ordering::SeqCst)), Ordering::SeqCst);
            region.dirty.store(true, Ordering::SeqCst);
            total_written += to_write;
            curr_pos += to_write as u64;
            self.file_size.fetch_max(curr_pos, Ordering::SeqCst);
            self.flush_region_if_needed(&region)?;
        }

        let mut cursors = self.cursors.write().unwrap();
        let cursor = cursors.get_mut(&cursor_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid cursor"))?;
        cursor.position = curr_pos;

        Ok(total_written)
    }

    // @DOUBT: How to ensure that the read/write_with_cursor functions
    // are not concurrently called for the same cursor? May be
    // self.cursors can be defined as `HashMap<u64,
    // RWLock<Cursor>>`. Then we can acquire a read lock on the
    // hashmap to identify and get the cursor from it. And then
    // acquire a write lock on the the value i.e. RWLock<Cursor> while
    // reading/writing is going on.

    pub fn seek_with_cursor(&self, cursor_id: u64, pos: SeekFrom) -> io::Result<u64> {
        let mut cursors = self.cursors.write().unwrap();
        let cursor = cursors.get_mut(&cursor_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid cursor"))?;

        let new_position = match pos {
            SeekFrom::Start(abs) => abs,
            SeekFrom::End(rel) => {
                (self.file_size.load(Ordering::SeqCst) as i64 + rel) as u64
            }
            SeekFrom::Current(rel) => {
                (cursor.position as i64 + rel) as u64
            }
        };

        cursor.position = new_position;
        Ok(new_position)
    }

    pub fn flush(&self) -> io::Result<()> {
        let regions = self.regions.read().unwrap();
        for region in regions.values() {
            if region.dirty.load(Ordering::SeqCst) {
                self.flush_region(region)?;
            }
        }
        self.file.write().unwrap().flush()
    }
}

#[cfg(test)]
mod tests {

    use std::thread;
    use tempfile::tempfile;
    use super::*;

    #[test]
    fn test_basic_usage() {
        // @NOTE: Not really using multiple threads here. This test
        // just ensures that read/write using multiple cursors work in
        // a single thread

        let mut file = tempfile().unwrap();
        file.write_all(&456_u32.to_le_bytes()).unwrap();
        let bufman = BufferManager::new(file).unwrap();

        let cursor1 = bufman.open_cursor();
        let cursor2 = bufman.open_cursor();

        // Thread 1: Read from the beginning of file
        bufman.seek_with_cursor(cursor1, SeekFrom::Start(0)).unwrap();
        let value1 = bufman.read_u32_with_cursor(cursor1).unwrap();
        assert_eq!(456_u32, value1);

        // Thread 2: Write to the end of file
        bufman.seek_with_cursor(cursor2, SeekFrom::End(0)).unwrap();
        bufman.write_with_cursor(cursor2, &789_u32.to_le_bytes()).unwrap();

        // Thread 1: Continue reading
        let value2 = bufman.read_u32_with_cursor(cursor1).unwrap();
        assert_eq!(789_u32, value2);

        // Thread 2: Again write to end of file
        bufman.write_with_cursor(cursor2, &12345_u32.to_le_bytes()).unwrap();

        // Thread 1: Continue reading
        let value3 = bufman.read_u32_with_cursor(cursor1).unwrap();
        assert_eq!(12345_u32, value3);

        bufman.close_cursor(cursor1);
        bufman.close_cursor(cursor1);
    }

    /// Create a large tmp content file about (5 * BUFFER_SIZE + 150)
    /// in size
    fn large_tmp_file() -> io::Result<File> {
        let mut file = tempfile()?;
        file.write_all(&vec![0_u8; (BUFFER_SIZE * 5) + 200])?;
        Ok(file)
    }

    fn file_offset(region: usize, region_offset: usize) -> u64 {
        if region_offset > BUFFER_SIZE {
            panic!("region_offset must be smaller than BUFFER_SIZE ({BUFFER_SIZE})");
        }
        (BUFFER_SIZE * (region - 1) + region_offset) as u64
    }

    #[test]
    fn test_reads_across_regions() {
        // Setup: Create a large tmp file and write 4 bytes into at
        // position 8190 i.e. BUFFER_SIZE - 2. This means first 2
        // bytes will be in region 1 and rest will be in region 2

        let mut file = large_tmp_file().unwrap();
        file.seek(SeekFrom::Start(8190)).unwrap();
        file.write_all(&1678_u32.to_le_bytes()).unwrap();

        let bufman = BufferManager::new(file).unwrap();

        let cursor1 = bufman.open_cursor();
        bufman.seek_with_cursor(cursor1, SeekFrom::Start(8190)).unwrap();
        let value1 = bufman.read_u32_with_cursor(cursor1).unwrap();
        assert_eq!(1678_u32, value1);

        bufman.close_cursor(cursor1);
    }


    #[test]
    fn test_conc_reads_different_regions() {
        let mut file = large_tmp_file().unwrap();

        // Write some data in region 1
        file.seek(SeekFrom::Start(100)).unwrap();
        file.write_all(&500_u16.to_le_bytes()).unwrap();

        // Write some data in region 3
        file.seek(SeekFrom::Start(file_offset(3, 147))).unwrap();
        file.write_all(&1000_u32.to_le_bytes()).unwrap();

        let bufman = Arc::new(BufferManager::new(file).unwrap());
        let t1 = {
            // Thread 1 that reads from region 1
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor();
                bm.seek_with_cursor(cid, SeekFrom::Start(100)).unwrap();
                let v = bm.read_u16_with_cursor(cid).unwrap();
                assert_eq!(500_u16, v);
                bm.close_cursor(cid);
            })
        };

        let t2 = {
            // Thread 2 that reads from region 3
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor();
                bm.seek_with_cursor(cid, SeekFrom::Start(file_offset(3, 147))).unwrap();
                let v = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(1000_u32, v);
                bm.close_cursor(cid);
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();
    }

    #[test]
    fn test_conc_reads_same_region() {
        let mut file = large_tmp_file().unwrap();

        // Write some data in region 1
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&500_u16.to_le_bytes()).unwrap();

        file.seek(SeekFrom::Start(2)).unwrap();
        file.write_all(&1000_u32.to_le_bytes()).unwrap();

        file.seek(SeekFrom::Start(6)).unwrap();
        file.write_all(&145_u32.to_le_bytes()).unwrap();

        // Thread 1 will read first two bytes in region 1
        let bufman = Arc::new(BufferManager::new(file).unwrap());
        let t1 = {
            // Thread 1 that reads from region 1
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor();
                bm.seek_with_cursor(cid, SeekFrom::Start(0)).unwrap();
                let v = bm.read_u16_with_cursor(cid).unwrap();
                assert_eq!(500_u16, v);
                bm.close_cursor(cid);
            })
        };

        // Thread 2 will read first 6 bytes in region 1
        let t2 = {
            // Thread 1 that reads from region 1
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor();
                bm.seek_with_cursor(cid, SeekFrom::Start(0)).unwrap();
                let v1 = bm.read_u16_with_cursor(cid).unwrap();
                assert_eq!(500_u16, v1);
                let v2 = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(1000_u32, v2);
                bm.close_cursor(cid);
            })
        };

        // Thread 3 will read last 4 bytes in region 1
        let t3 = {
            // Thread 1 that reads from region 1
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor();
                bm.seek_with_cursor(cid, SeekFrom::Start(6)).unwrap();
                let v = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(145_u32, v);
                bm.close_cursor(cid);
            })
        };

        for t in vec![t1, t2, t3] {
            t.join().unwrap();
        }
    }
}
