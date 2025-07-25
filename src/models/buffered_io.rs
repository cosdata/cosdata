use dashmap::DashMap;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::{fmt, fs};

use super::lru_cache::LRUCache;
use super::versioning::VersionNumber;

#[derive(Debug)]
pub enum BufIoError {
    Io(io::Error),
    Locking,
    InvalidCursor(u64),
}

impl From<io::Error> for BufIoError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl fmt::Display for BufIoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "IO error: {}", error),
            Self::Locking => f.write_str("Locking error"),
            Self::InvalidCursor(cursor) => write!(f, "Invalid cursor `{}`", cursor),
        }
    }
}

struct BufferRegion {
    start: u64,
    buffer: RwLock<Vec<u8>>,
    dirty: AtomicBool,
    end: AtomicUsize,
    file: Arc<RwLock<File>>,
}

impl BufferRegion {
    fn new(start: u64, file: Arc<RwLock<File>>, buffer_size: usize) -> Self {
        BufferRegion {
            start,
            buffer: RwLock::new(vec![0; buffer_size]),
            dirty: AtomicBool::new(false),
            end: AtomicUsize::new(0),
            file,
        }
    }

    fn should_final_flush(&self) -> bool {
        self.dirty.load(Ordering::SeqCst)
    }

    fn flush(&self) -> Result<(), BufIoError> {
        let mut file = self.file.write().map_err(|_| BufIoError::Locking)?;
        file.seek(SeekFrom::Start(self.start))
            .map_err(BufIoError::Io)?;
        let buffer = self.buffer.read().map_err(|_| BufIoError::Locking)?;
        let end = self.end.load(Ordering::SeqCst);
        file.write_all(&buffer[..end]).map_err(BufIoError::Io)?;
        self.dirty.store(false, Ordering::SeqCst);
        Ok(())
    }
}

impl Drop for BufferRegion {
    fn drop(&mut self) {
        if self.should_final_flush() {
            self.flush().unwrap();
        }
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

pub struct BufferManagerFactory<K> {
    bufmans: Arc<DashMap<K, Arc<BufferManager>>>,
    root_path: Arc<Path>,
    path_function: fn(&Path, &K) -> PathBuf,
    buffer_size: usize,
}

impl<K: Hash + Eq + Clone> BufferManagerFactory<K> {
    pub fn new(
        root_path: Arc<Path>,
        path_function: fn(&Path, &K) -> PathBuf,
        buffer_size: usize,
    ) -> Self {
        Self {
            bufmans: Arc::new(DashMap::new()),
            root_path,
            path_function,
            buffer_size,
        }
    }

    pub fn get(&self, key: K) -> Result<Arc<BufferManager>, BufIoError> {
        self.bufmans
            .entry(key.clone())
            .or_try_insert_with(|| {
                let path = (self.path_function)(&self.root_path, &key);

                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&path)?;
                let bufman = Arc::new(BufferManager::new(file, self.buffer_size)?);

                Ok(bufman)
            })
            .map(|bufman_ref| bufman_ref.value().clone())
    }

    pub fn flush_all(&self) -> Result<(), BufIoError> {
        for bufman in self.bufmans.iter() {
            bufman.flush()?;
        }
        Ok(())
    }
}

pub struct BufferManager {
    file: Arc<RwLock<File>>,
    regions: LRUCache<u64, Arc<BufferRegion>>,
    cursors: RwLock<HashMap<u64, Cursor>>,
    next_cursor_id: AtomicU64,
    file_size: RwLock<u64>,
    buffer_size: usize,
}

impl BufferManager {
    pub fn new(mut file: File, buffer_size: usize) -> io::Result<Self> {
        let file_size = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;
        let regions = LRUCache::with_prob_eviction(10000, 0.03125);
        let mut this = Self {
            file: Arc::new(RwLock::new(file)),
            regions,
            cursors: RwLock::new(HashMap::new()),
            next_cursor_id: AtomicU64::new(0),
            file_size: RwLock::new(file_size),
            buffer_size,
        };
        this.regions.set_evict_hook(Some(|region| {
            if region.should_final_flush() {
                region.flush().unwrap();
            }
        }));
        Ok(this)
    }

    pub fn open_cursor(&self) -> Result<u64, BufIoError> {
        let cursor_id = self.next_cursor_id.fetch_add(1, Ordering::SeqCst);
        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        cursors.insert(cursor_id, Cursor::new());
        Ok(cursor_id)
    }

    // @DOUBT: The caller will need to remember to call close_cursor,
    // other wise the cursors will keep accumulating. One way to
    // prevent that can be to implement Drop trait for the Cursor
    // struct. But for that, we'd need to have the cursors hashmap
    // inside an Arc so that a reference to it can be shared with the
    // Cursor struct. Then in the Cursor::drop method, the cursor can
    // be removed from the hashmap.
    pub fn close_cursor(&self, cursor_id: u64) -> Result<(), BufIoError> {
        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        cursors.remove(&cursor_id);
        Ok(())
    }

    fn get_or_create_region(&self, position: u64) -> Result<Arc<BufferRegion>, BufIoError> {
        let start = position - (position % self.buffer_size as u64);
        let cached_region = self.regions.get_or_insert::<BufIoError>(start, || {
            let mut region = BufferRegion::new(start, self.file.clone(), self.buffer_size);
            let mut file = self.file.write().map_err(|_| BufIoError::Locking)?;
            file.seek(SeekFrom::Start(start)).map_err(BufIoError::Io)?;
            let buffer = region.buffer.get_mut().map_err(|_| BufIoError::Locking)?;
            let bytes_read = file.read(&mut buffer[..]).map_err(BufIoError::Io)?;
            region.end.store(bytes_read, Ordering::SeqCst);
            Ok(Arc::new(region))
        });
        cached_region.map(|r| r.inner())
    }

    pub fn read_f32_with_cursor(&self, cursor_id: u64) -> Result<f32, BufIoError> {
        let mut buffer = [0u8; 4];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(f32::from_le_bytes(buffer))
    }

    pub fn read_i64_with_cursor(&self, cursor_id: u64) -> Result<i64, BufIoError> {
        let mut buffer = [0u8; 8];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(i64::from_le_bytes(buffer))
    }

    pub fn read_i32_with_cursor(&self, cursor_id: u64) -> Result<i32, BufIoError> {
        let mut buffer = [0u8; 4];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(i32::from_le_bytes(buffer))
    }

    pub fn read_u64_with_cursor(&self, cursor_id: u64) -> Result<u64, BufIoError> {
        let mut buffer = [0u8; 8];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(u64::from_le_bytes(buffer))
    }

    pub fn read_u32_with_cursor(&self, cursor_id: u64) -> Result<u32, BufIoError> {
        let mut buffer = [0u8; 4];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }

    pub fn read_u16_with_cursor(&self, cursor_id: u64) -> Result<u16, BufIoError> {
        let mut buffer = [0u8; 2];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(u16::from_le_bytes(buffer))
    }

    pub fn read_u8_with_cursor(&self, cursor_id: u64) -> Result<u8, BufIoError> {
        let mut buffer = [0u8; 1];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(u8::from_le_bytes(buffer))
    }

    pub fn cursor_position(&self, cursor_id: u64) -> Result<u64, BufIoError> {
        let cursors = self.cursors.read().map_err(|_| BufIoError::Locking)?;
        cursors
            .get(&cursor_id)
            .map(|cursor| cursor.position)
            .ok_or(BufIoError::InvalidCursor(cursor_id))
    }

    pub fn read_with_cursor(&self, cursor_id: u64, buf: &mut [u8]) -> Result<usize, BufIoError> {
        let mut curr_pos = {
            let cursors = self.cursors.read().map_err(|_| BufIoError::Locking)?;
            let cursor = cursors
                .get(&cursor_id)
                .ok_or(BufIoError::InvalidCursor(cursor_id))?;
            cursor.position
        };

        let mut total_read = 0;
        while total_read < buf.len() {
            let region = self.get_or_create_region(curr_pos)?;
            let buffer = region.buffer.read().map_err(|_| BufIoError::Locking)?;
            let buffer_pos = (curr_pos - region.start) as usize;
            let available = region.end.load(Ordering::SeqCst) - buffer_pos;
            if available == 0 {
                if total_read == 0
                    && curr_pos >= *self.file_size.read().map_err(|_| BufIoError::Locking)?
                {
                    return Ok(0); // EOF
                }
                break;
            }
            let to_read = (buf.len() - total_read).min(available);
            buf[total_read..total_read + to_read]
                .copy_from_slice(&buffer[buffer_pos..buffer_pos + to_read]);
            total_read += to_read;
            curr_pos += to_read as u64;
        }

        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        let cursor = cursors
            .get_mut(&cursor_id)
            .ok_or(BufIoError::InvalidCursor(cursor_id))?;
        cursor.position = curr_pos;

        Ok(total_read)
    }

    pub fn update_f32_with_cursor(&self, cursor_id: u64, value: f32) -> Result<u64, BufIoError> {
        let buffer = value.to_le_bytes();
        self.write_with_cursor(cursor_id, &buffer, false)
    }

    pub fn update_u64_with_cursor(&self, cursor_id: u64, value: u64) -> Result<u64, BufIoError> {
        let buffer = value.to_le_bytes();
        self.write_with_cursor(cursor_id, &buffer, false)
    }

    pub fn update_u32_with_cursor(&self, cursor_id: u64, value: u32) -> Result<u64, BufIoError> {
        let buffer = value.to_le_bytes();
        self.write_with_cursor(cursor_id, &buffer, false)
    }

    pub fn update_u16_with_cursor(&self, cursor_id: u64, value: u16) -> Result<u64, BufIoError> {
        let buffer = value.to_le_bytes();
        self.write_with_cursor(cursor_id, &buffer, false)
    }

    pub fn update_u8_with_cursor(&self, cursor_id: u64, value: u8) -> Result<u64, BufIoError> {
        let buffer = value.to_le_bytes();
        self.write_with_cursor(cursor_id, &buffer, false)
    }

    pub fn update_with_cursor(&self, cursor_id: u64, buf: &[u8]) -> Result<u64, BufIoError> {
        self.write_with_cursor(cursor_id, buf, false)
    }

    fn write_with_cursor(
        &self,
        cursor_id: u64,
        buf: &[u8],
        append: bool,
    ) -> Result<u64, BufIoError> {
        let curr_pos = {
            let cursors = self.cursors.read().map_err(|_| BufIoError::Locking)?;
            let cursor = cursors
                .get(&cursor_id)
                .ok_or(BufIoError::InvalidCursor(cursor_id))?;
            cursor.position
        };

        let input_size = buf.len();

        // Take write lock early to cover the entire write operation that might affect file size
        let mut file_size_guard = self.file_size.write().map_err(|_| BufIoError::Locking)?;
        let will_cross_eof = append || curr_pos + input_size as u64 >= *file_size_guard;
        let mut curr_pos = if append {
            curr_pos.max(*file_size_guard)
        } else {
            curr_pos
        };
        let start_pos = curr_pos;

        if will_cross_eof {
            let mut total_written = 0;
            while total_written < input_size {
                let region = self.get_or_create_region(curr_pos)?;
                {
                    let mut buffer = region.buffer.write().map_err(|_| BufIoError::Locking)?;
                    let buffer_pos = (curr_pos - region.start) as usize;
                    let available = self.buffer_size - buffer_pos;
                    let to_write = (input_size - total_written).min(available);
                    buffer[buffer_pos..buffer_pos + to_write]
                        .copy_from_slice(&buf[total_written..total_written + to_write]);
                    region.end.store(
                        (buffer_pos + to_write).max(region.end.load(Ordering::SeqCst)),
                        Ordering::SeqCst,
                    );
                    region.dirty.store(true, Ordering::SeqCst);
                    total_written += to_write;
                    curr_pos += to_write as u64;
                }
            }
            // Update file size using max to prevent shrinking
            *file_size_guard = (*file_size_guard).max(curr_pos);
        } else {
            // Normal write within existing file bounds
            let mut total_written = 0;
            while total_written < input_size {
                let region = self.get_or_create_region(curr_pos)?;
                {
                    let mut buffer = region.buffer.write().map_err(|_| BufIoError::Locking)?;
                    let buffer_pos = (curr_pos - region.start) as usize;
                    let available = self.buffer_size - buffer_pos;
                    let to_write = (input_size - total_written).min(available);
                    buffer[buffer_pos..buffer_pos + to_write]
                        .copy_from_slice(&buf[total_written..total_written + to_write]);
                    region.end.store(
                        (buffer_pos + to_write).max(region.end.load(Ordering::SeqCst)),
                        Ordering::SeqCst,
                    );
                    region.dirty.store(true, Ordering::SeqCst);
                    total_written += to_write;
                    curr_pos += to_write as u64;
                }
            }
            // Even for normal writes, we need to check if we accidentally crossed EOF
            *file_size_guard = (*file_size_guard).max(curr_pos);
        }

        // Drop file_size_guard before updating cursor
        drop(file_size_guard);

        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        let cursor = cursors
            .get_mut(&cursor_id)
            .ok_or(BufIoError::InvalidCursor(cursor_id))?;
        cursor.position = curr_pos;

        Ok(start_pos)
    }

    pub fn write_to_end_of_file(&self, cursor_id: u64, buf: &[u8]) -> Result<u64, BufIoError> {
        self.write_with_cursor(cursor_id, buf, true)
    }

    // @DOUBT: How to ensure that the read/write_with_cursor functions
    // are not concurrently called for the same cursor? May be
    // self.cursors can be defined as `HashMap<u64,
    // RWLock<Cursor>>`. Then we can acquire a read lock on the
    // hashmap to identify and get the cursor from it. And then
    // acquire a write lock on the the value i.e. RWLock<Cursor> while
    // reading/writing is going on.

    pub fn seek_with_cursor(&self, cursor_id: u64, pos: u64) -> Result<(), BufIoError> {
        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        let cursor = cursors
            .get_mut(&cursor_id)
            .ok_or(BufIoError::InvalidCursor(cursor_id))?;

        cursor.position = pos;
        Ok(())
    }

    pub fn flush(&self) -> Result<(), BufIoError> {
        for region in self.regions.values() {
            if region.should_final_flush() {
                region.flush()?;
            }
        }
        self.file
            .write()
            .map_err(|_| BufIoError::Locking)?
            .flush()
            .map_err(BufIoError::Io)
    }

    pub fn file_size(&self) -> u64 {
        *self.file_size.read().unwrap()
    }
}

pub struct FilelessBufferManager {
    regions: DashMap<u64, Arc<FilelessBufferRegion>>,
    cursors: RwLock<HashMap<u64, Cursor>>,
    next_cursor_id: AtomicU64,
    file_size: RwLock<u64>,
    buffer_size: usize,
}

struct FilelessBufferRegion {
    start: u64,
    buffer: RwLock<Vec<u8>>,
    dirty: AtomicBool,
    end: AtomicUsize,
}

impl FilelessBufferRegion {
    fn new(start: u64, buffer_size: usize) -> Self {
        Self {
            start,
            buffer: RwLock::new(vec![0; buffer_size]),
            dirty: AtomicBool::new(false),
            end: AtomicUsize::new(0),
        }
    }

    fn should_flush(&self) -> bool {
        self.dirty.load(Ordering::SeqCst)
    }

    fn flush(&self, file: &mut File) -> Result<(), BufIoError> {
        file.seek(SeekFrom::Start(self.start))
            .map_err(BufIoError::Io)?;
        let buffer = self.buffer.read().map_err(|_| BufIoError::Locking)?;
        let end = self.end.load(Ordering::SeqCst);
        file.write_all(&buffer[..end]).map_err(BufIoError::Io)?;
        self.dirty.store(false, Ordering::SeqCst);
        Ok(())
    }
}

impl FilelessBufferManager {
    pub fn new(buffer_size: usize) -> Result<Self, BufIoError> {
        Ok(Self {
            regions: DashMap::new(),
            cursors: RwLock::new(HashMap::new()),
            next_cursor_id: AtomicU64::new(0),
            file_size: RwLock::new(0),
            buffer_size,
        })
    }

    pub fn from_file(file: &mut File, buffer_size: usize) -> Result<Self, BufIoError> {
        let file_size = file.seek(SeekFrom::End(0))?;

        let mut offset = 0;
        let regions = DashMap::new();

        while offset < file_size {
            let mut region = FilelessBufferRegion::new(offset, buffer_size);
            file.seek(SeekFrom::Start(offset))?;
            let buffer = region.buffer.get_mut().map_err(|_| BufIoError::Locking)?;
            let bytes_read = file.read(&mut buffer[..]).map_err(BufIoError::Io)?;
            region.end.store(bytes_read, Ordering::SeqCst);
            regions.insert(offset, Arc::new(region));
            offset += buffer_size as u64;
        }

        file.seek(SeekFrom::Start(0))?;

        Ok(Self {
            regions,
            cursors: RwLock::new(HashMap::new()),
            next_cursor_id: AtomicU64::new(0),
            file_size: RwLock::new(file_size),
            buffer_size,
        })
    }

    pub fn from_versioned(
        buffer_size: usize,
        root_path: &Path,
        mapping_fn: impl Fn(&Path) -> Option<(VersionNumber, u64)>,
        latest_version: VersionNumber,
    ) -> Result<Self, BufIoError> {
        let regions = DashMap::new();
        let mut file_size = 0;
        let mut region_versions_map = FxHashMap::<u64, (VersionNumber, PathBuf)>::default();

        for entry in fs::read_dir(root_path)? {
            let entry = entry?;
            let path = entry.path();
            let Some((version, region_id)) = mapping_fn(&path) else {
                continue;
            };
            if *version > *latest_version {
                continue;
            }
            if let Some((existing_version, _)) = region_versions_map.get(&region_id) {
                if **existing_version > *version {
                    continue;
                }
            }
            region_versions_map.insert(region_id, (version, path));
        }

        for (region_id, (_, path)) in region_versions_map {
            let offset = region_id * buffer_size as u64;
            let mut file = OpenOptions::new().read(true).open(path)?;
            let mut region = FilelessBufferRegion::new(offset, buffer_size);
            let buffer = region.buffer.get_mut().map_err(|_| BufIoError::Locking)?;
            let bytes_read = file.read(&mut buffer[..]).map_err(BufIoError::Io)?;
            region.end.store(bytes_read, Ordering::SeqCst);
            regions.insert(offset, Arc::new(region));
            let end = offset + bytes_read as u64;
            file_size = file_size.max(end);
        }

        Ok(Self {
            regions,
            cursors: RwLock::new(HashMap::new()),
            next_cursor_id: AtomicU64::new(0),
            file_size: RwLock::new(file_size),
            buffer_size,
        })
    }

    pub fn open_cursor(&self) -> Result<u64, BufIoError> {
        let cursor_id = self.next_cursor_id.fetch_add(1, Ordering::SeqCst);
        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        cursors.insert(cursor_id, Cursor::new());
        Ok(cursor_id)
    }

    pub fn close_cursor(&self, cursor_id: u64) -> Result<(), BufIoError> {
        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        cursors.remove(&cursor_id);
        Ok(())
    }

    fn get_or_create_region(&self, position: u64) -> Arc<FilelessBufferRegion> {
        let start = position - (position % self.buffer_size as u64);
        self.regions
            .entry(start)
            .or_insert_with(|| Arc::new(FilelessBufferRegion::new(start, self.buffer_size)))
            .clone()
    }

    pub fn read_f32_with_cursor(&self, cursor_id: u64) -> Result<f32, BufIoError> {
        let mut buffer = [0u8; 4];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(f32::from_le_bytes(buffer))
    }

    pub fn read_i32_with_cursor(&self, cursor_id: u64) -> Result<i32, BufIoError> {
        let mut buffer = [0u8; 4];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(i32::from_le_bytes(buffer))
    }

    pub fn read_u32_with_cursor(&self, cursor_id: u64) -> Result<u32, BufIoError> {
        let mut buffer = [0u8; 4];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }

    pub fn read_u8_with_cursor(&self, cursor_id: u64) -> Result<u8, BufIoError> {
        let mut buffer = [0u8; 1];
        self.read_with_cursor(cursor_id, &mut buffer)?;
        Ok(u8::from_le_bytes(buffer))
    }

    pub fn cursor_position(&self, cursor_id: u64) -> Result<u64, BufIoError> {
        let cursors = self.cursors.read().map_err(|_| BufIoError::Locking)?;
        cursors
            .get(&cursor_id)
            .map(|cursor| cursor.position)
            .ok_or(BufIoError::InvalidCursor(cursor_id))
    }

    pub fn read_with_cursor(&self, cursor_id: u64, buf: &mut [u8]) -> Result<usize, BufIoError> {
        let mut curr_pos = {
            let cursors = self.cursors.read().map_err(|_| BufIoError::Locking)?;
            let cursor = cursors
                .get(&cursor_id)
                .ok_or(BufIoError::InvalidCursor(cursor_id))?;
            cursor.position
        };

        let mut total_read = 0;
        while total_read < buf.len() {
            let region = self.get_or_create_region(curr_pos);
            let buffer = region.buffer.read().map_err(|_| BufIoError::Locking)?;
            let buffer_pos = (curr_pos - region.start) as usize;
            let available = region.end.load(Ordering::SeqCst) - buffer_pos;
            if available == 0 {
                if total_read == 0
                    && curr_pos >= *self.file_size.read().map_err(|_| BufIoError::Locking)?
                {
                    return Ok(0); // EOF
                }
                break;
            }
            let to_read = (buf.len() - total_read).min(available);
            buf[total_read..total_read + to_read]
                .copy_from_slice(&buffer[buffer_pos..buffer_pos + to_read]);
            total_read += to_read;
            curr_pos += to_read as u64;
        }

        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        let cursor = cursors
            .get_mut(&cursor_id)
            .ok_or(BufIoError::InvalidCursor(cursor_id))?;
        cursor.position = curr_pos;

        Ok(total_read)
    }

    pub fn update_u32_with_cursor(&self, cursor_id: u64, value: u32) -> Result<u64, BufIoError> {
        let buffer = value.to_le_bytes();
        self.write_with_cursor(cursor_id, &buffer, false)
    }

    fn write_with_cursor(
        &self,
        cursor_id: u64,
        buf: &[u8],
        append: bool,
    ) -> Result<u64, BufIoError> {
        let curr_pos = {
            let cursors = self.cursors.read().map_err(|_| BufIoError::Locking)?;
            let cursor = cursors
                .get(&cursor_id)
                .ok_or(BufIoError::InvalidCursor(cursor_id))?;
            cursor.position
        };

        let input_size = buf.len();

        // Take write lock early to cover the entire write operation that might affect file size
        let mut file_size_guard = self.file_size.write().map_err(|_| BufIoError::Locking)?;
        let will_cross_eof = append || curr_pos + input_size as u64 >= *file_size_guard;
        let mut curr_pos = if append {
            curr_pos.max(*file_size_guard)
        } else {
            curr_pos
        };
        let start_pos = curr_pos;

        if will_cross_eof {
            let mut total_written = 0;
            while total_written < input_size {
                let region = self.get_or_create_region(curr_pos);
                {
                    let mut buffer = region.buffer.write().map_err(|_| BufIoError::Locking)?;
                    let buffer_pos = (curr_pos - region.start) as usize;
                    let available = self.buffer_size - buffer_pos;
                    let to_write = (input_size - total_written).min(available);
                    buffer[buffer_pos..buffer_pos + to_write]
                        .copy_from_slice(&buf[total_written..total_written + to_write]);
                    region.end.store(
                        (buffer_pos + to_write).max(region.end.load(Ordering::SeqCst)),
                        Ordering::SeqCst,
                    );
                    region.dirty.store(true, Ordering::SeqCst);
                    total_written += to_write;
                    curr_pos += to_write as u64;
                }
            }
            // Update file size using max to prevent shrinking
            *file_size_guard = (*file_size_guard).max(curr_pos);
        } else {
            // Normal write within existing file bounds
            let mut total_written = 0;
            while total_written < input_size {
                let region = self.get_or_create_region(curr_pos);
                {
                    let mut buffer = region.buffer.write().map_err(|_| BufIoError::Locking)?;
                    let buffer_pos = (curr_pos - region.start) as usize;
                    let available = self.buffer_size - buffer_pos;
                    let to_write = (input_size - total_written).min(available);
                    buffer[buffer_pos..buffer_pos + to_write]
                        .copy_from_slice(&buf[total_written..total_written + to_write]);
                    region.end.store(
                        (buffer_pos + to_write).max(region.end.load(Ordering::SeqCst)),
                        Ordering::SeqCst,
                    );
                    region.dirty.store(true, Ordering::SeqCst);
                    total_written += to_write;
                    curr_pos += to_write as u64;
                }
            }
            // Even for normal writes, we need to check if we accidentally crossed EOF
            *file_size_guard = (*file_size_guard).max(curr_pos);
        }

        // Drop file_size_guard before updating cursor
        drop(file_size_guard);

        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        let cursor = cursors
            .get_mut(&cursor_id)
            .ok_or(BufIoError::InvalidCursor(cursor_id))?;
        cursor.position = curr_pos;

        Ok(start_pos)
    }

    pub fn write_to_end_of_file(&self, cursor_id: u64, buf: &[u8]) -> Result<u64, BufIoError> {
        self.write_with_cursor(cursor_id, buf, true)
    }

    pub fn seek_with_cursor(&self, cursor_id: u64, pos: u64) -> Result<(), BufIoError> {
        let mut cursors = self.cursors.write().map_err(|_| BufIoError::Locking)?;
        let cursor = cursors
            .get_mut(&cursor_id)
            .ok_or(BufIoError::InvalidCursor(cursor_id))?;

        cursor.position = pos;
        Ok(())
    }

    pub fn flush(&self, file: &mut File) -> Result<(), BufIoError> {
        for region in self.regions.iter() {
            if region.should_flush() {
                region.flush(file)?;
            }
        }
        file.flush()?;
        file.sync_all()?;
        Ok(())
    }

    pub fn flush_versioned(&self, mapping_fn: impl Fn(u64) -> PathBuf) -> Result<(), BufIoError> {
        for region in self.regions.iter() {
            if region.should_flush() {
                let path = mapping_fn(region.start / self.buffer_size as u64);
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(path)?;
                let buffer = region.buffer.read().map_err(|_| BufIoError::Locking)?;
                let end = region.end.load(Ordering::SeqCst);
                file.write_all(&buffer[..end]).map_err(BufIoError::Io)?;
                region.dirty.store(false, Ordering::SeqCst);
                file.flush()?;
                file.sync_all()?;
            }
        }
        Ok(())
    }

    pub fn file_size(&self) -> u64 {
        *self.file_size.read().unwrap()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use quickcheck_macros::quickcheck;
    use rand::Rng;
    use std::thread;
    use tempfile::tempfile;

    const BUFFER_SIZE: usize = 8192;

    #[test]
    fn test_basic_usage() {
        // @NOTE: Not really using multiple threads here. This test
        // just ensures that read/write using multiple cursors work in
        // a single thread

        let mut file = tempfile().unwrap();
        file.write_all(&456_u32.to_le_bytes()).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();

        let cursor1 = bufman.open_cursor().unwrap();
        let cursor2 = bufman.open_cursor().unwrap();

        // Thread 1: Read from the beginning of file
        bufman.seek_with_cursor(cursor1, 0).unwrap();
        let value1 = bufman.read_u32_with_cursor(cursor1).unwrap();
        assert_eq!(456_u32, value1);

        // Thread 2: Write to the end of file
        bufman
            .write_with_cursor(cursor2, &789_u32.to_le_bytes(), true)
            .unwrap();

        // Thread 1: Continue reading
        let value2 = bufman.read_u32_with_cursor(cursor1).unwrap();
        assert_eq!(789_u32, value2);

        // Thread 2: Again write to end of file
        bufman
            .write_with_cursor(cursor2, &12345_u32.to_le_bytes(), true)
            .unwrap();

        // Thread 1: Continue reading
        let value3 = bufman.read_u32_with_cursor(cursor1).unwrap();
        assert_eq!(12345_u32, value3);

        bufman.close_cursor(cursor1).unwrap();
        bufman.close_cursor(cursor1).unwrap();
    }

    /// Test util function to create a temp file of size calculated
    /// from `num_regions` and `extra` bytes after that.
    ///
    /// E.g. `create_tmp_file(5, 20)` will create a temp file with 5
    /// regions of BUFFER_SIZE and 20 more bytes in addition to it.
    fn create_tmp_file(num_regions: u8, extra: u16) -> io::Result<File> {
        let mut file = tempfile()?;
        file.write_all(&vec![
            0_u8;
            (BUFFER_SIZE * num_regions as usize) + extra as usize
        ])?;
        Ok(file)
    }

    /// Test util function to create a temp file of a specific size
    fn create_tmp_file_of_size(size: u16) -> io::Result<File> {
        let mut file = tempfile()?;
        file.write_all(&vec![0_u8; size as usize])?;
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

        let mut file = create_tmp_file(5, 200).unwrap();
        file.seek(SeekFrom::Start(8190)).unwrap();
        file.write_all(&1678_u32.to_le_bytes()).unwrap();

        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();

        let cursor1 = bufman.open_cursor().unwrap();
        bufman.seek_with_cursor(cursor1, 8190).unwrap();
        let value1 = bufman.read_u32_with_cursor(cursor1).unwrap();
        assert_eq!(1678_u32, value1);

        bufman.close_cursor(cursor1).unwrap();
    }

    #[test]
    fn test_conc_reads_different_regions() {
        let mut file = create_tmp_file(4, 200).unwrap();

        // Write some data in region 1
        file.seek(SeekFrom::Start(100)).unwrap();
        file.write_all(&500_u16.to_le_bytes()).unwrap();

        // Write some data in region 3
        file.seek(SeekFrom::Start(file_offset(3, 147))).unwrap();
        file.write_all(&1000_u32.to_le_bytes()).unwrap();

        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());
        let t1 = {
            // Thread 1 that reads from region 1
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, 100).unwrap();
                let v = bm.read_u16_with_cursor(cid).unwrap();
                assert_eq!(500_u16, v);
                bm.close_cursor(cid).unwrap();
            })
        };

        let t2 = {
            // Thread 2 that reads from region 3
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, file_offset(3, 147)).unwrap();
                let v = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(1000_u32, v);
                bm.close_cursor(cid).unwrap();
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();
    }

    #[test]
    fn test_conc_reads_same_region() {
        let mut file = create_tmp_file(1, 200).unwrap();

        // Write some data in region 1
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&500_u16.to_le_bytes()).unwrap();

        file.seek(SeekFrom::Start(2)).unwrap();
        file.write_all(&1000_u32.to_le_bytes()).unwrap();

        file.seek(SeekFrom::Start(6)).unwrap();
        file.write_all(&145_u32.to_le_bytes()).unwrap();

        // Thread 1 will read first two bytes in region 1
        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());
        let t1 = {
            // Thread 1 that reads from region 1
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, 0).unwrap();
                let v = bm.read_u16_with_cursor(cid).unwrap();
                assert_eq!(500_u16, v);
                bm.close_cursor(cid).unwrap();
            })
        };

        // Thread 2 will read first 6 bytes in region 1
        let t2 = {
            // Thread 1 that reads from region 1
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, 0).unwrap();
                let v1 = bm.read_u16_with_cursor(cid).unwrap();
                assert_eq!(500_u16, v1);
                let v2 = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(1000_u32, v2);
                bm.close_cursor(cid).unwrap();
            })
        };

        // Thread 3 will read last 4 bytes in region 1
        let t3 = {
            // Thread 1 that reads from region 1
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, 6).unwrap();
                let v = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(145_u32, v);
                bm.close_cursor(cid).unwrap();
            })
        };

        for t in [t1, t2, t3] {
            t.join().unwrap();
        }
    }

    #[test]
    fn test_conc_reads_across_regions() {
        let mut file = create_tmp_file(3, 200).unwrap();

        let pos1 = file_offset(1, 8190);
        let pos2 = file_offset(2, 8190);

        file.seek(SeekFrom::Start(pos1)).unwrap();
        file.write_all(&1000_u32.to_le_bytes()).unwrap();

        file.seek(SeekFrom::Start(pos2)).unwrap();
        file.write_all(&2000_u32.to_le_bytes()).unwrap();

        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());
        let t1 = {
            // Thread 1 that reads from region 1 and 2
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, pos1).unwrap();
                let v = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(1000_u32, v);
                bm.close_cursor(cid).unwrap();
            })
        };

        let t2 = {
            // Thread 2 that reads from region 2 and 3
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, pos2).unwrap();
                let v = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(2000_u32, v);
                bm.close_cursor(cid).unwrap();
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();
    }

    #[test]
    fn test_writes_across_regions() {
        let file = create_tmp_file(5, 200).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();

        let cursor = bufman.open_cursor().unwrap();
        bufman.seek_with_cursor(cursor, 8190).unwrap();
        let res = bufman
            .write_with_cursor(cursor, &100000_u32.to_le_bytes(), false)
            .unwrap();
        assert_eq!(8190, res);

        bufman.seek_with_cursor(cursor, 8190).unwrap();
        let x = bufman.read_u32_with_cursor(cursor).unwrap();
        assert_eq!(100000, x);

        // Verify that `bufman.file_size` remains the same
        assert_eq!(
            (BUFFER_SIZE * 5 + 200) as u64,
            *bufman.file_size.read().unwrap()
        );
    }

    // Where a thread starts writing from an offset in the middle of
    // the file but ends up writing beyond end of file
    #[test]
    fn test_writes_beyond_eof() {
        let file = create_tmp_file(0, 10).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();

        let cursor = bufman.open_cursor().unwrap();
        bufman.seek_with_cursor(cursor, 7).unwrap();
        let txt = String::from("Hello, World");
        let data = txt.as_bytes();
        let res = bufman.write_with_cursor(cursor, data, false).unwrap();
        assert_eq!(res, 7);

        // Verify that `bufman.file_size` has correctly increased
        assert_eq!(19_u64, *bufman.file_size.read().unwrap());

        bufman.seek_with_cursor(cursor, 7).unwrap();
        let mut output = [0u8; 12];
        bufman.read_with_cursor(cursor, &mut output).unwrap();
        assert_eq!(b"Hello, World", &output);
    }

    #[test]
    fn test_conc_writes_different_regions() {
        let file = create_tmp_file(3, 200).unwrap();

        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());

        // Assert that the bytes at the position we will be writing
        // to (in 2 separate threads) is initially 0
        let cursor = bufman.open_cursor().unwrap();
        bufman.seek_with_cursor(cursor, 10).unwrap();
        assert_eq!(0, bufman.read_u32_with_cursor(cursor).unwrap());

        bufman.seek_with_cursor(cursor, file_offset(2, 45)).unwrap();
        assert_eq!(0, bufman.read_u32_with_cursor(cursor).unwrap());

        let t1 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, 10).unwrap();
                let res = bm
                    .write_with_cursor(cid, &123_u32.to_le_bytes(), false)
                    .unwrap();
                assert_eq!(10, res);
                bm.close_cursor(cid).unwrap();
            })
        };

        let t2 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                let pos = file_offset(2, 45);
                bm.seek_with_cursor(cid, pos).unwrap();
                let res = bm
                    .write_with_cursor(cid, &456_u32.to_le_bytes(), false)
                    .unwrap();
                assert_eq!(pos, res);
                bm.close_cursor(cid).unwrap();
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();

        // Read the bytes that were just written and verify the values
        bufman.seek_with_cursor(cursor, 10).unwrap();
        assert_eq!(123, bufman.read_u32_with_cursor(cursor).unwrap());
        bufman.seek_with_cursor(cursor, file_offset(2, 45)).unwrap();
        assert_eq!(456, bufman.read_u32_with_cursor(cursor).unwrap());

        // Verify that `bufman.file_size` remains the same
        assert_eq!(
            (BUFFER_SIZE * 3 + 200) as u64,
            *bufman.file_size.read().unwrap()
        );
    }

    // Two threads, both writing to the end of file
    #[test]
    fn test_conc_writes_to_end_of_file() {
        let file = create_tmp_file(0, 45).unwrap();

        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());

        let t1 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.write_with_cursor(cid, &8_u16.to_le_bytes(), true)
                    .unwrap();
                bm.close_cursor(cid).unwrap();
            })
        };

        let t2 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.write_with_cursor(cid, &9_u16.to_le_bytes(), true)
                    .unwrap();
                bm.close_cursor(cid).unwrap();
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();

        let cursor = bufman.open_cursor().unwrap();
        bufman.seek_with_cursor(cursor, 45).unwrap();
        let x = bufman.read_u16_with_cursor(cursor).unwrap();
        let y = bufman.read_u16_with_cursor(cursor).unwrap();

        // We don't know which one of the two threads will run
        // first. So verify both cases.
        if x == 8 {
            assert_eq!(9, y);
        } else if x == 9 {
            assert_eq!(8, y);
        } else {
            panic!();
        }

        bufman.close_cursor(cursor).unwrap();

        // Verify that `bufman.file_size` has increased by 4 bytes
        assert_eq!(49_u64, *bufman.file_size.read().unwrap());
    }

    // Concurrent writes to the same region, where both threads are
    // writing to two different positions in the same region, other
    // than the end of file
    #[test]
    fn test_conc_writes_to_same_region_1() {
        let file = create_tmp_file(2, 23).unwrap();
        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());

        let pos1 = (BUFFER_SIZE * 2 + 12) as u64;
        let pos2 = (BUFFER_SIZE * 2 + 5) as u64;

        // Thread 1 will write to pos1
        let t1 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, pos1).unwrap();
                let res = bm
                    .write_with_cursor(cid, &42_u16.to_le_bytes(), false)
                    .unwrap();
                assert_eq!(pos1, res);
                bm.close_cursor(cid).unwrap();
            })
        };

        // Thread 2 will write to pos2
        let t2 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, pos2).unwrap();
                let res = bm
                    .write_with_cursor(cid, &2024_u32.to_le_bytes(), false)
                    .unwrap();
                assert_eq!(pos2, res);
                bm.close_cursor(cid).unwrap();
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();

        let cursor = bufman.open_cursor().unwrap();

        bufman.seek_with_cursor(cursor, pos1).unwrap();
        let x = bufman.read_u16_with_cursor(cursor).unwrap();
        assert_eq!(42, x);

        bufman.seek_with_cursor(cursor, pos2).unwrap();
        let y = bufman.read_u32_with_cursor(cursor).unwrap();
        assert_eq!(2024, y);

        bufman.close_cursor(cursor).unwrap();

        // Verify that `bufman.file_size` increases by 2 bytes
        assert_eq!(
            (BUFFER_SIZE * 2 + 23) as u64,
            *bufman.file_size.read().unwrap()
        );
    }

    // Concurrent writes to the same region, where one thread is
    // writing to the end of file and other is writing to some other
    // position
    #[test]
    fn test_conc_writes_to_same_region_2() {
        let file = create_tmp_file(2, 23).unwrap();
        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());

        // Thread 1 will write to the end of file
        let t1 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.write_with_cursor(cid, &42_u16.to_le_bytes(), true)
                    .unwrap();
                bm.close_cursor(cid).unwrap();
            })
        };

        // Thread 2 will write to some position in the last region
        // (same region as thread 1 is writing to)
        let t2 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                let some_pos = (BUFFER_SIZE * 2 + 5) as u64;
                bm.seek_with_cursor(cid, some_pos).unwrap();
                let res = bm
                    .write_with_cursor(cid, &2024_u32.to_le_bytes(), false)
                    .unwrap();
                assert_eq!(some_pos, res);
                bm.close_cursor(cid).unwrap();
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();

        let cursor = bufman.open_cursor().unwrap();

        bufman
            .seek_with_cursor(cursor, bufman.file_size() - 2)
            .unwrap();
        let x = bufman.read_u16_with_cursor(cursor).unwrap();
        assert_eq!(42, x);

        bufman
            .seek_with_cursor(cursor, (BUFFER_SIZE * 2 + 5) as u64)
            .unwrap();
        let y = bufman.read_u32_with_cursor(cursor).unwrap();
        assert_eq!(2024, y);

        bufman.close_cursor(cursor).unwrap();

        // Verify that `bufman.file_size` increases by 2 bytes
        assert_eq!(
            (BUFFER_SIZE * 2 + 23 + 2) as u64,
            *bufman.file_size.read().unwrap()
        );
    }

    // 2 threads, one thread writes from region 1 + 2 and another one
    // writes to region 2 + 3
    #[test]
    fn test_conc_writes_across_regions() {
        let file = create_tmp_file(3, 200).unwrap();
        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());

        let pos1 = file_offset(1, 8190);
        let pos2 = file_offset(2, 8190);

        let t1 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, pos1).unwrap();
                let res = bm
                    .write_with_cursor(cid, &123_u32.to_le_bytes(), false)
                    .unwrap();
                assert_eq!(pos1, res);
                bm.close_cursor(cid).unwrap();
            })
        };

        let t2 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, pos2).unwrap();
                let res = bm
                    .write_with_cursor(cid, &456_u32.to_le_bytes(), false)
                    .unwrap();
                assert_eq!(pos2, res);
                bm.close_cursor(cid).unwrap();
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();

        let cursor = bufman.open_cursor().unwrap();

        bufman.seek_with_cursor(cursor, pos1).unwrap();
        let x = bufman.read_u32_with_cursor(cursor).unwrap();
        assert_eq!(123, x);

        bufman.seek_with_cursor(cursor, pos2).unwrap();
        let y = bufman.read_u32_with_cursor(cursor).unwrap();
        assert_eq!(456, y);

        bufman.close_cursor(cursor).unwrap();

        // Verify that `bufman.file_size` remain the same
        assert_eq!(
            (BUFFER_SIZE * 3 + 200) as u64,
            *bufman.file_size.read().unwrap()
        );
    }

    #[test]
    fn test_conc_reads_writes() {
        // 2 threads, one thread reads from region 5 and another thread
        // writes to region 5 as well.
        let mut file = create_tmp_file(1, 10).unwrap();

        let pos1 = file_offset(1, 5);
        let pos2 = file_offset(1, 45);

        file.seek(SeekFrom::Start(pos1)).unwrap();
        file.write_all(&1000_u32.to_le_bytes()).unwrap();

        let bufman = Arc::new(BufferManager::new(file, BUFFER_SIZE).unwrap());
        let t1 = {
            // Thread 1 that reads from region 1 and 2
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, pos1).unwrap();
                let v = bm.read_u32_with_cursor(cid).unwrap();
                assert_eq!(1000_u32, v);
                bm.close_cursor(cid).unwrap();
            })
        };

        let t2 = {
            let bm = bufman.clone();
            thread::spawn(move || {
                let cid = bm.open_cursor().unwrap();
                bm.seek_with_cursor(cid, pos2).unwrap();
                let res = bm
                    .write_with_cursor(cid, &456_u16.to_le_bytes(), false)
                    .unwrap();
                assert_eq!(pos2, res);
                bm.close_cursor(cid).unwrap();
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();

        let cursor = bufman.open_cursor().unwrap();

        bufman.seek_with_cursor(cursor, pos2).unwrap();
        let y = bufman.read_u16_with_cursor(cursor).unwrap();
        assert_eq!(456, y);

        bufman.close_cursor(cursor).unwrap();

        // Verify that `bufman.file_size` remain the same
        assert_eq!((BUFFER_SIZE + 10) as u64, *bufman.file_size.read().unwrap());
    }

    #[test]
    fn test_seek_with_cursor() {
        let filesize = 1000_usize;
        let file = create_tmp_file_of_size(filesize as u16).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let cursor = bufman.open_cursor().unwrap();

        // Test cases
        bufman.seek_with_cursor(cursor, 0).unwrap();
        {
            let cursors = bufman.cursors.read().unwrap();
            let c = cursors.get(&cursor).unwrap();
            assert_eq!(0, c.position);
        }

        bufman.seek_with_cursor(cursor, 478).unwrap();
        {
            let cursors = bufman.cursors.read().unwrap();
            let c = cursors.get(&cursor).unwrap();
            assert_eq!(478, c.position);
        }

        bufman.seek_with_cursor(cursor, filesize as u64).unwrap();
        {
            let cursors = bufman.cursors.read().unwrap();
            let c = cursors.get(&cursor).unwrap();
            assert_eq!(filesize as u64, c.position);
        }

        bufman.close_cursor(cursor).unwrap();
    }

    #[test]
    fn test_read_large_file() {
        let mut rng = rand::thread_rng();

        let mut file = create_tmp_file(0, 0).unwrap();

        let written_data = (0..(100 * 1024 * 1024))
            .map(|_| rng.gen())
            .collect::<Vec<_>>();
        file.write_all(&written_data).unwrap();

        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let cursor = bufman.open_cursor().unwrap();

        let mut read_data = vec![0; 100 * 1024 * 1024];
        let bytes_read = bufman.read_with_cursor(cursor, &mut read_data).unwrap();

        assert_eq!(bytes_read, read_data.len());
        assert_eq!(read_data, written_data);
    }

    #[test]
    fn test_write_large_file() {
        let mut rng = rand::thread_rng();

        let file = create_tmp_file(0, 0).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let cursor = bufman.open_cursor().unwrap();

        let written_data = (0..(100 * 1024 * 1024))
            .map(|_| rng.gen())
            .collect::<Vec<_>>();
        bufman
            .write_with_cursor(cursor, &written_data, true)
            .unwrap();
        bufman.flush().unwrap();

        let mut read_data = vec![0; 100 * 1024 * 1024];

        let mut file = bufman.file.write().unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.read_exact(&mut read_data).unwrap();

        for (i, (r, w)) in read_data.into_iter().zip(written_data).enumerate() {
            assert_eq!(r, w, "mismatch at {}", i);
        }
    }

    // Prop test for `get_or_create_region` to check that
    // `region.start` is a multiple of BUFFER_SIZE
    #[quickcheck]
    fn prop_get_or_create_region_start(pos: u16) -> bool {
        let file = create_tmp_file_of_size(u16::MAX).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let region = bufman.get_or_create_region(pos as u64).unwrap();
        region.start % (BUFFER_SIZE as u64) == 0
    }

    // Prop test for `get_or_create_region` to check that
    // `region.buffer` size is at most equal to BUFFER_SIZE
    #[quickcheck]
    fn prop_get_or_create_region_buffer_size(pos: u16) -> bool {
        let file = create_tmp_file_of_size(u16::MAX).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let region = bufman.get_or_create_region(pos as u64).unwrap();
        let buffer = region.buffer.read().unwrap();
        buffer.len() <= BUFFER_SIZE
    }

    // Prop test for `get_or_create_region` to check that
    // `region.end` is at most equal to BUFFER_SIZE
    #[quickcheck]
    fn prop_get_or_create_region_end(pos: u16) -> bool {
        let file = create_tmp_file_of_size(u16::MAX).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let region = bufman.get_or_create_region(pos as u64).unwrap();
        region.end.load(Ordering::SeqCst) <= BUFFER_SIZE
    }

    // Prop test for `read_with_cursor` to check that the return value
    // (total bytes read) is at most equal to the size of the input
    // buffer
    #[quickcheck]
    fn prop_read_with_cursor(size: u16) -> bool {
        let bufsize = size as usize;
        let file = create_tmp_file_of_size(u16::MAX).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let mut buffer = vec![0; bufsize];
        let cursor = bufman.open_cursor().unwrap();
        let bytes_read = bufman.read_with_cursor(cursor, &mut buffer[..]).unwrap();
        bufman.close_cursor(cursor).unwrap();
        bytes_read <= bufsize
    }

    // Prop test for `write_with_cursor` to check that the return
    // value (write offset) is `EOF - buffer_size` (0 in this case)
    #[quickcheck]
    fn prop_write_with_cursor(size: u16) -> bool {
        let bufsize = size as usize;
        let file = create_tmp_file_of_size(0).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let buffer = vec![0; bufsize];
        let cursor = bufman.open_cursor().unwrap();
        let written_at = bufman.write_with_cursor(cursor, &buffer[..], true).unwrap();
        bufman.close_cursor(cursor).unwrap();
        written_at == 0
    }

    // Prop test for `write_with_cursor` to check that all regions
    // that are written to are dirty
    #[quickcheck]
    fn prop_write_then_read(size: u16) -> bool {
        let bufsize = size as usize;
        let file = create_tmp_file_of_size(0).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();

        // Write to the file
        let input_buffer = vec![1; bufsize];
        let cursor = bufman.open_cursor().unwrap();
        bufman
            .write_with_cursor(cursor, &input_buffer[..], true)
            .unwrap();
        bufman.close_cursor(cursor).unwrap();

        // Read from the file
        let mut output_buffer = vec![0; bufsize];
        let cursor = bufman.open_cursor().unwrap();
        bufman
            .read_with_cursor(cursor, &mut output_buffer[..])
            .unwrap();
        bufman.close_cursor(cursor).unwrap();

        output_buffer[..bufsize] == input_buffer[..bufsize]
    }

    // For seek_with_cursor, we just verify that it doesn't crash with
    // u16 type for filesize and position.

    fn check_seek_with_cursor_doesnt_crash(filesize: u16, pos: u64) -> bool {
        let file = create_tmp_file_of_size(filesize).unwrap();
        let bufman = BufferManager::new(file, BUFFER_SIZE).unwrap();
        let cursor = bufman.open_cursor().unwrap();
        let result = bufman.seek_with_cursor(cursor, pos);
        bufman.close_cursor(cursor).unwrap();
        result.is_ok()
    }

    #[quickcheck]
    fn prop_seek_with_cursor_from_start(filesize: u16, pos: u16) -> bool {
        check_seek_with_cursor_doesnt_crash(filesize, pos as u64)
    }
}
