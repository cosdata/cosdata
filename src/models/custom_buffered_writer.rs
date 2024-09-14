use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::rc::Rc;

pub const BUFFER_SIZE: usize = 8192; // 8 KB buffer, adjust as needed
pub const FLUSH_THRESHOLD: usize = (BUFFER_SIZE as f32 * 0.7) as usize; // 70% of buffer size

pub struct CustomBufferedWriter {
    file: Rc<RefCell<File>>,
    buffer: [u8; BUFFER_SIZE],
    buffer_position: usize,
    buffer_end: usize,
    file_position: u64,
}

impl CustomBufferedWriter {
    pub fn new(file: Rc<RefCell<File>>) -> io::Result<Self> {
        let file_position = file.borrow_mut().stream_position()?;
        Ok(CustomBufferedWriter {
            file,
            buffer: [0; BUFFER_SIZE],
            buffer_position: 0,
            buffer_end: 0,
            file_position,
        })
    }

    pub fn flush_buffer(&mut self) -> io::Result<()> {
        if self.buffer_end > 0 {
            self.file
                .borrow_mut()
                .write_all(&self.buffer[..self.buffer_end])?;
            self.file_position += self.buffer_end as u64;
            self.buffer_position = 0;
            self.buffer_end = 0;
        }
        Ok(())
    }

    pub fn write_u32(&mut self, value: u32) -> io::Result<usize> {
        self.write(&value.to_le_bytes())
    }

    pub fn write_u16(&mut self, value: u16) -> io::Result<usize> {
        self.write(&value.to_le_bytes())
    }

    pub fn stream_position(&self) -> io::Result<u64> {
        Ok(self.file_position + self.buffer_position as u64)
    }
}

impl Write for CustomBufferedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let available_space = BUFFER_SIZE - self.buffer_position;
        let to_write = buf.len().min(available_space);

        self.buffer[self.buffer_position..self.buffer_position + to_write]
            .copy_from_slice(&buf[..to_write]);
        self.buffer_position += to_write;
        self.buffer_end = self.buffer_end.max(self.buffer_position);

        if self.buffer_end >= FLUSH_THRESHOLD {
            self.flush_buffer()?;
        }

        Ok(to_write)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        self.file.borrow_mut().flush()
    }
}

impl Seek for CustomBufferedWriter {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let current_position = self.file_position + self.buffer_position as u64;
        let new_position = match pos {
            SeekFrom::Start(abs) => abs,
            SeekFrom::End(rel) => {
                if rel == 0 || ((rel < 0) && (-rel as u64) <= self.buffer_end as u64) {
                    // We can handle this seek within the buffer
                    (self.file_position + self.buffer_end as u64) + rel as u64
                } else {
                    // Need to flush and seek in the file
                    self.flush_buffer()?;
                    let file_len = self.file.borrow_mut().seek(SeekFrom::End(0))?;
                    (file_len as i64 + rel) as u64
                }
            }
            SeekFrom::Current(rel) => (current_position as i64 + rel) as u64,
        } as u64;

        if new_position < self.file_position
            || new_position > self.file_position + self.buffer_end as u64
        {
            // Seeking outside the current buffer
            self.flush_buffer()?;
            self.file.borrow_mut().seek(SeekFrom::Start(new_position))?;
            self.file_position = new_position;
            self.buffer_position = 0;
            self.buffer_end = 0;
        } else {
            // Seeking within the buffer
            self.buffer_position = (new_position - self.file_position) as usize;
        }

        Ok(new_position)
    }
}
