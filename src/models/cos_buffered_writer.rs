use byteorder::{LittleEndian, WriteBytesExt};
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::rc::Rc;

const BUFFER_SIZE: usize = 16384; // 16 KB buffer, adjust as needed
const FLUSH_THRESHOLD: usize = (BUFFER_SIZE as f32 * 0.7) as usize; // 70% of buffer size

pub struct CustomBufferedWriter {
    file: Rc<RefCell<File>>,
    buffer: [u8; BUFFER_SIZE],
    buffer_position: usize,
    file_position: u64,
}

impl CustomBufferedWriter {
    pub fn new(file: Rc<RefCell<File>>) -> io::Result<Self> {
        let file_position = file.borrow_mut().stream_position()?;
        Ok(CustomBufferedWriter {
            file,
            buffer: [0; BUFFER_SIZE],
            buffer_position: 0,
            file_position,
        })
    }

    pub fn flush_buffer(&mut self) -> io::Result<()> {
        if self.buffer_position > 0 {
            self.file
                .borrow_mut()
                .write_all(&self.buffer[..self.buffer_position])?;
            self.file_position += self.buffer_position as u64;
            self.buffer_position = 0;
        }
        Ok(())
    }

    pub fn write_u32(&mut self, value: u32) -> io::Result<()> {
        let bytes = value.to_le_bytes();
        self.write(&bytes)?;
        Ok(())
    }

    pub fn stream_position(&self) -> io::Result<u64> {
        Ok(self.file_position + self.buffer_position as u64)
    }
}
impl Write for CustomBufferedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        debug_assert!(
            buf.len() <= BUFFER_SIZE - FLUSH_THRESHOLD,
            "Write size exceeds remaining buffer capacity after flush threshold"
        );

        self.buffer[self.buffer_position..self.buffer_position + buf.len()].copy_from_slice(buf);
        self.buffer_position += buf.len();

        if self.buffer_position >= FLUSH_THRESHOLD {
            self.flush()?;
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        self.file.borrow_mut().flush()
    }
}

impl Seek for CustomBufferedWriter {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Start(abs) => {
                self.file_position = abs;
            }
            SeekFrom::End(rel) => {
                let file_len = self.file.borrow_mut().seek(SeekFrom::End(0))?;
                self.file_position = (file_len as i64 + rel) as u64;
                self.file
                    .borrow_mut()
                    .seek(SeekFrom::Start(self.file_position))?;
            }
            SeekFrom::Current(rel) => {
                self.file_position = (self.file_position as i64 + rel) as u64;
            }
        }
        Ok(self.file_position)
    }
}
