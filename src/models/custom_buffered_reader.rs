use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::rc::Rc;

pub const BUFFER_SIZE: usize = 8192; // 8 KB buffer, same as CustomBufferedWriter

pub struct CustomBufferedReader {
    file: Rc<RefCell<File>>,
    buffer: [u8; BUFFER_SIZE],
    buffer_position: usize,
    buffer_end: usize,
    file_position: u64,
}

impl CustomBufferedReader {
    pub fn new(file: Rc<RefCell<File>>) -> io::Result<Self> {
        let file_position = file.borrow_mut().stream_position()?;
        Ok(CustomBufferedReader {
            file,
            buffer: [0; BUFFER_SIZE],
            buffer_position: 0,
            buffer_end: 0,
            file_position,
        })
    }

    fn fill_buffer(&mut self) -> io::Result<()> {
        self.file
            .borrow_mut()
            .seek(SeekFrom::Start(self.file_position))?;
        self.buffer_end = self.file.borrow_mut().read(&mut self.buffer)?;
        self.buffer_position = 0;
        Ok(())
    }

    pub fn read_u32(&mut self) -> io::Result<u32> {
        let mut buffer = [0u8; 4];
        self.read_exact(&mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }

    pub fn read_u16(&mut self) -> io::Result<u16> {
        let mut buffer = [0u8; 2];
        self.read_exact(&mut buffer)?;
        Ok(u16::from_le_bytes(buffer))
    }

    pub fn stream_position(&self) -> io::Result<u64> {
        Ok(self.file_position + self.buffer_position as u64)
    }
}

impl Read for CustomBufferedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.buffer_position >= self.buffer_end {
            self.fill_buffer()?;
            if self.buffer_end == 0 {
                return Ok(0); // EOF
            }
        }

        let available = self.buffer_end - self.buffer_position;
        let to_read = buf.len().min(available);

        buf[..to_read]
            .copy_from_slice(&self.buffer[self.buffer_position..self.buffer_position + to_read]);
        self.buffer_position += to_read;
        self.file_position += to_read as u64;

        Ok(to_read)
    }
}

impl Seek for CustomBufferedReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let current_position = self.file_position + self.buffer_position as u64;
        let new_position = match pos {
            SeekFrom::Start(abs) => abs,
            SeekFrom::End(rel) => {
                let file_len = self.file.borrow_mut().seek(SeekFrom::End(0))?;
                (file_len as i64 + rel) as u64
            }
            SeekFrom::Current(rel) => (current_position as i64 + rel) as u64,
        };

        if new_position < self.file_position
            || new_position >= self.file_position + self.buffer_end as u64
        {
            // Seeking outside the current buffer
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
