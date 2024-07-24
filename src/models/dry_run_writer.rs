use std::io::{self, Seek, SeekFrom, Write};

pub struct DryRunWriter {
    position: u64,
}

impl DryRunWriter {
    pub fn new() -> Self {
        DryRunWriter { position: 0 }
    }

    pub fn bytes_written(&self) -> u64 {
        self.position
    }
}

impl Write for DryRunWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len();
        self.position += len as u64;
        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for DryRunWriter {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Start(offset) => {
                self.position = offset;
            }
            SeekFrom::End(offset) => {
                self.position = self.position.saturating_add_signed(offset);
            }
            SeekFrom::Current(offset) => {
                self.position = self.position.saturating_add_signed(offset);
            }
        }
        Ok(self.position)
    }
}

// Extension methods for writing specific types
impl DryRunWriter {
    pub fn write_u32(&mut self, _value: u32) -> io::Result<usize> {
        self.write(&[0; 4])
    }

    pub fn write_u16(&mut self, _value: u16) -> io::Result<usize> {
        self.write(&[0; 2])
    }

    pub fn write_u8(&mut self, _value: u8) -> io::Result<usize> {
        self.write(&[0; 1])
    }
}
