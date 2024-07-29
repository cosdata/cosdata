#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::RefCell,
        fs::File,
        io::{Read, Seek, SeekFrom, Write},
        rc::Rc,
    };
    use tempfile::tempfile;

    fn create_test_reader(initial_content: &[u8]) -> (CustomBufferedReader, Rc<RefCell<File>>) {
        let file = Rc::new(RefCell::new(tempfile().unwrap()));
        file.borrow_mut().write_all(initial_content).unwrap();
        file.borrow_mut().seek(SeekFrom::Start(0)).unwrap();
        let reader = CustomBufferedReader::new(file.clone()).unwrap();
        (reader, file)
    }

    #[test]
    fn test_read_within_buffer() {
        let (mut reader, _) = create_test_reader(b"Hello, World!");
        let mut buffer = [0u8; 5];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello");

        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b", Wor");

        // Check the stream position
        assert_eq!(reader.stream_position().unwrap(), 10);
    }

    #[test]
    fn test_read_across_buffer_boundary() {
        let large_content = vec![0xAA; BUFFER_SIZE + 100];
        let (mut reader, _) = create_test_reader(&large_content);

        let mut buffer = vec![0u8; BUFFER_SIZE + 50];
        let bytes_read = reader.read(&mut buffer).unwrap();

        assert_eq!(bytes_read, BUFFER_SIZE + 50);
        assert_eq!(&buffer[..bytes_read], &large_content[..BUFFER_SIZE + 50]);
    }

    #[test]
    fn test_seek_within_buffer() {
        let (mut reader, _) = create_test_reader(b"Hello, World!");
        reader.seek(SeekFrom::Start(7)).unwrap();

        let mut buffer = [0u8; 5];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"World");
    }

    #[test]
    fn test_seek_outside_buffer() {
        let content = b"Hello, World! This is a test.";
        let (mut reader, _) = create_test_reader(content);

        reader.seek(SeekFrom::End(-10)).unwrap();

        let mut buffer = [0u8; 10];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"is a test.");
    }

    #[test]
    fn test_read_u32_and_u16() {
        let content = [0x78, 0x56, 0x34, 0x12, 0xCD, 0xAB];
        let (mut reader, _) = create_test_reader(&content);

        assert_eq!(reader.read_u32().unwrap(), 0x12345678);
        assert_eq!(reader.read_u16().unwrap(), 0xABCD);
    }

    #[test]
    fn test_large_read() {
        let large_content = vec![0xBB; BUFFER_SIZE * 2];
        let (mut reader, _) = create_test_reader(&large_content);

        let mut buffer = vec![0u8; BUFFER_SIZE * 2];
        reader.read_exact(&mut buffer).unwrap();

        assert_eq!(buffer, large_content);
    }

    #[test]
    fn test_read_to_end() {
        let content = b"Hello, World! This is a test.";
        let (mut reader, _) = create_test_reader(content);

        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer, content);
    }

    #[test]
    fn test_seek_and_read() {
        let content = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let (mut reader, _) = create_test_reader(content);

        reader.seek(SeekFrom::Start(10)).unwrap();
        let mut buffer = [0u8; 5];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"KLMNO");

        reader.seek(SeekFrom::Current(-10)).unwrap();
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"FGHIJ");

        reader.seek(SeekFrom::End(-5)).unwrap();
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"VWXYZ");
    }

    #[test]
    fn test_read_beyond_eof() {
        let content = b"Short content";
        let (mut reader, _) = create_test_reader(content);

        let mut buffer = [0u8; 20];
        let bytes_read = reader.read(&mut buffer).unwrap();

        assert_eq!(bytes_read, content.len());
        assert_eq!(&buffer[..bytes_read], content);

        // Attempt to read beyond EOF
        let additional_bytes = reader.read(&mut buffer).unwrap();
        assert_eq!(additional_bytes, 0);
    }
}
