#[cfg(tests)]
mod tests {
    use crate::models::custom_buffered_writer::{
        CustomBufferedWriter, BUFFER_SIZE, FLUSH_THRESHOLD,
    };
    use std::{
        cell::RefCell,
        fs::File,
        io::{Read, Seek, SeekFrom, Write},
        rc::Rc,
    };
    use tempfile::tempfile;

    fn create_test_writer() -> (CustomBufferedWriter, Rc<RefCell<File>>) {
        let file = Rc::new(RefCell::new(tempfile().unwrap()));
        let writer = CustomBufferedWriter::new(file.clone()).unwrap();
        (writer, file)
    }

    fn read_file_contents(file: &mut File) -> Vec<u8> {
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut contents = Vec::new();
        file.read_to_end(&mut contents).unwrap();
        contents
    }

    #[test]
    fn test_write_within_buffer() {
        let (mut writer, file) = create_test_writer();
        writer.write_all(b"Hello, World!").unwrap();

        // Check that the file is still empty (not flushed)
        assert_eq!(read_file_contents(&mut file.borrow_mut()), Vec::<u8>::new());

        // Check the stream position
        assert_eq!(writer.stream_position().unwrap(), 13);
    }

    #[test]
    fn test_flush_on_threshold() {
        let (mut writer, file) = create_test_writer();
        let threshold_data = vec![0; FLUSH_THRESHOLD - 1];
        writer.write_all(&threshold_data).unwrap();

        // Check that the file is still empty (not flushed)
        assert_eq!(read_file_contents(&mut file.borrow_mut()), Vec::<u8>::new());

        // Write one more byte to trigger flush
        writer.write_all(&[1]).unwrap();

        // Check that the file now contains the data
        assert_eq!(
            read_file_contents(&mut file.borrow_mut()),
            [&threshold_data[..], &[1]].concat()
        );
    }

    #[test]
    fn test_manual_flush() {
        let (mut writer, file) = create_test_writer();
        writer.write_all(b"Test data").unwrap();
        writer.flush().unwrap();

        // Check that the file contains the flushed data
        assert_eq!(read_file_contents(&mut file.borrow_mut()), b"Test data");
    }

    #[test]
    fn test_seek_within_buffer() {
        let (mut writer, file) = create_test_writer();
        writer.write_all(b"Hello, World!").unwrap();
        writer.seek(SeekFrom::Start(7)).unwrap();
        writer.write_all(b"Human").unwrap();

        // Check that the file is still empty (not flushed)
        assert_eq!(read_file_contents(&mut file.borrow_mut()), Vec::<u8>::new());

        writer.flush().unwrap();

        // Check the final content
        assert_eq!(read_file_contents(&mut file.borrow_mut()), b"Hello, Human!");
    }

    #[test]
    fn test_seek_outside_buffer() {
        let (mut writer, file) = create_test_writer();
        writer.write_all(b"Initial data").unwrap();
        writer.seek(SeekFrom::Start(100)).unwrap();
        writer.write_all(b"More data").unwrap();
        writer.flush_buffer();

        // Check that the seek caused a flush
        let contents = read_file_contents(&mut file.borrow_mut());
        assert_eq!(&contents[..12], b"Initial data");
        assert_eq!(&contents[100..109], b"More data");
    }

    #[test]
    fn test_write_u32_and_u16() {
        let (mut writer, file) = create_test_writer();
        writer.write_u32(0x12345678).unwrap();
        writer.write_u16(0xABCD).unwrap();
        writer.flush().unwrap();

        let contents = read_file_contents(&mut file.borrow_mut());
        assert_eq!(contents, [0x78, 0x56, 0x34, 0x12, 0xCD, 0xAB]);
    }

    #[test]
    fn test_large_write() {
        let (mut writer, file) = create_test_writer();
        let large_data = vec![0xAA; BUFFER_SIZE * 2];
        writer.write_all(&large_data).unwrap();

        // Check that at least one flush occurred
        let contents = read_file_contents(&mut file.borrow_mut());
        assert!(!contents.is_empty());
        assert!(contents.len() >= BUFFER_SIZE);

        writer.flush().unwrap();

        // Check the final content
        assert_eq!(read_file_contents(&mut file.borrow_mut()), large_data);
    }

    #[test]
    fn test_seek_to_end() {
        let (mut writer, file) = create_test_writer();
        writer.write_all(b"Initial").unwrap();
        writer.flush().unwrap();

        writer.seek(SeekFrom::End(0)).unwrap();
        writer.write_all(b" Appended").unwrap();
        writer.flush().unwrap();

        assert_eq!(
            read_file_contents(&mut file.borrow_mut()),
            b"Initial Appended"
        );
    }
}
