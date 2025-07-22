use crate::indexes::hnsw::offset_counter::IndexFileId;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::serializer::*;
use crate::models::types::*;
use crate::storage::Storage;
use half::f16;
use tempfile::TempDir;

#[test]
fn test_storage_serialization() {
    let storages = [
        Storage::UnsignedByte {
            mag: 10.0,
            quant_vec: vec![0, 1, 4],
        },
        Storage::SubByte {
            mag: 34.0,
            quant_vec: vec![vec![55, 35], vec![56, 23]],
            resolution: 2,
        },
        Storage::HalfPrecisionFP {
            mag: 4234.34,
            quant_vec: vec![f16::from_f32(534.324), f16::from_f32(6453.3)],
        },
    ];
    let tempdir = TempDir::new().unwrap();
    let bufmans = BufferManagerFactory::new(
        tempdir.as_ref().into(),
        |root, ver: &IndexFileId| root.join(format!("{}.index", **ver)),
        8192,
    );

    for (file_id, storage) in storages.into_iter().enumerate() {
        let version_id = IndexFileId::from(file_id as u32);
        let bufman = bufmans.get(version_id).unwrap();
        let cursor = bufman.open_cursor().unwrap();
        let offset = SimpleSerialize::serialize(&storage, &bufman, cursor).unwrap();
        let deserialized: Storage =
            SimpleSerialize::deserialize(&bufman, FileOffset(offset)).unwrap();

        assert_eq!(deserialized, storage);
    }
}
