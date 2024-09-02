use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::file_persist::*;
use super::lazy_load::{FileIndex, LazyItem, LazyItemMap};
use super::serializer::CustomSerialize;
use super::types::*;
use arcshift::ArcShift;
use dashmap::DashMap;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::collections::HashSet;
use std::io;
use std::path::Path;
use std::sync::{atomic::AtomicBool, Arc, RwLock};

pub struct NodeRegistry {
    cuckoo_filter: RwLock<CuckooFilter<u64>>,
    registry: DashMap<u64, LazyItem<MergedNode>>,
    bufmans: Arc<BufferManagerFactory>,
}

impl NodeRegistry {
    pub fn new(cuckoo_filter_capacity: usize, bufmans: Arc<BufferManagerFactory>) -> Self {
        let cuckoo_filter = CuckooFilter::new(cuckoo_filter_capacity);
        let registry = DashMap::new();
        NodeRegistry {
            cuckoo_filter: RwLock::new(cuckoo_filter),
            registry,
            bufmans,
        }
    }
    pub fn get_object<F>(
        self: Arc<Self>,
        file_index: FileIndex,
        load_function: F,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<LazyItem<MergedNode>, BufIoError>
    where
        F: Fn(
            Arc<BufferManagerFactory>,
            FileIndex,
            Arc<Self>,
            u16,
            &mut HashSet<u64>,
        ) -> Result<LazyItem<MergedNode>, BufIoError>,
    {
        println!(
            "get_object called with file_index: {:?}, max_loads: {}",
            file_index, max_loads
        );

        let combined_index = Self::combine_index(&file_index);

        {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            println!("Acquired read lock on cuckoo_filter");

            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&combined_index) {
                println!("FileIndex found in cuckoo_filter");
                if let Some(obj) = self.registry.get(&combined_index) {
                    println!("Object found in registry, returning");
                    return Ok(obj.clone());
                } else {
                    println!("Object not found in registry despite being in cuckoo_filter");
                }
            } else {
                println!("FileIndex not found in cuckoo_filter");
            }
        }
        println!("Released read lock on cuckoo_filter");

        let version_id = if let FileIndex::Valid { version, .. } = &file_index {
            *version
        } else {
            0.into()
        };

        if max_loads == 0 || !skipm.insert(combined_index) {
            println!("Either max_loads hit 0 or loop detected, returning LazyItem with no data");
            return Ok(LazyItem::Valid {
                data: None,
                file_index: ArcShift::new(Some(file_index)),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions: LazyItemMap::new(),
                version_id,
                serialized_flag: Arc::new(AtomicBool::new(true)),
            });
        }

        println!("Calling load_function");
        let item = load_function(
            self.bufmans.clone(),
            file_index.clone(),
            self.clone(),
            max_loads - 1,
            skipm,
        )?;
        println!("load_function returned successfully");

        if let Some(obj) = self.registry.get(&combined_index) {
            println!("Object found in registry after load, returning");
            return Ok(obj.clone());
        }

        println!("Inserting key into cuckoo_filter");
        self.cuckoo_filter.write().unwrap().insert(&combined_index);

        println!("Inserting item into registry");
        self.registry.insert(combined_index, item.clone());

        println!("Returning newly created LazyItem");
        Ok(item)
    }

    pub fn load_item<T: CustomSerialize>(
        self: Arc<Self>,
        file_index: FileIndex,
    ) -> Result<T, BufIoError> {
        let mut skipm: HashSet<u64> = HashSet::new();

        if file_index == FileIndex::Invalid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize with an invalid FileIndex",
            )
            .into());
        };

        T::deserialize(
            self.bufmans.clone(),
            file_index,
            self.clone(),
            1000,
            &mut skipm,
        )
    }

    pub fn combine_index(file_index: &FileIndex) -> u64 {
        match file_index {
            FileIndex::Valid { offset, version } => ((offset.0 as u64) << 32) | (**version as u64),
            FileIndex::Invalid => u64::MAX, // Use max u64 value for Invalid
        }
    }

    pub fn split_combined_index(combined: u64) -> FileIndex {
        if combined == u64::MAX {
            FileIndex::Invalid
        } else {
            FileIndex::Valid {
                offset: FileOffset((combined >> 32) as u32),
                version: (combined as u32).into(),
            }
        }
    }
}

pub fn load_cache() {
    // TODO: include db name in the path
    let bufmans = Arc::new(BufferManagerFactory::new(Path::new(".").into()));

    let file_index = FileIndex::Valid {
        offset: FileOffset(0),
        version: 0.into(),
    }; // Assuming initial version is 0
    let cache = Arc::new(NodeRegistry::new(1000, bufmans));
    match read_node_from_file(file_index.clone(), cache) {
        Ok(_) => println!(
            "Successfully read and printed node from file_index {:?}",
            file_index
        ),
        Err(e) => println!("Failed to read node: {}", e),
    }
}
