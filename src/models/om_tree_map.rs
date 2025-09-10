use std::{
    fs::OpenOptions,
    marker::PhantomData,
    path::Path,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    common::TSHashTable,
    om_versioned_vec::{LazyOmVersionedVec, OmVersionedVec},
    serializer::InlineSerialize,
    types::FileOffset,
    utils::calculate_path,
    versioning::VersionNumber,
};

pub trait OmTreeMapKey {
    fn primary_key(&self) -> u16;
    fn secondary_key(&self) -> u64;
}

pub struct OmTreeMap<K, V> {
    pub(crate) root: OmTreeMapNode<V>,
    pub(crate) dim_bufman: BufferManager,
    pub(crate) data_bufmans: BufferManagerFactory<VersionNumber>,
    _marker: PhantomData<K>,
}

unsafe impl<K, V: Send> Send for OmTreeMap<K, V> {}
unsafe impl<K, V: Sync> Sync for OmTreeMap<K, V> {}

pub struct OmTreeMapNode<T> {
    pub node_idx: u16,
    pub offset: RwLock<Option<FileOffset>>,
    pub quotients: OmQuotientsMap<T>,
    pub children: AtomicArray<Self, 8>,
    pub quotient_last_update_version: AtomicU32,
    pub children_last_update_version: AtomicU32,
}

pub struct OmQuotientsMap<T> {
    pub offset: RwLock<Option<FileOffset>>,
    pub map: TSHashTable<u64, Arc<OmQuotient<T>>>,
    pub len: AtomicU64,
    pub serialized_upto: AtomicU64,
}

pub struct OmQuotient<T> {
    pub sequence_idx: u64,
    pub values: RwLock<LazyOmVersionedVec<T>>,
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for OmTreeMapNode<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.offset.read() == *other.offset.read()
            && self.quotients == other.quotients
            && self.children == other.children
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for OmTreeMapNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapNode")
            .field("offset", &*self.offset.read())
            .field("quotients", &self.quotients)
            .field("children", &self.children)
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for OmQuotientsMap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
            && self.len.load(Ordering::Relaxed) == other.len.load(Ordering::Relaxed)
            && self.serialized_upto.load(Ordering::Relaxed)
                == other.serialized_upto.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for OmQuotientsMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotientMap")
            .field("map", &self.map)
            .field("len", &self.len.load(Ordering::Relaxed))
            .field(
                "serialized_upto",
                &self.serialized_upto.load(Ordering::Relaxed),
            )
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for OmQuotient<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_idx == other.sequence_idx && *self.values.read() == *other.values.read()
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for OmQuotient<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Quotient")
            .field("sequence_idx", &self.sequence_idx)
            .field("values", &*self.values.read())
            .finish()
    }
}

impl<T> Default for OmQuotientsMap<T> {
    fn default() -> Self {
        Self {
            offset: RwLock::new(None),
            map: TSHashTable::new(16),
            len: AtomicU64::new(0),
            serialized_upto: AtomicU64::new(0),
        }
    }
}

impl<T> OmQuotientsMap<T> {
    pub fn push(&self, version: VersionNumber, key: u64, value: T) -> bool {
        self.map.modify_or_insert_with_value(
            key,
            value,
            |value, quotient| {
                let mut values = quotient.values.write();
                values.push(version, value);
            },
            |value| {
                let mut values = OmVersionedVec::new(version);
                values.push(version, value);
                Arc::new(OmQuotient {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    values: RwLock::new(LazyOmVersionedVec::Vec(Box::new(values))),
                })
            },
        )
    }
}

impl<T: Ord> OmQuotientsMap<T> {
    pub fn push_sorted(&self, version: VersionNumber, key: u64, value: T) -> bool {
        self.map.modify_or_insert_with_value(
            key,
            value,
            |value, quotient| {
                let mut values = quotient.values.write();
                values.push(version, value);
            },
            |value| {
                let mut values = OmVersionedVec::new(version);
                values.push(version, value);
                Arc::new(OmQuotient {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    values: RwLock::new(LazyOmVersionedVec::Vec(Box::new(values))),
                })
            },
        )
    }
}

impl<T> OmTreeMapNode<T> {
    pub fn new(node_idx: u16) -> Self {
        Self {
            node_idx,
            offset: RwLock::new(None),
            quotients: OmQuotientsMap::default(),
            children: AtomicArray::default(),
            quotient_last_update_version: AtomicU32::new(0),
            children_last_update_version: AtomicU32::new(0),
        }
    }

    pub fn find_or_create_node(&self, path: &[u8], version: VersionNumber) -> &Self {
        let mut current = self;
        for &idx in path {
            let new_node_idx = current.node_idx + (1u16 << (idx * 2));
            let (child, _) = current.children.get_or_insert(idx as usize, || {
                Box::into_raw(Box::new(Self::new(new_node_idx)))
            });
            current
                .children_last_update_version
                .store(*version, Ordering::Release);
            current = unsafe { &*child };
        }
        current
    }

    pub fn push(&self, version: VersionNumber, key: u64, value: T) -> bool {
        self.quotient_last_update_version
            .store(*version, Ordering::Release);
        self.quotients.push(version, key, value)
    }

    pub fn push_sorted(&self, version: VersionNumber, key: u64, value: T) -> bool
    where
        T: Ord,
    {
        self.quotient_last_update_version
            .store(*version, Ordering::Release);
        self.quotients.push_sorted(version, key, value)
    }
}

impl<K: OmTreeMapKey, V> OmTreeMap<K, V> {
    pub fn new(
        dim_bufman: BufferManager,
        data_bufmans: BufferManagerFactory<VersionNumber>,
    ) -> Self {
        Self {
            root: OmTreeMapNode::new(0),
            dim_bufman,
            data_bufmans,
            _marker: PhantomData,
        }
    }

    pub fn find_node(&self, primary_key: u16) -> Option<&OmTreeMapNode<V>> {
        let mut current_node = &self.root;
        let path = calculate_path(primary_key as u32, 0);
        for child_idx in path {
            let child = current_node.children.get(child_idx as usize)?;
            let node_res = unsafe { &*child };
            current_node = node_res;
        }

        Some(current_node)
    }

    pub fn push(&self, version: VersionNumber, key: &K, value: V) -> bool {
        let primary_key = key.primary_key();
        let secondary_key = key.secondary_key();
        let path = calculate_path(primary_key as u32, 0);
        let node = self.root.find_or_create_node(&path, version);
        node.push(version, secondary_key, value)
    }

    pub fn push_sorted(&self, version: VersionNumber, key: &K, value: V) -> bool
    where
        V: Ord,
    {
        let primary_key = key.primary_key();
        let secondary_key = key.secondary_key();
        let path = calculate_path(primary_key as u32, 0);
        let node = self.root.find_or_create_node(&path, version);
        node.push_sorted(version, secondary_key, value)
    }

    pub fn get(&self, key: &K) -> Option<MappedRwLockReadGuard<'_, Vec<V>>> {
        let primary_key = key.primary_key();
        let secondary_key = key.secondary_key();
        let node = self.find_node(primary_key)?;
        let quotient = node.quotients.map.lookup(&secondary_key)?;
        let guard = RwLockReadGuard::try_map(quotient.values.read(), |guard| match guard {
            LazyOmVersionedVec::Vec(vec) => Some(&vec.list),
            LazyOmVersionedVec::FileIndex(_) => None,
        })
        .ok();

        unsafe {
            std::mem::transmute::<
                Option<MappedRwLockReadGuard<'_, Vec<V>>>,
                Option<MappedRwLockReadGuard<'_, Vec<V>>>,
            >(guard)
        }
    }
}

impl<K, V: InlineSerialize> OmTreeMap<K, V> {
    pub fn serialize(&self, version: VersionNumber) -> Result<(), BufIoError> {
        let start = Instant::now();
        let cursor = self.dim_bufman.open_cursor()?;
        self.dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        let offset = self
            .root
            .serialize(&self.dim_bufman, &self.data_bufmans, cursor, version)?;

        self.dim_bufman.seek_with_cursor(cursor, 0)?;
        self.dim_bufman.update_u32_with_cursor(cursor, offset)?;
        self.dim_bufman.close_cursor(cursor)?;
        println!("serialized in {:?}", start.elapsed());
        let start = Instant::now();
        self.dim_bufman.flush()?;
        self.data_bufmans.flush_all()?;
        println!("flushed in {:?}", start.elapsed());
        Ok(())
    }

    pub fn deserialize(
        dim_bufman: BufferManager,
        data_bufmans: BufferManagerFactory<VersionNumber>,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        let offset = dim_bufman.read_u32_with_cursor(cursor)?;
        dim_bufman.close_cursor(cursor)?;

        Ok(Self {
            root: OmTreeMapNode::deserialize(
                &dim_bufman,
                &data_bufmans,
                FileOffset(offset),
                VersionNumber::from(u32::MAX),
            )?,
            dim_bufman,
            data_bufmans,
            _marker: PhantomData,
        })
    }
}

pub struct ShardedOmTreeMap<K, V> {
    pub shards: Vec<OmTreeMap<K, V>>,
}

pub trait ShardIndex {
    fn shard_index(&self) -> usize;
}

impl<K: OmTreeMapKey + ShardIndex, V> ShardedOmTreeMap<K, V> {
    pub fn new(
        root_path: Arc<Path>,
        num_shards: usize,
        name: &'static str,
    ) -> Result<Self, BufIoError> {
        let mut shards = Vec::with_capacity(num_shards);

        for i in 0..num_shards {
            let dim_file = OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(false)
                .create(true)
                .open(root_path.join(format!("{}.dim_{}", name, i)))?;
            let dim_bufman = BufferManager::new(dim_file, 8192)?;
            let data_bufmans = BufferManagerFactory::new(
                root_path.clone(),
                move |root, version: &VersionNumber| {
                    root.join(format!("{}.{}.data_{}", name, **version, i))
                },
                8192,
            );
            let shard = OmTreeMap::new(dim_bufman, data_bufmans);
            shards.push(shard);
        }

        Ok(Self { shards })
    }

    pub fn push(&self, version: VersionNumber, key: &K, value: V) -> bool {
        let shard_index = key.shard_index();
        self.shards[shard_index].push(version, key, value)
    }

    pub fn push_sorted(&self, version: VersionNumber, key: &K, value: V) -> bool
    where
        V: Ord,
    {
        let shard_index = key.shard_index();
        self.shards[shard_index].push_sorted(version, key, value)
    }

    pub fn get(&self, key: &K) -> Option<MappedRwLockReadGuard<'_, Vec<V>>> {
        let shard_index = key.shard_index();
        self.shards[shard_index].get(key)
    }
}

impl<K, V: InlineSerialize + Send + Sync> ShardedOmTreeMap<K, V> {
    pub fn serialize(&self, version: VersionNumber) -> Result<(), BufIoError> {
        self.shards
            .par_iter()
            .try_for_each(|map| map.serialize(version))
    }

    pub fn deserialize() -> Result<Self, BufIoError> {
        todo!()
    }
}
