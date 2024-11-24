use arcshift::ArcShift;
use core::array::from_fn;
use core::hash::Hash;
use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hasher};
use std::io::SeekFrom;
use std::thread;
use std::{path::Path, sync::RwLock};

use std::sync::{Arc, Mutex};

use crate::models::{
    buffered_io::BufferManagerFactory,
    cache_loader::NodeRegistry,
    lazy_load::{FileIndex, LazyItem, LazyItemArray},
    serializer::CustomSerialize,
    types::{FileOffset, SparseVector},
};

use super::page::Pagepool;

// Size of a page in the hash table
const PAGE_SIZE: usize = 32;

// TODO: Add more powers for larger jumps
// TODO: Or switch to dynamic calculation of power of max power of 4
const POWERS_OF_4: [u32; 8] = [1, 4, 16, 64, 256, 1024, 4096, 16384];

/// Returns the largest power of 4 that is less than or equal to `n`.
/// Iteratively multiplies by 4 until the result exceeds `n`.
pub fn largest_power_of_4_below(n: u32) -> (usize, u32) {
    assert_ne!(n, 0, "Cannot find largest power of 4 below 0");
    POWERS_OF_4
        .into_iter()
        .enumerate()
        .rev()
        .find(|&(_, pow4)| pow4 <= n)
        .unwrap()
}

/// Calculates the path from `current_dim_index` to `target_dim_index`.
/// Decomposes the difference into powers of 4 and returns the indices.
fn calculate_path(target_dim_index: u32, current_dim_index: u32) -> Vec<usize> {
    let mut path = Vec::new();
    let mut remaining = target_dim_index - current_dim_index;

    while remaining > 0 {
        let (child_index, pow_4) = largest_power_of_4_below(remaining);
        path.push(child_index);
        remaining -= pow_4;
    }

    path
}

/// [InvertedIndexSparseAnnNodeBasic] is a node in InvertedIndexSparseAnnBasic structure
/// data in InvertedIndexSparseAnnNode holds list of Vec_Ids corresponding to the quantized u8 value (which is the index of array)
#[derive(Clone)]
pub struct InvertedIndexSparseAnnNodeBasic {
    pub dim_index: u32,
    pub implicit: bool,
    pub data: [Arc<RwLock<Vec<LazyItem<u32>>>>; 64],
    pub lazy_children: LazyItemArray<InvertedIndexSparseAnnNodeBasic, 16>,
}

impl InvertedIndexSparseAnnNodeBasic {
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        let data: [Arc<RwLock<Vec<LazyItem<u32>>>>; 64] =
            from_fn(|_| Arc::new(RwLock::new(Vec::new())));

        InvertedIndexSparseAnnNodeBasic {
            dim_index,
            implicit,
            data,
            lazy_children: LazyItemArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: ArcShift<InvertedIndexSparseAnnNodeBasic>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> ArcShift<InvertedIndexSparseAnnNodeBasic> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let new_child = LazyItem::new(
                0.into(),
                0u16,
                InvertedIndexSparseAnnNodeBasic::new(new_dim_index, true),
            );
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(child_index, new_child.clone())
                {
                    let res: Arc<InvertedIndexSparseAnnNodeBasic> = child.get_data(cache.clone());
                    current_node = ArcShift::new((*res).clone());
                    break;
                }
            }
        }

        current_node
    }

    pub fn quantize(value: f32) -> u8 {
        ((value * 63.0).clamp(0.0, 63.0) as u8).min(63)
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Finds the quantized value and pushes the vec_Id in array at index = quantized_value
    pub fn insert(node: ArcShift<InvertedIndexSparseAnnNodeBasic>, value: f32, vector_id: u32) {
        let quantized_value = Self::quantize(value);
        let data = node.data.clone();

        // Insert into the specific LazyItem at the index quantized_value
        if let Some(arc_lazy_item) = data.get(quantized_value as usize) {
            let mut vec = arc_lazy_item.write().unwrap();
            vec.push(LazyItem::new(0.into(), 0u16, vector_id));
        }
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Calculates the path and delegates to `get_value`.
    pub fn get(&self, dim_index: u32, vector_id: u32, cache: Arc<NodeRegistry>) -> Option<u8> {
        let path = calculate_path(dim_index, self.dim_index);
        self.get_value(&path, vector_id, cache)
    }

    /// Retrieves a value from the index following the specified path.
    /// Recursively traverses child nodes or searches the data vector.
    fn get_value(&self, path: &[usize], vector_id: u32, cache: Arc<NodeRegistry>) -> Option<u8> {
        match path.get(0) {
            Some(child_index) => self
                .lazy_children
                .get(*child_index)
                .map(|data| {
                    data.get_data(cache.clone())
                        .get_value(&path[1..], vector_id, cache)
                })
                .flatten(),
            None => {
                for (index, arc_rwlock_lazy_item) in self.data.iter().enumerate() {
                    let arc_rwlock_lazy_item = arc_rwlock_lazy_item.read().unwrap();
                    if arc_rwlock_lazy_item
                        .iter()
                        .any(|item| *item.get_data(cache.clone()) == vector_id)
                    {
                        return Some(index as u8);
                    }
                }
                None
            }
        }
    }
}

/// [InvertedIndexSparseAnnBasic] is a improved version which only holds quantized u8 values instead of f32 inside [InvertedIndexSparseAnnNodeBasic]
#[derive(Clone)]
pub struct InvertedIndexSparseAnnBasic {
    pub root: ArcShift<InvertedIndexSparseAnnNodeBasic>,
    pub cache: Arc<NodeRegistry>,
}

impl InvertedIndexSparseAnnBasic {
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(
            Path::new(".").into(),
            |root, ver| root.join(format!("{}.index", **ver)),
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndexSparseAnnBasic {
            root: ArcShift::new(InvertedIndexSparseAnnNodeBasic::new(0, false)),
            cache,
        }
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<ArcShift<InvertedIndexSparseAnnNodeBasic>> {
        let mut current_node = self.root.clone();
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            let node_res = child.get_data(self.cache.clone());
            current_node = ArcShift::new((*node_res).clone());
        }

        Some(current_node)
    }

    //Fetches quantized u8 value for a dim_index and vector_Id present at respective node in index
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<u8> {
        self.root
            .shared_get()
            .get(dim_index, vector_id, self.cache.clone())
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = InvertedIndexSparseAnnNodeBasic::find_or_create_node(
            self.root.clone(),
            &path,
            self.cache.clone(),
        );
        //value will be quantized while being inserted into the Node.
        InvertedIndexSparseAnnNodeBasic::insert(node, value, vector_id)
    }

    /// Adds a sparse vector to the index.
    pub fn add_sparse_vector(&self, vector: SparseVector) -> Result<(), String> {
        let vector_id = vector.vector_id;
        vector.entries.par_iter().for_each(|(dim_index, value)| {
            if *value != 0.0 {
                self.insert(*dim_index, *value, vector_id);
            }
        });
        Ok(())
    }
}

#[derive(Clone)]
pub struct InvertedIndexSparseAnnNodeBasicTSHashmap {
    pub dim_index: u32,
    pub implicit: bool,
    pub data: TSHashTable<u8, Pagepool<PAGE_SIZE>>,
    pub lazy_children: LazyItemArray<InvertedIndexSparseAnnNodeBasicTSHashmap, 16>,
}

#[derive(Clone)]
pub struct InvertedIndexSparseAnnBasicTSHashmap {
    pub root: ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>,
    pub cache: Arc<NodeRegistry>,
}

impl InvertedIndexSparseAnnNodeBasicTSHashmap {
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        let data = TSHashTable::new(16);

        InvertedIndexSparseAnnNodeBasicTSHashmap {
            dim_index,
            implicit,
            data,
            lazy_children: LazyItemArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let new_child = LazyItem::new(
                0.into(),
                0u16,
                InvertedIndexSparseAnnNodeBasicTSHashmap::new(new_dim_index, true),
            );
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(child_index, new_child.clone())
                {
                    let res: Arc<InvertedIndexSparseAnnNodeBasicTSHashmap> =
                        child.get_data(cache.clone());
                    current_node = ArcShift::new((*res).clone());
                    break;
                }
            }
        }

        current_node
    }

    pub fn quantize(value: f32) -> u8 {
        ((value * 63.0).clamp(0.0, 63.0) as u8).min(63)
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Finds the quantized value and pushes the vec_Id in array at index = quantized_value
    pub fn insert(
        node: ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>,
        value: f32,
        vector_id: u32,
    ) {
        let quantized_value = Self::quantize(value);
        let data = node.data.clone();
        data.get_or_create(quantized_value, || Pagepool::default());
        data.mutate(quantized_value, |x| {
            let mut vecof_vec_id = x.unwrap();
            vecof_vec_id.push(vector_id);
            Some(vecof_vec_id)
        });
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Calculates the path and delegates to `get_value`.
    pub fn get(&self, dim_index: u32, vector_id: u32, cache: Arc<NodeRegistry>) -> Option<u8> {
        let path = calculate_path(dim_index, self.dim_index);
        self.get_value(&path, vector_id, cache)
    }

    /// Retrieves a value from the index following the specified path.
    /// Recursively traverses child nodes or searches the data vector.
    fn get_value(&self, path: &[usize], vector_id: u32, cache: Arc<NodeRegistry>) -> Option<u8> {
        match path.get(0) {
            Some(child_index) => self
                .lazy_children
                .get(*child_index)
                .map(|data| {
                    data.get_data(cache.clone())
                        .get_value(&path[1..], vector_id, cache)
                })
                .flatten(),
            None => {
                let res = self.data.to_list();
                for (x, y) in res {
                    if y.contains(vector_id) {
                        return Some(x);
                    }
                }
                None
            }
        }
    }
}

impl InvertedIndexSparseAnnBasicTSHashmap {
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(
            Path::new(".").into(),
            |root, ver| root.join(format!("{}.index", **ver)),
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndexSparseAnnBasicTSHashmap {
            root: ArcShift::new(InvertedIndexSparseAnnNodeBasicTSHashmap::new(0, false)),
            cache,
        }
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(
        &self,
        dim_index: u32,
    ) -> Option<ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>> {
        let mut current_node = self.root.clone();
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            let node_res = child.get_data(self.cache.clone());
            current_node = ArcShift::new((*node_res).clone());
        }

        Some(current_node)
    }

    //Fetches quantized u8 value for a dim_index and vector_Id present at respective node in index
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<u8> {
        self.root
            .shared_get()
            .get(dim_index, vector_id, self.cache.clone())
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = InvertedIndexSparseAnnNodeBasicTSHashmap::find_or_create_node(
            self.root.clone(),
            &path,
            self.cache.clone(),
        );
        //value will be quantized while being inserted into the Node.
        InvertedIndexSparseAnnNodeBasicTSHashmap::insert(node, value, vector_id)
    }

    /// Adds a sparse vector to the index.
    pub fn add_sparse_vector(&self, vector: SparseVector) -> Result<(), String> {
        let vector_id = vector.vector_id;
        vector.entries.par_iter().for_each(|(dim_index, value)| {
            if *value != 0.0 {
                self.insert(*dim_index, *value, vector_id);
            }
        });
        Ok(())
    }
}

type HashTable<K, V> = HashMap<K, V>;

/// This is a custom Hashtable made to use for data variable in Node of InvertedIndex
#[derive(Clone)]
pub struct TSHashTable<K, V> {
    hash_table_list: Vec<Arc<Mutex<HashTable<K, V>>>>,
    size: i16,
}

impl<K, V> CustomSerialize for TSHashTable<K, V>
where
    K: CustomSerialize + Eq + Hash,
    V: CustomSerialize,
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: crate::models::versioning::Hash,
        cursor: u64,
    ) -> Result<u32, crate::models::buffered_io::BufIoError> {
        let bufman = bufmans.get(&version)?;

        // Move the cursor to the end of the file and start writing from there
        bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

        let start_offset = bufman.cursor_position(cursor)? as u32;

        // Write the size of the list at first
        bufman.write_u32_with_cursor(cursor, self.size as u32)?;

        for i in &self.hash_table_list {
            let item = i
                .lock()
                .map_err(|_| crate::models::buffered_io::BufIoError::Locking)?;

            // Write the size of the hashmap
            bufman.write_u32_with_cursor(cursor, item.len() as u32)?;

            for (key, value) in item.iter() {
                key.serialize(bufmans.clone(), version, cursor)?;
                value.serialize(bufmans.clone(), version, cursor)?;
            }
        }
        Ok(start_offset)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: crate::models::lazy_load::FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut std::collections::HashSet<u64>,
    ) -> Result<Self, crate::models::buffered_io::BufIoError> {
        match file_index {
            crate::models::lazy_load::FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                version_number,
            } => {
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;

                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

                // Read the length of the vec
                let len = bufman.read_u32_with_cursor(cursor)?;

                let mut vec = Vec::<HashTable<K, V>>::with_capacity(len as usize);

                for _ in 0..len as usize {
                    let table_len = bufman.read_u32_with_cursor(cursor)? as usize;

                    let mut hash_table = HashTable::<K, V>::with_capacity(table_len);
                    for _ in 0..table_len {
                        let current_offset = bufman.cursor_position(cursor)?;

                        let file_index = FileIndex::Valid {
                            offset: FileOffset(current_offset as u32),
                            version_id,
                            version_number,
                        };

                        let key = K::deserialize(
                            bufmans.clone(),
                            file_index.clone(),
                            cache.clone(),
                            max_loads,
                            skipm,
                        )?;

                        let value = V::deserialize(
                            bufmans.clone(),
                            file_index,
                            cache.clone(),
                            max_loads,
                            skipm,
                        )?;

                        hash_table.insert(key, value);
                    }

                    vec.push(hash_table);
                }

                bufman.close_cursor(cursor)?;

                Ok(Self {
                    hash_table_list: vec
                        .into_iter()
                        .map(|item| Arc::new(Mutex::new(item)))
                        .collect(),
                    size: len as i16,
                })
            }

            FileIndex::Invalid => Err(crate::models::buffered_io::BufIoError::Locking),
        }
    }
}

impl<K: Eq + Hash, V> TSHashTable<K, V> {
    pub fn new(size: i16) -> Self {
        let hash_table_list = (0..size)
            .map(|_| Arc::new(Mutex::new(HashMap::new())))
            .collect();
        TSHashTable {
            hash_table_list,
            size,
        }
    }
    pub fn hash_key(&self, k: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        (hasher.finish() as usize) % (self.size as usize)
    }

    pub fn insert(&self, k: K, v: V) {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        ht.insert(k, v);
    }

    pub fn delete(&self, k: &K) {
        let index = self.hash_key(k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        ht.remove(k);
    }

    pub fn lookup(&self, k: &K) -> Option<V>
    where
        V: Clone,
    {
        let index = self.hash_key(k);
        let ht = self.hash_table_list[index].lock().unwrap();
        ht.get(k).cloned()
    }

    pub fn mutate<F>(&self, k: K, f: F)
    where
        F: FnOnce(Option<V>) -> Option<V>,
        V: Clone,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        let v = ht.remove(&k);
        let new_v = f(v);
        if let Some(new_v) = new_v {
            ht.insert(k, new_v);
        }
    }

    pub fn get_or_create<F>(&self, k: K, f: F) -> V
    where
        F: FnOnce() -> V,
        V: Clone,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        match ht.get(&k) {
            Some(v) => v.clone(),
            None => {
                let new_v = f();
                ht.insert(k, new_v.clone());
                new_v
            }
        }
    }

    pub fn map_m<F>(&self, f: F)
    where
        F: Fn(&K, &V) + Send + Sync + 'static,
        K: Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        let f = Arc::new(f);
        let handles: Vec<_> = self
            .hash_table_list
            .iter()
            .map(|ht| {
                let ht = Arc::clone(ht);
                let f = f.clone();
                thread::spawn(move || {
                    let ht = ht.lock().unwrap();
                    for (k, v) in ht.iter() {
                        f(k, v);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    pub fn to_list(&self) -> Vec<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        self.hash_table_list
            .iter()
            .flat_map(|ht| {
                let ht = ht.lock().unwrap();
                ht.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn from_list(size: i16, kv: Vec<(K, V)>) -> Self {
        let tsh = Self::new(size);
        for (k, v) in kv {
            tsh.insert(k, v);
        }
        tsh
    }

    pub fn purge_all(&self) {
        for ht in &self.hash_table_list {
            let mut ht = ht.lock().unwrap();
            *ht = HashMap::new();
        }
    }
}

#[derive(Clone)]
pub struct InvertedIndexSparseAnnNodeBasicDashMap {
    pub dim_index: u32,
    pub implicit: bool,
    pub data: DashMap<u32, u8>,
    pub lazy_children: LazyItemArray<InvertedIndexSparseAnnNodeBasicDashMap, 16>,
}

#[derive(Clone)]
pub struct InvertedIndexSparseAnnBasicDashMap {
    pub root: ArcShift<InvertedIndexSparseAnnNodeBasicDashMap>,
    pub cache: Arc<NodeRegistry>,
}

impl InvertedIndexSparseAnnNodeBasicDashMap {
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        let data: DashMap<u32, u8> = DashMap::new();

        InvertedIndexSparseAnnNodeBasicDashMap {
            dim_index,
            implicit,
            data,
            lazy_children: LazyItemArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: ArcShift<InvertedIndexSparseAnnNodeBasicDashMap>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> ArcShift<InvertedIndexSparseAnnNodeBasicDashMap> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let new_child = LazyItem::new(
                0.into(),
                0u16,
                InvertedIndexSparseAnnNodeBasicDashMap::new(new_dim_index, true),
            );
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(child_index, new_child.clone())
                {
                    let res: Arc<InvertedIndexSparseAnnNodeBasicDashMap> =
                        child.get_data(cache.clone());
                    current_node = ArcShift::new((*res).clone());
                    break;
                }
            }
        }

        current_node
    }

    pub fn quantize(value: f32) -> u8 {
        ((value * 63.0).clamp(0.0, 63.0) as u8).min(63)
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Finds the quantized value and pushes the vec_Id in array at index = quantized_value
    pub fn insert(
        node: ArcShift<InvertedIndexSparseAnnNodeBasicDashMap>,
        value: f32,
        vector_id: u32,
    ) {
        let quantized_value = Self::quantize(value);
        let data = node.data.clone();
        data.insert(vector_id, quantized_value);
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Calculates the path and delegates to `get_value`.
    pub fn get(&self, dim_index: u32, vector_id: u32, cache: Arc<NodeRegistry>) -> Option<u8> {
        let path = calculate_path(dim_index, self.dim_index);
        self.get_value(&path, vector_id, cache)
    }

    /// Retrieves a value from the index following the specified path.
    /// Recursively traverses child nodes or searches the data vector.
    fn get_value(&self, path: &[usize], vector_id: u32, cache: Arc<NodeRegistry>) -> Option<u8> {
        match path.get(0) {
            Some(child_index) => self
                .lazy_children
                .get(*child_index)
                .map(|data| {
                    data.get_data(cache.clone())
                        .get_value(&path[1..], vector_id, cache)
                })
                .flatten(),
            None => {
                let res = self.data.get(&vector_id);
                match res {
                    Some(val) => {
                        let p = *val;
                        return Some(p);
                    }
                    None => return None,
                }
            }
        }
    }
}

impl InvertedIndexSparseAnnBasicDashMap {
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(
            Path::new(".").into(),
            |root, ver| root.join(format!("{}.index", **ver)),
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndexSparseAnnBasicDashMap {
            root: ArcShift::new(InvertedIndexSparseAnnNodeBasicDashMap::new(0, false)),
            cache,
        }
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(
        &self,
        dim_index: u32,
    ) -> Option<ArcShift<InvertedIndexSparseAnnNodeBasicDashMap>> {
        let mut current_node = self.root.clone();
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            let node_res = child.get_data(self.cache.clone());
            current_node = ArcShift::new((*node_res).clone());
        }

        Some(current_node)
    }

    //Fetches quantized u8 value for a dim_index and vector_Id present at respective node in index
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<u8> {
        self.root
            .shared_get()
            .get(dim_index, vector_id, self.cache.clone())
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = InvertedIndexSparseAnnNodeBasicDashMap::find_or_create_node(
            self.root.clone(),
            &path,
            self.cache.clone(),
        );
        //value will be quantized while being inserted into the Node.
        InvertedIndexSparseAnnNodeBasicDashMap::insert(node, value, vector_id)
    }

    /// Adds a sparse vector to the index.
    pub fn add_sparse_vector(&self, vector: SparseVector) -> Result<(), String> {
        let vector_id = vector.vector_id;
        vector.entries.par_iter().for_each(|(dim_index, value)| {
            if *value != 0.0 {
                self.insert(*dim_index, *value, vector_id);
            }
        });
        Ok(())
    }
}
