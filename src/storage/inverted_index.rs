use std::fmt::Debug;
use std::path::Path;
use std::sync::{Arc, RwLock};

use arcshift::ArcShift;

use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::NodeRegistry;
use crate::models::identity_collections::IdentityMapKey;
use crate::models::lazy_load::{LazyItem, LazyItemMap};
use crate::models::serializer::CustomSerialize;

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

/// [InvertedIndexItem] stores non-zero values at `dim_index` dimension of all input vectors
///
/// The `InvertedIndex` struct uses [LazyItemMap] to store data and
/// references to children.
///
/// The use of lazy items internally allows for better performance by deferring the loading of
/// data until it is actually needed. This can be beneficial when dealing with large amounts of data
/// or when the data is expensive to load.
#[derive(Clone)]
pub struct InvertedIndexItem<T>
where
    T: Clone + 'static,
{
    pub dim_index: u32,
    pub implicit: bool,
    pub data: LazyItemMap<T>,
    // TODO: benchmark if fixed size children array with lazy item refs
    //  yields better performance
    pub lazy_children: LazyItemMap<InvertedIndexItem<T>>,
}

impl<T> CustomSerialize for InvertedIndexItem<T>
where
    T: Clone + 'static,
{
    fn serialize(
        &self,
        bufmans: Arc<crate::models::buffered_io::BufferManagerFactory>,
        version: crate::models::versioning::Hash,
        cursor: u64,
    ) -> Result<u32, crate::models::buffered_io::BufIoError> {
        todo!()
    }

    fn deserialize(
        bufmans: Arc<crate::models::buffered_io::BufferManagerFactory>,
        file_index: crate::models::lazy_load::FileIndex,
        cache: Arc<crate::models::cache_loader::NodeRegistry>,
        max_loads: u16,
        skipm: &mut std::collections::HashSet<u64>,
    ) -> Result<Self, crate::models::buffered_io::BufIoError>
    where
        Self: Sized,
    {
        todo!()
    }
}

impl<T> InvertedIndexItem<T>
where
    T: Clone + CustomSerialize + 'static,
{
    /// Creates a new `InvertedIndexItem` with the given dimension index and implicit flag.
    /// Initializes the data vector and children array.
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        InvertedIndexItem {
            dim_index,
            implicit,
            data: LazyItemMap::new(),
            lazy_children: LazyItemMap::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: ArcShift<InvertedIndexItem<T>>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> ArcShift<InvertedIndexItem<T>> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let key = IdentityMapKey::Int(child_index as u32);
            let new_child = LazyItem::new(0.into(), InvertedIndexItem::new(new_dim_index, true));
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(key.clone(), new_child.clone())
                {
                    current_node = child.get_data(cache.clone());
                    break;
                }
            }
        }

        current_node
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Calculates the path and delegates to `insert_with_path`.
    pub fn insert(node: ArcShift<InvertedIndexItem<T>>, value: T, vector_id: u32) {
        let key = IdentityMapKey::Int(vector_id);
        let value = LazyItem::new(0.into(), value.clone());
        node.data.insert(key, value);
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Calculates the path and delegates to `get_value`.
    pub fn get(&self, dim_index: u32, vector_id: u32, cache: Arc<NodeRegistry>) -> Option<T> {
        let path = calculate_path(dim_index, self.dim_index);
        self.get_value(&path, vector_id, cache)
    }

    /// Retrieves a value from the index following the specified path.
    /// Recursively traverses child nodes or searches the data vector.
    fn get_value(&self, path: &[usize], vector_id: u32, cache: Arc<NodeRegistry>) -> Option<T> {
        match path.get(0) {
            Some(child_index) => self
                .lazy_children
                .get(&IdentityMapKey::Int((*child_index) as u32))
                .map(|data| {
                    data.get_data(cache.clone())
                        .get_value(&path[1..], vector_id, cache)
                })
                .flatten(),
            None => self
                .data
                .get(&IdentityMapKey::Int(vector_id))
                .map(|lazy_item| lazy_item.get_data(cache.clone()).get().clone()),
        }
    }
}

impl<T> InvertedIndexItem<T>
where
    T: Debug + CustomSerialize + Clone,
{
    /// Prints the tree structure of the index starting from the current node.
    /// Recursively prints child nodes with increased indentation.
    ///
    /// # Safety:
    /// The function does not modify the structure but accesses internal state.
    /// It is meant only for debugging and testing purposes
    pub fn print_tree(&mut self, depth: usize, prev_dim_index: u32, cache: Arc<NodeRegistry>) {}
    // pub fn print_tree(&mut self, depth: usize, prev_dim_index: u32, cache: Arc<NodeRegistry>) {
    //     let indent = "  ".repeat(depth);
    //     let dim_index = prev_dim_index + self.dim_index;
    //     println!(
    //         "{}Dimension-Index {}: {}",
    //         indent,
    //         dim_index,
    //         if self.implicit {
    //             "Implicit"
    //         } else {
    //             "Explicit"
    //         }
    //     );
    //     println!(
    //         "{}Data (value, vector_id): {:?}",
    //         indent,
    //         self.data
    //             .items
    //             .get()
    //             .iter()
    //             .map(|(k, v)| {
    //                 let val = v.get_data(cache.clone()).get().clone();
    //                 (val, k.clone())
    //             })
    //             .collect::<Vec<_>>()
    //     );
    //     for (i, (_, child)) in self.lazy_children.items.get().iter().enumerate() {
    //         let mut item = child.get_data(cache.clone());
    //         let item = item.get();
    //         println!("{}-> 4^{} to:", indent, i);
    //         item.print_tree(depth + 1, dim_index, cache.clone());
    //     }
    // }
}

#[derive(Clone)]
pub struct InvertedIndex<T>
where
    T: Clone + 'static,
{
    pub root: ArcShift<InvertedIndexItem<T>>,
    cache: Arc<NodeRegistry>,
}

impl<T> InvertedIndex<T>
where
    T: Clone + CustomSerialize + 'static,
{
    /// Creates a new `InvertedIndex` with an initial root node.
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(Path::new(".").into()));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndex {
            root: ArcShift::new(InvertedIndexItem::new(0, false)),
            cache,
        }
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Delegates to the root node's `get` method.
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<T> {
        self.root
            .shared_get()
            .get(dim_index, vector_id, self.cache.clone())
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Delegates to the root node's `insert` method.
    pub fn insert(&self, dim_index: u32, value: T, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node =
            InvertedIndexItem::find_or_create_node(self.root.clone(), &path, self.cache.clone());
        InvertedIndexItem::insert(node, value, vector_id)
    }
}

impl<T> InvertedIndex<T>
where
    T: Debug + Clone + 'static,
{
    /// Prints the tree structure of the entire index.
    /// Delegates to the root node's `print_tree` method.
    pub fn print_tree(&mut self) {
        // self.root.get().print_tree(0, 0);
    }
}

impl InvertedIndex<f32> {
    /// Adds a sparse vector to the index.
    /// Iterates over the vector and inserts non-zero values.
    pub fn add_sparse_vector(&self, vector: Vec<f32>, vector_id: u32) -> Result<(), String> {
        for (dim_index, &value) in vector.iter().enumerate() {
            if value != 0.0 {
                self.insert(dim_index as u32, value, vector_id)
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use quickcheck::{Arbitrary, Gen};
    use quickcheck_macros::quickcheck;

    use std::{collections::HashMap, thread};

    #[test]
    fn test_calculate_path() {
        assert_eq!(calculate_path(16, 0), vec![2]);
        assert_eq!(calculate_path(20, 0), vec![2, 1]);
        assert_eq!(calculate_path(2, 0), vec![0, 0]);
        assert_eq!(calculate_path(64, 16), vec![2, 2, 2]);
    }

    #[quickcheck]
    fn prop_calculate_path_correctness(target_dim_index: u32, current_dim_index: u32) -> bool {
        if target_dim_index < current_dim_index {
            return true; // Skip invalid cases
        }
        let path = calculate_path(target_dim_index, current_dim_index);
        let sum: u32 = path.iter().map(|&index| POWERS_OF_4[index]).sum();
        sum == target_dim_index - current_dim_index
    }

    #[test]
    /// Prints the structure of the `InvertedIndex` after adding sparse vectors.
    pub fn small_test() {
        let mut inverted_index: InvertedIndex<f32> = InvertedIndex::new();

        let sparse_vector1 = vec![0.0, 15.0];

        inverted_index.add_sparse_vector(sparse_vector1, 0).unwrap();

        println!("\nFinal Inverted Index structure:");
        inverted_index.print_tree();

        // Look up using dimension index and vector ID
        assert_eq!(inverted_index.get(1, 0), Some(15.0));
    }

    #[test]
    /// Prints the structure of the `InvertedIndex` after adding sparse vectors.
    pub fn simple_test() {
        let mut inverted_index: InvertedIndex<f32> = InvertedIndex::new();

        let sparse_vector1 = vec![0.0, 15.0, 0.0, 0.0, 27.0, 0.0, 31.0, 0.0, 0.0, 42.0];
        let sparse_vector2 = vec![1.0, 0.0, 23.0, 0.0, 0.0, 38.0, 0.0, 45.0, 0.0, 56.0];

        inverted_index.add_sparse_vector(sparse_vector1, 0).unwrap();
        inverted_index.add_sparse_vector(sparse_vector2, 1).unwrap();

        println!("\nFinal Inverted Index structure:");
        inverted_index.print_tree();

        // Look up using dimension index and vector ID
        assert_eq!(inverted_index.get(1, 0), Some(15.0));
        assert_eq!(inverted_index.get(4, 0), Some(27.0));
        assert_eq!(inverted_index.get(6, 0), Some(31.0));
        assert_eq!(inverted_index.get(9, 0), Some(42.0));

        assert_eq!(inverted_index.get(0, 1), Some(1.0));
        assert_eq!(inverted_index.get(2, 1), Some(23.0));
        assert_eq!(inverted_index.get(5, 1), Some(38.0));
        assert_eq!(inverted_index.get(7, 1), Some(45.0));
        assert_eq!(inverted_index.get(9, 1), Some(56.0));
    }

    // Custom type for generating sparse vectors
    #[derive(Clone, Debug)]
    struct SparseVector(Vec<f32>);

    impl Arbitrary for SparseVector {
        /// Generates arbitrary sparse vectors for testing.
        fn arbitrary(g: &mut Gen) -> Self {
            let size = usize::arbitrary(g) % (20 - 10) + 10;
            let mut vec = vec![0.0; size];
            for _ in 0..size / 4 {
                // Make it sparse by only setting ~25% of elements
                let index = usize::arbitrary(g) % size;
                let mut val = f32::arbitrary(g);
                if val.is_nan() {
                    val = 0.0
                };
                vec[index] = val;
            }
            SparseVector(vec)
        }
    }

    /// Converts a vector to a HashMap of non-zero elements.
    fn vector_to_hashmap(vec: &[f32]) -> HashMap<usize, f32> {
        vec.iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, &v)| (i, v))
            .collect()
    }

    /// Property-based test: Verifies handling of multiple vectors.
    /// Property-based test: Verifies insertion correctness and vector ID preservation.
    #[quickcheck]
    fn prop_multiple_vector_handling(vectors: Vec<SparseVector>) -> bool {
        let index = InvertedIndex::new();

        for (id, SparseVector(vec)) in vectors.iter().enumerate() {
            index.add_sparse_vector(vec.clone(), id as u32).unwrap();
        }

        // Verify each vector is stored separately
        vectors.iter().enumerate().all(|(id, SparseVector(vec))| {
            vector_to_hashmap(vec).iter().all(|(&dim, &value)| {
                index
                    .get(dim as u32, id as u32)
                    .map_or(false, |v| v == value)
            })
        })
    }

    /// Tests for a subtle race condition in `find_or_create_node`.
    ///
    /// A thread can acquire read lock on child to check if it exists. If it
    /// doesn't it drops the lock and acquires a write lock to create the child.
    /// In between, a different thread could have created the child node.
    ///
    /// The race condition is triggered when locking pattern is like this -
    /// r1, r2, w1, w2.
    #[test]
    fn test_find_or_create_race_condition() {
        for _ in 0..1000 {
            let root: ArcShift<InvertedIndexItem<f32>> =
                ArcShift::new(InvertedIndexItem::new(0, false));
            let root_clone1 = root.clone();
            let mut root_clone2 = root.clone();
            let bufmans = Arc::new(BufferManagerFactory::new(Path::new(".").into()));
            let cache = Arc::new(NodeRegistry::new(1000, bufmans));
            let cache1 = cache.clone();
            let cache2 = cache.clone();

            // Insert a value for dimension 5
            // Creates child 1 in depth 1 and child 0 in depth 0.
            let handle2 = thread::spawn(move || {
                let path = vec![0, 1];
                InvertedIndexItem::find_or_create_node(root_clone1, &path, cache1);
            });

            // Insert a value for dimension 1
            // Creates child 0 in depth 0 but no other child
            // Can overwrite child 0's children created by handle 2
            let handle1 = thread::spawn(move || {
                let path = vec![0];
                InvertedIndexItem::find_or_create_node(root, &path, cache2);
            });

            handle1.join().unwrap();
            handle2.join().unwrap();

            // Check the final state
            root_clone2.reload();
            let node = root_clone2.shared_get();

            let child_0 = node
                .lazy_children
                .get(&IdentityMapKey::Int(0 as u32))
                .map(|data| data.get_data(cache.clone()));
            assert!(child_0.is_some(), "Child 0 should exist");
            let child_0 = child_0.unwrap();

            let child_1 = child_0
                .shared_get()
                .lazy_children
                .get(&IdentityMapKey::Int(1 as u32))
                .map(|data| data.get_data(cache.clone()));
            assert!(child_1.is_some(), "Child 1 should exist");
        }
    }
}
