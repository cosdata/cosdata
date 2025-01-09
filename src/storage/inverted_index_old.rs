use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use dashmap::DashMap;

use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::{Cacheable, NodeRegistry};
use crate::models::identity_collections::IdentityMapKey;
use crate::models::lazy_load::{LazyItem, LazyItemArray};
use crate::models::serializer::CustomSerialize;
use crate::models::types::SparseVector;
use crate::models::versioning::Hash;

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
    pub data: Arc<DashMap<IdentityMapKey, LazyItem<T>>>,
    pub lazy_children: LazyItemArray<InvertedIndexItem<T>, 16>,
}

impl<T> InvertedIndexItem<T>
where
    T: Clone + Cacheable + CustomSerialize + 'static,
    InvertedIndexItem<T>: CustomSerialize + Cacheable,
{
    /// Creates a new `InvertedIndexItem` with the given dimension index and implicit flag.
    /// Initializes the data vector and children array.
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        InvertedIndexItem {
            dim_index,
            implicit,
            data: Arc::new(DashMap::new()),
            lazy_children: LazyItemArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: Arc<InvertedIndexItem<T>>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> Arc<InvertedIndexItem<T>> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let new_child = LazyItem::new(0.into(), 0, InvertedIndexItem::new(new_dim_index, true));
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(child_index, new_child.clone())
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
    pub fn insert(node: Arc<InvertedIndexItem<T>>, value: T, vector_id: u32) {
        let key = IdentityMapKey::Int(vector_id);
        let value = LazyItem::new(0.into(), 0, value.clone());
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
                .get(*child_index)
                .map(|data| {
                    data.get_data(cache.clone())
                        .get_value(&path[1..], vector_id, cache)
                })
                .flatten(),
            None => self
                .data
                .get(&IdentityMapKey::Int(vector_id))
                .map(|lazy_item| (*lazy_item.get_data(cache.clone())).clone()),
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
    pub fn print_tree(&mut self, _depth: usize, _prev_dim_index: u32, _cache: Arc<NodeRegistry>) {}
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
    pub root: Arc<InvertedIndexItem<T>>,
    pub cache: Arc<NodeRegistry>,
}

impl<T> InvertedIndex<T>
where
    T: Cacheable + Clone + CustomSerialize + 'static,
    InvertedIndexItem<T>: CustomSerialize + Cacheable,
{
    /// Creates a new `InvertedIndex` with an initial root node.
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(
            Path::new(".").into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            1.0,
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndex {
            root: Arc::new(InvertedIndexItem::new(0, false)),
            cache,
        }
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<Arc<InvertedIndexItem<T>>> {
        let mut current_node = self.root.clone();
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            current_node = child.get_data(self.cache.clone());
        }

        Some(current_node)
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Delegates to the root node's `get` method.
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<T> {
        self.root.get(dim_index, vector_id, self.cache.clone())
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
    pub fn add_sparse_vector(&self, vector: SparseVector) -> Result<(), String> {
        let vector_id = vector.vector_id;
        for (dim_index, value) in vector.entries.iter() {
            if *value != 0.0 {
                self.insert(*dim_index, *value, vector_id);
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

    use std::{
        collections::{BTreeMap, HashMap},
        thread,
    };

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

        let sparse_vector1 = SparseVector::new(0, vec![(0, 0.0), (1, 15.0)]);

        inverted_index.add_sparse_vector(sparse_vector1).unwrap();

        println!("\nFinal Inverted Index structure:");
        inverted_index.print_tree();

        // Look up using dimension index and vector ID
        assert_eq!(inverted_index.get(1, 0), Some(15.0));
    }

    #[test]
    /// Prints the structure of the `InvertedIndex` after adding sparse vectors.
    pub fn simple_test() {
        let mut inverted_index: InvertedIndex<f32> = InvertedIndex::new();

        let sparse_vector1 = SparseVector::new(0, vec![(1, 15.0), (4, 27.0), (6, 31.0), (9, 42.0)]);
        let sparse_vector2 = SparseVector::new(
            1,
            vec![(0, 1.0), (2, 23.0), (5, 38.0), (7, 45.0), (9, 56.0)],
        );

        inverted_index.add_sparse_vector(sparse_vector1).unwrap();
        inverted_index.add_sparse_vector(sparse_vector2).unwrap();

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

    impl Arbitrary for SparseVector {
        /// Generates arbitrary sparse vectors for testing.
        fn arbitrary(g: &mut Gen) -> Self {
            let size = usize::arbitrary(g) % (20 - 10) + 10;
            let mut vec: Vec<(u32, f32)> = Vec::with_capacity(size);
            let mut entries: BTreeMap<usize, f32> = BTreeMap::new();

            for _ in 0..size / 4 {
                // Make it sparse by only setting ~25% of elements
                let index = usize::arbitrary(g) % size;
                let val = f32::arbitrary(g);
                if !val.is_nan() && val != 0.0 {
                    entries.insert(index, val);
                };
            }

            for (index, &val) in entries.iter() {
                vec.push((*index as u32, val));
            }

            SparseVector::new(rand::random(), vec)
        }
    }

    /// Converts a vector to a HashMap of non-zero elements.
    #[allow(dead_code)]
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

        for vector in &vectors {
            let _ = index.add_sparse_vector(vector.clone());
        }

        // Verify each vector is stored separately
        vectors.iter().all(|vector| {
            let id = vector.vector_id;
            vector
                .entries
                .iter()
                .all(|(dim, value)| index.get(*dim, id).map_or(false, |v| v == *value))
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
            let root: Arc<InvertedIndexItem<f32>> = Arc::new(InvertedIndexItem::new(0, false));
            let root_clone1 = root.clone();
            let root_clone2 = root.clone();
            let bufmans = Arc::new(BufferManagerFactory::new(
                Path::new(".").into(),
                |root, ver: &Hash| root.join(format!("{}.index", **ver)),
                1.0,
            ));
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

            let child_0 = root_clone2
                .lazy_children
                .get(0)
                .map(|data| data.get_data(cache.clone()));
            assert!(child_0.is_some(), "Child 0 should exist");
            let child_0 = child_0.unwrap();

            let child_1 = child_0
                .lazy_children
                .get(1)
                .map(|data| data.get_data(cache.clone()));
            assert!(child_1.is_some(), "Child 1 should exist");
        }
    }
}
