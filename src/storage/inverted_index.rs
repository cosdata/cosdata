use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use crate::models::identity_collections::IdentityMapKey;
use crate::models::lazy_load::{LazyItem, LazyItemMap};

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

pub struct InvertedIndexItem<T>
where
    T: Clone + 'static,
{
    pub dim_index: u32,
    pub implicit: bool,
    pub data: LazyItemMap<T>,
    pub children: [Option<SharedInvertedIndexItem<T>>; 8],
    // pub lazy_children: LazyItem<[Option<SharedInvertedIndexItem<T>>; 8]>,
}

pub type SharedInvertedIndexItem<T> = Arc<RwLock<InvertedIndexItem<T>>>;

impl<T> InvertedIndexItem<T>
where
    T: Copy,
{
    /// Creates a new `InvertedIndexItem` with the given dimension index and implicit flag.
    /// Initializes the data vector and children array.
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        InvertedIndexItem {
            dim_index,
            implicit,
            data: LazyItemMap::new(),
            children: Default::default(),
            // lazy_children: LazyItem::new(0.into(), Default::default()),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: SharedInvertedIndexItem<T>,
        path: &[usize],
    ) -> SharedInvertedIndexItem<T> {
        let mut current_node = node;
        for &child_index in path {
            current_node = {
                let read_guard = current_node.read().unwrap();
                let child = read_guard.children.get(child_index);
                let dim_index = read_guard.dim_index;
                match child {
                    Some(Some(child_node)) => child_node.clone(),
                    Some(None) => {
                        drop(read_guard);
                        let mut write_guard = current_node.write().unwrap();
                        if let Some(child) = &write_guard.children[child_index] {
                            // Check if another thread created the child node
                            child.clone()
                        } else {
                            let new_dim_index = dim_index + POWERS_OF_4[child_index];
                            let new_item =
                                Arc::new(RwLock::new(InvertedIndexItem::new(new_dim_index, true)));
                            write_guard.children[child_index] = Some(new_item.clone());
                            new_item
                        }
                    }
                    None => panic!("Invalid child index: {}", child_index),
                }
            };
        }

        current_node
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Calculates the path and delegates to `insert_with_path`.
    pub fn insert(
        node: SharedInvertedIndexItem<T>,
        value: T,
        vector_id: u32,
    ) -> Result<(), String> {
        let mut node = node.write().unwrap();
        node.implicit = false;
        node.data.insert(
            IdentityMapKey::Int(vector_id),
            LazyItem::new(0.into(), value),
        );
        Ok(())
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Calculates the path and delegates to `get_value`.
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<T> {
        let path = calculate_path(dim_index, self.dim_index);
        self.get_value(&path, vector_id)
    }

    /// Retrieves a value from the index following the specified path.
    /// Recursively traverses child nodes or searches the data vector.
    fn get_value(&self, path: &[usize], vector_id: u32) -> Option<T> {
        match path.get(0) {
            Some(child_index) => self.children[*child_index]
                .as_ref()?
                .read()
                .ok()?
                .get_value(&path[1..], vector_id),
            None => self
                .data
                .get(&IdentityMapKey::Int(vector_id))
                .map(|lazy_item| lazy_item.get_data().map(|data| data.shared_get().clone()))
                .flatten(),
        }
    }
}

impl<T> InvertedIndexItem<T>
where
    T: Debug + Clone,
{
    /// Prints the tree structure of the index starting from the current node.
    /// Recursively prints child nodes with increased indentation.
    pub fn print_tree(&self, depth: usize, prev_dim_index: u32) {
        let indent = "  ".repeat(depth);
        let dim_index = prev_dim_index + self.dim_index;
        println!(
            "{}Dimension-Index {}: {}",
            indent,
            dim_index,
            if self.implicit {
                "Implicit"
            } else {
                "Explicit"
            }
        );
        // println!("{}Data: {:?}", indent, self.data);
        for (i, child) in self.children.iter().enumerate() {
            if let Some(item) = child {
                println!("{}-> 4^{} to:", indent, i);
                item.read().unwrap().print_tree(depth + 1, dim_index);
            }
        }
    }
}

#[derive(Clone)]
pub struct InvertedIndex<T>
where
    T: Clone + 'static,
{
    pub root: SharedInvertedIndexItem<T>,
}

impl<T> InvertedIndex<T>
where
    T: Copy,
{
    /// Creates a new `InvertedIndex` with an initial root node.
    pub fn new() -> Self {
        InvertedIndex {
            root: Arc::new(RwLock::new(InvertedIndexItem::new(0, false))),
        }
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Delegates to the root node's `get` method.
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<T> {
        self.root.read().unwrap().get(dim_index, vector_id)
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Delegates to the root node's `insert` method.
    pub fn insert(&self, dim_index: u32, value: T, vector_id: u32) -> Result<(), String> {
        let path = calculate_path(dim_index, self.root.read().unwrap().dim_index);
        println!("Path: {:?}", path);
        let node = InvertedIndexItem::find_or_create_node(self.root.clone(), &path);
        InvertedIndexItem::insert(node, value, vector_id)
    }
}

impl<T> InvertedIndex<T>
where
    T: Clone + 'static + Debug,
{
    /// Prints the tree structure of the entire index.
    /// Delegates to the root node's `print_tree` method.
    pub fn print_tree(&self) {
        self.root.read().unwrap().print_tree(0, 0);
    }
}

impl InvertedIndex<u8> {
    /// Adds a sparse vector to the index.
    /// Iterates over the vector and inserts non-zero values.
    pub fn add_sparse_vector(&self, vector: Vec<u8>, vector_id: u32) -> Result<(), String> {
        for (dim_index, &value) in vector.iter().enumerate() {
            if value != 0 {
                println!("Inserting value {} at dimension index {}", value, dim_index);
                self.insert(dim_index as u32, value, vector_id)?;
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
    pub fn print_index() {
        let inverted_index: InvertedIndex<u8> = InvertedIndex::new();

        let sparse_vector1 = vec![0, 15, 0, 0, 27, 0, 31, 0, 0, 42];
        let sparse_vector2 = vec![1, 0, 23, 0, 0, 38, 0, 45, 0, 56];

        inverted_index.add_sparse_vector(sparse_vector1, 0).unwrap();
        inverted_index.add_sparse_vector(sparse_vector2, 1).unwrap();

        println!("\nFinal Inverted Index structure:");
        inverted_index.print_tree();

        // Look up using dimension index and vector ID
        assert_eq!(inverted_index.get(1, 0), Some(15));
        assert_eq!(inverted_index.get(4, 0), Some(27));
        assert_eq!(inverted_index.get(6, 0), Some(31));
        assert_eq!(inverted_index.get(9, 0), Some(42));

        assert_eq!(inverted_index.get(0, 1), Some(1));
        assert_eq!(inverted_index.get(2, 1), Some(23));
        assert_eq!(inverted_index.get(5, 1), Some(38));
        assert_eq!(inverted_index.get(7, 1), Some(45));
        assert_eq!(inverted_index.get(9, 1), Some(56));
    }

    // Custom type for generating sparse vectors
    #[derive(Clone, Debug)]
    struct SparseVector(Vec<u8>);

    impl Arbitrary for SparseVector {
        /// Generates arbitrary sparse vectors for testing.
        fn arbitrary(g: &mut Gen) -> Self {
            let size = usize::arbitrary(g) % (20 - 10) + 10;
            let mut vec = vec![0; size];
            for _ in 0..size / 4 {
                // Make it sparse by only setting ~25% of elements
                let index = usize::arbitrary(g) % size;
                vec[index] = u8::arbitrary(g);
            }
            SparseVector(vec)
        }
    }

    /// Converts a vector to a HashMap of non-zero elements.
    fn vector_to_hashmap(vec: &[u8]) -> HashMap<usize, u8> {
        vec.iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
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
            // Run the test 100 times
            let root: SharedInvertedIndexItem<u8> =
                Arc::new(RwLock::new(InvertedIndexItem::new(0, false)));
            let root_clone1 = root.clone();
            let root_clone2 = root.clone();

            // Creates child 2 in depth 2 and child 1 in depth 1.
            let handle2 = thread::spawn(move || {
                let path = vec![0, 1, 2];
                InvertedIndexItem::find_or_create_node(root_clone1, &path);
            });

            // Creates child 1 in depth 1
            // Can overwrite child 1 created by handle 2
            let handle1 = thread::spawn(move || {
                let path = vec![0, 1];
                InvertedIndexItem::find_or_create_node(root, &path);
            });

            handle1.join().unwrap();
            handle2.join().unwrap();

            // Check the final state
            let node = root_clone2.read().unwrap();
            assert!(node.children[0].is_some(), "Child 0 should exist");

            let node = node.children[0].as_ref().unwrap().read().unwrap();
            assert!(node.children[1].is_some(), "Child 1 should exist");

            let node = node.children[1].as_ref().unwrap().read().unwrap();
            assert!(node.children[2].is_some(), "Child 2 should exist");
        }
    }
}
