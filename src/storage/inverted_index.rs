use std::fmt::Debug;
use std::sync::{Arc, RwLock};

const POWERS_OF_4: [u32; 12] = [
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
];

/// Returns the largest power of 4 that is less than or equal to `n`.
/// Iteratively multiplies by 4 until the result exceeds `n`.
pub fn largest_power_of_4_below(n: u32) -> u32 {
    let mut power = 1;
    while power <= n / 4 {
        power *= 4;
    }
    power
}

/// Returns the index of the power of 4 that matches `x`.
/// Searches the `POWERS_OF_4` array for the value `x`.
pub fn power_of_4_with_index(x: u32) -> Option<usize> {
    if x == 0 {
        return None;
    }
    POWERS_OF_4.iter().position(|&power| power == x)
}

/// Calculates the path from `current_dim_index` to `target_dim_index`.
/// Decomposes the difference into powers of 4 and returns the indices.
fn calculate_path(target_dim_index: u32, current_dim_index: u32) -> Vec<usize> {
    let mut path = Vec::new();
    let mut remaining = target_dim_index - current_dim_index;

    while remaining > 0 {
        let largest_power = largest_power_of_4_below(remaining);
        let child_index = power_of_4_with_index(largest_power).unwrap();
        path.push(child_index);
        remaining -= largest_power;
    }

    path
}

#[derive(Debug)]
pub struct InvertedIndexItem<T> {
    pub dim_index: u32,
    pub implicit: bool,
    pub data: Vec<(T, usize)>,
    pub children: [Option<Arc<RwLock<InvertedIndexItem<T>>>>; 8],
}

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
            data: Vec::new(),
            children: Default::default(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree and returns a reference to the node.
    fn find_or_create_node(
        node: Arc<RwLock<InvertedIndexItem<T>>>,
        path: &[usize],
    ) -> Arc<RwLock<InvertedIndexItem<T>>> {
        if path.is_empty() {
            node
        } else {
            let child_index = path[0];
            let mut node_write = node.write().unwrap();
            if node_write.children[child_index].is_none() {
                let new_dim_index = node_write.dim_index + POWERS_OF_4[child_index];
                let new_item = Arc::new(RwLock::new(InvertedIndexItem::new(new_dim_index, true)));
                node_write.children[child_index] = Some(new_item.clone());
                new_item
            } else {
                node_write.children[child_index].as_ref().unwrap().clone()
            }
        }
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Calculates the path and delegates to `insert_with_path`.
    pub fn insert(
        root: Arc<RwLock<InvertedIndexItem<T>>>,
        target_dim_index: u32,
        value: T,
        vector_id: usize,
    ) -> Result<(), String> {
        let path = calculate_path(target_dim_index, root.read().unwrap().dim_index);
        let node = InvertedIndexItem::find_or_create_node(root, &path);
        let mut node_write = node.write().unwrap();
        if node_write.implicit {
            node_write.implicit = false;
        }
        node_write.data.push((value, vector_id));
        Ok(())
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Calculates the path and delegates to `get_value`.
    pub fn get(&self, dim_index: u32, vector_id: usize) -> Option<T> {
        let path = calculate_path(dim_index, self.dim_index);
        self.get_value(&path, vector_id)
    }

    /// Retrieves a value from the index following the specified path.
    /// Recursively traverses child nodes or searches the data vector.
    fn get_value(&self, path: &[usize], vector_id: usize) -> Option<T> {
        match path.get(0) {
            Some(child_index) => self.children[*child_index]
                .as_ref()?
                .read()
                .ok()?
                .get_value(&path[1..], vector_id),
            None => self
                .data
                .iter()
                .find(|&&(_, id)| id == vector_id)
                .map(|(value, _)| *value),
        }
    }
}

impl<T: Debug> InvertedIndexItem<T> {
    /// Prints the tree structure of the index starting from the current node.
    /// Recursively prints child nodes with increased indentation.
    pub fn print_tree(&self, depth: usize) {
        let indent = "  ".repeat(depth);
        println!(
            "{}Dimension-Index {}: {}",
            indent,
            self.dim_index,
            if self.implicit {
                "Implicit"
            } else {
                "Explicit"
            }
        );
        println!("{}Data: {:?}", indent, self.data);
        for (i, child) in self.children.iter().enumerate() {
            if let Some(item) = child {
                println!("{}-> 4^{} to:", indent, i);
                item.read().unwrap().print_tree(depth + 1);
            }
        }
    }
}

pub struct InvertedIndex<T> {
    pub root: Arc<RwLock<InvertedIndexItem<T>>>,
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
    pub fn get(&self, dim_index: u32, vector_id: usize) -> Option<T> {
        self.root.read().unwrap().get(dim_index, vector_id)
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Delegates to the root node's `insert` method.
    pub fn insert(&self, dim_index: u32, value: T, vector_id: usize) -> Result<(), String> {
        InvertedIndexItem::insert(self.root.clone(), dim_index, value, vector_id)
    }
}

impl<T: Debug> InvertedIndex<T> {
    /// Prints the tree structure of the entire index.
    /// Delegates to the root node's `print_tree` method.
    pub fn print_tree(&self) {
        self.root.read().unwrap().print_tree(0);
    }
}

impl InvertedIndex<u8> {
    /// Adds a sparse vector to the index.
    /// Iterates over the vector and inserts non-zero values.
    pub fn add_sparse_vector(&self, vector: Vec<u8>, vector_id: usize) -> Result<(), String> {
        for (dim_index, &value) in vector.iter().enumerate() {
            if value != 0 {
                self.insert(dim_index as u32, value, vector_id)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::InvertedIndex;

    use quickcheck::{Arbitrary, Gen};
    use quickcheck_macros::quickcheck;
    use rand::seq::IteratorRandom;
    use rand::Rng;
    use std::collections::HashMap;

    #[test]
    /// Tests the basic functionality of the `InvertedIndex`.
    pub fn test_inverted_index() {
        let inverted_index: InvertedIndex<u8> = InvertedIndex::new();

        let sparse_vector1 = vec![0, 1, 0, 0, 2, 0, 3, 0, 0, 4];
        let sparse_vector2 = vec![1, 0, 2, 0, 0, 3, 0, 4, 0, 5];

        assert_eq!(inverted_index.add_sparse_vector(sparse_vector1, 0), Ok(()));
        assert_eq!(inverted_index.add_sparse_vector(sparse_vector2, 1), Ok(()));
    }

    #[test]
    /// Prints the structure of the `InvertedIndex` after adding sparse vectors.
    pub fn print_index() {
        let inverted_index: InvertedIndex<u8> = InvertedIndex::new();

        let sparse_vector1 = vec![0, 15, 0, 0, 27, 0, 31, 0, 0, 42];
        let sparse_vector2 = vec![1, 0, 23, 0, 0, 38, 0, 45, 0, 56];

        match inverted_index.add_sparse_vector(sparse_vector1, 0) {
            Ok(_) => println!("Successfully added sparse vector 1"),
            Err(e) => println!("Error adding sparse vector 1: {}", e),
        }

        match inverted_index.add_sparse_vector(sparse_vector2, 1) {
            Ok(_) => println!("Successfully added sparse vector 2"),
            Err(e) => println!("Error adding sparse vector 2: {}", e),
        }

        println!("\nFinal Inverted Index structure:");
        inverted_index.print_tree();
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

    /// Property-based test: Verifies insertion correctness and vector ID preservation.
    #[quickcheck]
    fn prop_insertion_and_id_correctness(vectors: Vec<SparseVector>) -> bool {
        let index = InvertedIndex::new();
        let mut expected = Vec::new();

        for (id, SparseVector(vec)) in vectors.iter().enumerate() {
            index.add_sparse_vector(vec.clone(), id).unwrap();
            expected.push(vector_to_hashmap(vec));
        }

        // Verify all inserted elements are present and associated with correct vector IDs
        expected.iter().enumerate().all(|(id, map)| {
            map.iter()
                .all(|(&dim, &value)| index.get(dim as u32, id).map_or(false, |v| v == value))
        })
    }

    /// Property-based test: Verifies handling of multiple vectors.
    #[quickcheck]
    fn prop_multiple_vector_handling(vectors: Vec<SparseVector>) -> bool {
        let index = InvertedIndex::new();

        for (id, SparseVector(vec)) in vectors.iter().enumerate() {
            index.add_sparse_vector(vec.clone(), id).unwrap();
        }

        // Verify each vector is stored separately
        vectors.iter().enumerate().all(|(id, SparseVector(vec))| {
            vector_to_hashmap(vec)
                .iter()
                .all(|(&dim, &value)| index.get(dim as u32, id).map_or(false, |v| v == value))
        })
    }
}
