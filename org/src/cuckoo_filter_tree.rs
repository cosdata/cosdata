//! This program implements a binary tree of cuckoo filters for efficient range-based set membership checks.
//!
//! # Overview
//! Each `TreeNode` in the binary tree contains a `CuckooFilter`, a probabilistic data structure for quick
//! set membership tests. Items are added to the tree based on a range, allowing queries to efficiently
//! narrow down a search path by traversing nodes.
//!
//! # Key Components
//! - **CuckooFilter**: A simplified filter using a vector for item storage, with methods `add` and `contains` 
//!   to manage and check set membership.
//! - **TreeNode**: A node in the binary tree that contains a `CuckooFilter` and child pointers (left, right).
//!   The `add_item` method places an item in the correct node based on value, and `search` uses only the `id` 
//!   to determine set membership, traversing left or right depending on node filters.
//! - **Tree Construction**: The `build_tree` function creates a binary tree by dividing the range recursively
//!   until reaching the specified depth, ensuring each leaf node covers a unique subrange.
//!
//! # Usage
//! - Build a tree with a specified depth and range (e.g., `range_min=0, range_max=1000`).
//! - Use `add_item` to insert items (as `(id, value)`), where the value determines where to add the `id`.
//! - Use `search` to check membership by `id`, returning a tuple of a boolean and a custom value based on
//!   leaf node index.
//!
//! This structure supports efficient hierarchical membership tests, ideal for applications needing fast 
//! range-based lookups or hierarchical data organization.

struct CuckooFilter {
    items: Vec<u64>,
}

impl CuckooFilter {
    fn new() -> Self {
        CuckooFilter { items: Vec::new() }
    }
    
    fn add(&mut self, id: u64) {
        self.items.push(id);
    }
    
    fn contains(&self, id: u64) -> bool {
        self.items.contains(&id)
    }
}

struct TreeNode {
    filter: CuckooFilter,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    index: usize,
    range_min: f32,
    range_max: f32,
}

impl TreeNode {
    fn new(index: usize, range_min: f32, range_max: f32) -> Self {
        TreeNode {
            filter: CuckooFilter::new(),
            left: None,
            right: None,
            index,
            range_min,
            range_max,
        }
    }
    
    fn add_item(&mut self, id: u64, value: f32) {
        if self.left.is_none() && self.right.is_none() {
            self.filter.add(id);
            return;
        }
        
        self.filter.add(id);
        let mid = (self.range_min + self.range_max) / 2.0;
        
        if value <= mid {
            if let Some(ref mut left_child) = self.left {
                left_child.add_item(id, value);
            }
        } else {
            if let Some(ref mut right_child) = self.right {
                right_child.add_item(id, value);
            }
        }
    }

    fn search(&self, id: u64) -> (bool, usize) {
        if !self.filter.contains(id) {
            return (false, 0);
        }
        
        if self.left.is_none() && self.right.is_none() {
            let found = self.filter.contains(id);
            return (found, if found { self.index * 2 + 1 } else { self.index * 2 });
        }
        
        if let Some(ref left_child) = self.left {
            if left_child.filter.contains(id) {
                return left_child.search(id);
            }
        }
        
        if let Some(ref right_child) = self.right {
            if right_child.filter.contains(id) {
                return right_child.search(id);
            }
        }
        
        (false, 0)
    }
}

fn build_tree(depth: usize, index: usize, range_min: f32, range_max: f32) -> TreeNode {
    let mut node = TreeNode::new(index, range_min, range_max);

    if depth > 0 {
        let mid = (range_min + range_max) / 2.0;
        node.left = Some(Box::new(build_tree(depth - 1, index * 2, range_min, mid)));
        node.right = Some(Box::new(build_tree(depth - 1, index * 2 + 1, mid, range_max)));
    }

    node
}

fn main() {
    let range_min = 0.0;
    let range_max = 10.0;
    let mut root = build_tree(5, 0, range_min, range_max);

    root.add_item(1, 1.25);
    root.add_item(2, 8.98);

    let search_item_1 = 1;
    let search_item_2 = 2;
    let search_item_3 = 3;

    println!("Result for id {}: {:?}", search_item_1, root.search(search_item_1));
    println!("Result for id {}: {:?}", search_item_2, root.search(search_item_2));
    println!("Result for id {}: {:?}", search_item_3, root.search(search_item_3));
}
