use cuckoofilter::CuckooFilter;
use std::hash::DefaultHasher;

pub struct CuckooFilterTreeNode {
    pub filter: CuckooFilter<DefaultHasher>,
    pub left: Option<Box<CuckooFilterTreeNode>>,
    pub right: Option<Box<CuckooFilterTreeNode>>,
    pub index: usize,
    pub range_min: f32,
    pub range_max: f32,
}

impl Clone for CuckooFilterTreeNode {
    fn clone(&self) -> Self {
        let exported_filter = self.filter.export();
        let cloned_filter = CuckooFilter::from(exported_filter);

        CuckooFilterTreeNode {
            filter: cloned_filter,
            left: self.left.clone(),
            right: self.right.clone(),
            index: self.index,
            range_min: self.range_min,
            range_max: self.range_max,
        }
    }
}

impl CuckooFilterTreeNode {
    pub fn new(index: usize, range_min: f32, range_max: f32) -> Self {
        CuckooFilterTreeNode {
            filter: CuckooFilter::new(),
            left: None,
            right: None,
            index,
            range_min,
            range_max,
        }
    }

    pub fn build_tree(depth: usize, index: usize, range_min: f32, range_max: f32) -> Self {
        let mut node = CuckooFilterTreeNode::new(index, range_min, range_max);

        if depth > 0 {
            let mid = (range_min + range_max) / 2.0;
            node.left = Some(Box::new(CuckooFilterTreeNode::build_tree(
                depth - 1,
                index * 2,
                range_min,
                mid,
            )));
            node.right = Some(Box::new(CuckooFilterTreeNode::build_tree(
                depth - 1,
                index * 2 + 1,
                mid,
                range_max,
            )));
        }

        node
    }

    pub fn add_item(&mut self, id: u64, value: f32) {
        if self.left.is_none() && self.right.is_none() {
            self.filter.add(&id);
            return;
        }

        self.filter.add(&id);
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

    pub fn search(&self, id: u64) -> (bool, usize) {
        if !self.filter.contains(&id) {
            return (false, 0);
        }

        if self.left.is_none() && self.right.is_none() {
            let found = self.filter.contains(&id);
            return (
                found,
                if found {
                    self.index * 2 + 1
                } else {
                    self.index * 2
                },
            );
        }

        if let Some(ref left_child) = self.left {
            if left_child.filter.contains(&id) {
                return left_child.search(id);
            }
        }

        if let Some(ref right_child) = self.right {
            if right_child.filter.contains(&id) {
                return right_child.search(id);
            }
        }

        (false, 0)
    }
}
