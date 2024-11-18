use probabilistic_collections::cuckoo::CuckooFilter;

struct CuckooFilterTreeNode {
    filter: CuckooFilter<u64>,
    left: Option<Box<CuckooFilterTreeNode>>,
    right: Option<Box<CuckooFilterTreeNode>>,
    index: usize,
    range_min: f32,
    range_max: f32,
}

impl CuckooFilterTreeNode {
    fn new(index: usize, range_min: f32, range_max: f32) -> Self {
        CuckooFilterTreeNode {
            filter: CuckooFilter::new(100),
            left: None,
            right: None,
            index,
            range_min,
            range_max,
        }
    }

    fn add_item(&mut self, id: u64, value: f32) {
        if self.left.is_none() && self.right.is_none() {
            self.filter.insert(&id);
            return;
        }

        self.filter.insert(&id);
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
