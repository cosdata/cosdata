use std::sync::{Arc, Mutex};

use super::inverted_index_item::InvertedIndexItem;

pub(crate) struct InvertedIndex {
    root: Arc<Mutex<InvertedIndexItem>>,
}

impl InvertedIndex {
    fn new() -> Self {
        InvertedIndex {
            root: Arc::new(Mutex::new(InvertedIndexItem::new(0, false))),
        }
    }

    fn add_dim_index(&self, dim_index: u32, value: f32, vector_id: u32) -> Result<(), String> {
        self.root
            .lock()
            .unwrap()
            .insert_dim_index(dim_index, value, vector_id)
    }

    fn print_tree(&self) {
        self.root.lock().unwrap().print_tree(0);
    }
}
