use std::sync::{Arc, Mutex};

use super::helpers::generate_power_of_4_list;

#[derive(Debug)]
pub(crate) struct InvertedIndexItem {
    dim_index: u32,
    implicit: bool,
    data: Vec<(f32, u32)>,
    pointers: Vec<Option<Arc<Mutex<InvertedIndexItem>>>>,
}

impl InvertedIndexItem {
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        InvertedIndexItem {
            dim_index,
            implicit,
            data: vec![],
            pointers: vec![None; 8], // Space for exponents 0 to 7 (1 to 16384)
        }
    }

    pub fn insert_dim_index(
        &mut self,
        target_dim_index: u32,
        value: f32,
        vector_id: u32,
    ) -> Result<(), String> {
        let path = generate_power_of_4_list(target_dim_index - self.dim_index);
        self.insert_recursive(target_dim_index, &path, 0, value, vector_id)
    }

    fn insert_recursive(
        &mut self,
        target_dim_index: u32,
        path: &[(u32, u32)],
        path_index: usize,
        value: f32,
        vector_id: u32,
    ) -> Result<(), String> {
        if path_index == path.len() {
            // We've reached the target dimension index
            if !self.implicit {
                return Err(format!(
                    "Dimension-Index {} is already explicit",
                    self.dim_index
                ));
            }
            self.dim_index = target_dim_index;
            self.implicit = false;
            self.insert_data(value, vector_id);
            return Ok(());
        }

        let (_, exponent) = path[path_index];
        let next_dim_index = self.dim_index + 4u32.pow(exponent);

        if self.pointers[exponent as usize].is_none() {
            let new_item = Arc::new(Mutex::new(InvertedIndexItem::new(next_dim_index, true)));
            self.pointers[exponent as usize] = Some(new_item);
        }

        if let Some(next_item) = &self.pointers[exponent as usize] {
            next_item.lock().unwrap().insert_recursive(
                target_dim_index,
                path,
                path_index + 1,
                value,
                vector_id,
            )
        } else {
            Err("Failed to create or access the next item".to_string())
        }
    }

    fn insert_data(&mut self, value: f32, vector_id: u32) {
        let is_repeated = self.data.iter().find(|i| i.1 == vector_id);

        // TODO should this return error if the vector id is already registered ?
        //  or should it skip inserting and return nothing ?
        if is_repeated.is_none() {
            self.data.push((value, vector_id))
        }
    }

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
        for (i, pointer) in self.pointers.iter().enumerate() {
            if let Some(item) = pointer {
                println!("{}-> 4^{} to:", indent, i);
                item.lock().unwrap().print_tree(depth + 1);
            }
        }
    }
}
