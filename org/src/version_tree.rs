    
use std::sync::{Arc, Mutex};

fn largest_power_of_4_below(n: u32) -> u32 {
    let mut power = 1;
    while power <= n / 4 {
        power *= 4;
    }
    power
}

// Define a hard-coded list of known powers of 4
const POWERS_OF_4: [u32; 12] = [
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
];

pub fn power_of_4_with_index(x: u32) -> Option<usize> {
    // Check if the number is 0 (not a power of 4)
    if x == 0 {
        return None;
    }
    // Check if x is in the list of powers of 4 and return its index
    match POWERS_OF_4.iter().position(|&power| power == x) {
        Some(index) => Some(index),
        None => None,
    }
}

#[derive(Debug)]
struct VersionNode {
    version: u32,
    data: bool,
    pointers: Vec<Option<Arc<Mutex<VersionNode>>>>,
}

impl VersionNode {
    fn new(version: u32, data: bool) -> Self {
        VersionNode {
            version,
            data,
            pointers: vec![None; 10], // Initialize with None values of length 10
        }
    }

    fn insert_node(&mut self, target_version: Arc<Mutex<VersionNode>>) {
        let target_version_lock = target_version.lock().unwrap();
        let delta = largest_power_of_4_below(target_version_lock.version - self.version);


        if let Some(index) = power_of_4_with_index(delta) {
            println!(
                "self.ver {} | target.ver {} | delta {} | index {}",
                self.version, target_version_lock.version, delta, index
            );

            drop(target_version_lock); // Drop the lock here to avoid deadlock

            if let Some(p) = self.pointers[index].clone() {
                let mut p_lock = p.lock().unwrap();
                p_lock.insert_node(target_version.clone());
            } else {
                self.pointers[index] = Some(target_version.clone());
                println!("{}'s pointers {:?}", self.version, self.pointers);
            }
        } else {
            println!("error, cant happen {}", delta);
        }
    }

    fn print_graph(&self, depth: usize) {
        let indent = " ".repeat(depth * 4);
        println!("{}Version {}: {:?}", indent, self.version, self.data);

        for pointer in &self.pointers {
            if let Some(ref p) = pointer {
                println!(
                    "{}-> points to Version {}",
                    indent,
                    p.lock().unwrap().version
                );
                p.lock().unwrap().print_graph(depth + 1);
            }
        }
    }
}

#[derive(Debug)]
struct VersionControl {
    root: Option<Arc<Mutex<VersionNode>>>,
}

impl VersionControl {
    fn new() -> Self {
        VersionControl { root: None }
    }

    fn add_version(&mut self, version: u32, data: bool) {
        let new_node = Arc::new(Mutex::new(VersionNode::new(version, data)));

        if self.root.is_none() {
            self.root = Some(new_node);
            return; // No need to add pointers if this is the first node
        }

        let current = self.root.clone();
        println!("\n \n \n ====> Version {} <====", version);
        if let Some(node) = current {
            node.lock().unwrap().insert_node(new_node);
        } else {
            println!("error ");
        }
    }

    fn print_graph(&self) {
        if let Some(ref rt) = self.root {
            rt.lock().unwrap().print_graph(0);
        } else {
            println!("No versions available.");
        }
    }
}

fn main() {
    let mut vc = VersionControl::new();

    vc.add_version(0, true);
    vc.add_version(1, true);
    vc.add_version(2, true);
    vc.add_version(3, true);
    vc.add_version(4, true);
    vc.add_version(5, true);
    vc.add_version(6, true);
    vc.add_version(7, true);
    vc.add_version(8, true);
    vc.add_version(9, true);
    vc.add_version(10, true);
    vc.add_version(11, true);
    vc.add_version(12, true);
    vc.add_version(13, true);
    vc.add_version(16, true);
    vc.add_version(17, true);
    vc.add_version(64, true);
    vc.add_version(65, true);

    vc.add_version(128, true);
    vc.add_version(129, true);
    vc.add_version(191, true);
    vc.add_version(192, true);
    vc.add_version(193, true);
    vc.add_version(256, true);
    vc.add_version(512, true);
    vc.add_version(768, true);
    vc.add_version(1024, true);

    vc.print_graph();
}
