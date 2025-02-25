use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, Mutex,
    },
};

use super::{
    cache_loader::DenseIndexCache,
    prob_lazy_load::{lazy_item::ProbLazyItem, lazy_item_array::ProbLazyItemArray},
    types::{HNSWLevel, MetricResult, NodeProp, VectorId},
};

pub type SharedNode = *mut ProbLazyItem<ProbNode>;

pub struct ProbNode {
    pub hnsw_level: HNSWLevel,
    pub prop: Arc<NodeProp>,
    // (neighbor_id, neighbor_node, distance)
    // even though `VectorId` is an u64 we don't need the full range here.
    neighbors: Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]>,
    parent: AtomicPtr<ProbLazyItem<ProbNode>>,
    child: AtomicPtr<ProbLazyItem<ProbNode>>,
    pub versions: ProbLazyItemArray<ProbNode, 8>,
    lowest_index: Mutex<(u8, f32)>,
    pub root_version: SharedNode,
}

unsafe impl Send for ProbNode {}
unsafe impl Sync for ProbNode {}

impl ProbNode {
    pub fn new(
        hnsw_level: HNSWLevel,
        prop: Arc<NodeProp>,
        parent: SharedNode,
        child: SharedNode,
        neighbors_count: usize,
    ) -> Self {
        let mut neighbors = Vec::with_capacity(neighbors_count);

        for _ in 0..neighbors_count {
            neighbors.push(AtomicPtr::new(ptr::null_mut()));
        }

        Self {
            hnsw_level,
            prop,
            neighbors: neighbors.into_boxed_slice(),
            parent: AtomicPtr::new(parent),
            child: AtomicPtr::new(child),
            versions: ProbLazyItemArray::new(),
            lowest_index: Mutex::new((0, -1.0)),
            root_version: ptr::null_mut(),
        }
    }

    pub fn new_with_neighbors_and_versions_and_root_version(
        hnsw_level: HNSWLevel,
        prop: Arc<NodeProp>,
        neighbors: Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]>,
        parent: SharedNode,
        child: SharedNode,
        versions: ProbLazyItemArray<ProbNode, 8>,
        root_version: SharedNode,
    ) -> Self {
        let mut lowest_idx = 0;
        let mut lowest_sim = 2.0;

        for (idx, neighbor) in neighbors.iter().enumerate() {
            let neighbor = unsafe { neighbor.load(Ordering::Relaxed).as_ref() };
            let Some((_, _, sim)) = neighbor else {
                lowest_idx = idx;
                lowest_sim = -1.0;
                break;
            };
            let sim = sim.get_value();
            if sim < lowest_sim {
                lowest_idx = idx;
                lowest_sim = sim;
            }
        }

        Self {
            hnsw_level,
            prop,
            neighbors,
            parent: AtomicPtr::new(parent),
            child: AtomicPtr::new(child),
            versions,
            lowest_index: Mutex::new((lowest_idx as u8, lowest_sim)),
            root_version,
        }
    }

    pub fn get_parent(&self) -> SharedNode {
        self.parent.load(Ordering::Acquire)
    }

    pub fn set_parent(&self, parent: SharedNode) {
        self.parent.store(parent, Ordering::Release);
    }

    pub fn get_child(&self) -> SharedNode {
        self.child.load(Ordering::Acquire)
    }

    pub fn set_child(&self, child: SharedNode) {
        self.child.store(child, Ordering::Release);
    }

    pub fn get_id(&self) -> &VectorId {
        &self.prop.id
    }

    pub fn add_neighbor(
        &self,
        neighbor_id: u32,
        neighbor_node: SharedNode,
        dist: MetricResult,
        cache: &DenseIndexCache,
    ) -> Option<u8> {
        // First find an empty slot or the slot with lowest similarity
        let mut lowest_idx_guard = self.lowest_index.lock().unwrap();
        let (lowest_idx, lowest_sim) = *lowest_idx_guard;

        // If we didn't find an empty slot and new neighbor isn't better, return
        if dist.get_value() <= lowest_sim {
            return None;
        }

        // Create new neighbor and attempt atomic update
        let new_neighbor = Box::new((neighbor_id, neighbor_node.clone(), dist));
        let new_ptr = Box::into_raw(new_neighbor);

        let result = self.neighbors[lowest_idx as usize].fetch_update(
            Ordering::Release,
            Ordering::Acquire,
            |current| match unsafe { current.as_ref() } {
                None => Some(new_ptr),
                Some((_, _, curr_dist)) if dist.get_value() > curr_dist.get_value() => {
                    Some(new_ptr)
                }
                _ => None,
            },
        );

        let ret = match result {
            Ok(old_ptr) => {
                // Successful update
                unsafe {
                    if let Some((_, node, _)) = old_ptr.as_ref() {
                        (**node)
                            .try_get_data(cache)
                            .unwrap()
                            .remove_neighbor_by_id(self.get_id().0 as u32);
                        drop(Box::from_raw(old_ptr));
                    }
                }
                Some(lowest_idx)
            }
            Err(_) => {
                // Update failed, clean up new neighbor
                unsafe {
                    drop(Box::from_raw(new_ptr));
                }
                None
            }
        };

        let mut new_lowest_idx = 0;
        let mut new_lowest_sim = 2.0;

        for (idx, nbr) in self.neighbors.iter().enumerate() {
            let nbr = unsafe { nbr.load(Ordering::Relaxed).as_ref() };
            let Some((_, _, nbr_sim)) = nbr else {
                new_lowest_sim = -1.0;
                new_lowest_idx = idx;
                break;
            };
            let nbr_sim = nbr_sim.get_value();
            if nbr_sim < new_lowest_sim {
                new_lowest_sim = nbr_sim;
                new_lowest_idx = idx;
            }
        }

        *lowest_idx_guard = (new_lowest_idx as u8, new_lowest_sim);

        ret
    }

    pub fn remove_neighbor_by_id(&self, id: u32) {
        for neighbor in &self.neighbors {
            let res = neighbor.fetch_update(Ordering::Release, Ordering::Acquire, |nbr| {
                if let Some((nbr_id, _, _)) = unsafe { nbr.as_ref() } {
                    if nbr_id == &id {
                        return Some(ptr::null_mut());
                    }
                }
                None
            });

            if let Ok(nbr) = res {
                // cant be null, did the null check in fetch_update
                unsafe {
                    drop(Box::from_raw(nbr));
                }
                break;
            }
        }
    }

    pub fn remove_neighbor(&self, index: u8, id: u32) {
        let _ = self.neighbors[index as usize].fetch_update(
            Ordering::Release,
            Ordering::Acquire,
            |nbr| {
                if let Some((nbr_id, _, _)) = unsafe { nbr.as_ref() } {
                    if nbr_id == &id {
                        return Some(ptr::null_mut());
                    }
                }
                None
            },
        );
    }

    pub fn get_neighbors(&self) -> Vec<SharedNode> {
        self.neighbors
            .iter()
            .flat_map(|neighbor| unsafe {
                neighbor
                    .load(Ordering::Relaxed)
                    .as_ref()
                    .map(|neighbor| neighbor.1)
            })
            .collect()
    }

    pub fn clone_neighbors(&self) -> Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> {
        self.neighbors
            .iter()
            .map(|neighbor| unsafe {
                AtomicPtr::new(neighbor.load(Ordering::SeqCst).as_ref().map_or_else(
                    || ptr::null_mut(),
                    |neighbor| Box::into_raw(Box::new(neighbor.clone())),
                ))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn get_neighbors_raw(&self) -> &Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> {
        &self.neighbors
    }

    /// See [`crate::models::serializer::dense::node`] for how its calculated
    pub fn get_serialized_size(neighbors_len: usize) -> usize {
        neighbors_len * 19 + 121
    }
}

impl Drop for ProbNode {
    fn drop(&mut self) {
        for neighbor in &self.neighbors {
            let ptr = neighbor.load(Ordering::SeqCst);

            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}
