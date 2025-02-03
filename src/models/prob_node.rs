use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

use super::{
    cache_loader::ProbCache,
    prob_lazy_load::{lazy_item::ProbLazyItem, lazy_item_array::ProbLazyItemArray},
    types::{HNSWLevel, MetricResult, NodeProp, VectorId},
};

pub type SharedNode = *mut ProbLazyItem<ProbNode>;

pub struct ProbNode {
    pub hnsw_level: HNSWLevel,
    pub prop: Arc<NodeProp>,
    // (neighbor_id, neighbor_node, distance)
    // even though `VectorId` is an u64 we don't need the full precision here.
    neighbors: Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]>,
    parent: AtomicPtr<ProbLazyItem<ProbNode>>,
    child: AtomicPtr<ProbLazyItem<ProbNode>>,
    pub versions: ProbLazyItemArray<ProbNode, 8>,
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
        }
    }

    pub fn new_with_neighbors(
        hnsw_level: HNSWLevel,
        prop: Arc<NodeProp>,
        neighbors: Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]>,
        parent: SharedNode,
        child: SharedNode,
    ) -> Self {
        Self {
            hnsw_level,
            prop,
            neighbors,
            parent: AtomicPtr::new(parent),
            child: AtomicPtr::new(child),
            versions: ProbLazyItemArray::new(),
        }
    }

    pub fn new_with_neighbors_and_versions(
        hnsw_level: HNSWLevel,
        prop: Arc<NodeProp>,
        neighbors: Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]>,
        parent: SharedNode,
        child: SharedNode,
        versions: ProbLazyItemArray<ProbNode, 8>,
    ) -> Self {
        Self {
            hnsw_level,
            prop,
            neighbors,
            parent: AtomicPtr::new(parent),
            child: AtomicPtr::new(child),
            versions,
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
        cache: &ProbCache,
    ) -> bool {
        let mut neighbor_dist = dist.get_value();
        let neighbor = Box::new((neighbor_id, neighbor_node.clone(), dist));
        let mut neighbor_ptr = Box::into_raw(neighbor);
        // true if the current neighbor stored in `neighbor_ptr` have a link with current node, so we know if we need to free it or not
        let mut inserted = false;
        // true if the initial neighbor was inserted, to return
        let mut initial_neighbor_inserted = false;
        let mut node_to_remove_backlink = None;

        for neighbor in &self.neighbors {
            let result =
                neighbor.fetch_update(Ordering::Release, Ordering::Acquire, |current_neighbor| {
                    if let Some((_curr_id, curr_node, curr_dist)) =
                        unsafe { current_neighbor.as_ref() }
                    {
                        // If new neighbor is farther, continue searching
                        if neighbor_dist < curr_dist.get_value() {
                            return None;
                        }
                        // If new neighbor is closer, mark the previous neighbor for backlink removal
                        node_to_remove_backlink = Some(curr_node.clone());
                    }
                    // Either slot is empty or new neighbor is closer - attempt insertion
                    Some(neighbor_ptr)
                });

            if let Ok(prev_neighbor_ptr) = result {
                initial_neighbor_inserted = true;
                if let Some((_, _, prev_dist)) = unsafe { prev_neighbor_ptr.as_ref() } {
                    neighbor_dist = prev_dist.get_value();
                    neighbor_ptr = prev_neighbor_ptr;
                    inserted = false;
                } else {
                    inserted = true;
                    break;
                }
            }
        }

        if !inserted {
            if let Some(node) = node_to_remove_backlink {
                if let Ok(n) = unsafe { &*node }.try_get_data(cache) {
                    n.remove_neighbor(self.get_id().0 as u32);
                }
            }
            unsafe {
                drop(Box::from_raw(neighbor_ptr));
            }
        }

        initial_neighbor_inserted
    }

    pub fn all_neighbor_slots_filled(&self) -> bool {
        self.neighbors
            .iter()
            .all(|ptr| !ptr.load(Ordering::Relaxed).is_null())
    }

    pub fn remove_neighbor(&self, id: u32) {
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

    /// See [`crate::models::serializer::prob::node`] for how its calculated
    pub fn get_serialized_size(neighbors_len: usize) -> usize {
        neighbors_len * 19 + 111
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
