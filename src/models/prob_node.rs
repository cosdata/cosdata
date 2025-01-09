use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

use super::{
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
    pub versions: ProbLazyItemArray<ProbNode, 4>,
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
        versions: ProbLazyItemArray<ProbNode, 4>,
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

    pub fn add_neighbor(&self, neighbor_id: u32, neighbor_node: SharedNode, dist: MetricResult) {
        let mut neighbor_dist = dist.get_value();
        let neighbor = Box::new((neighbor_id, neighbor_node, dist));
        let mut neighbor_ptr = Box::into_raw(neighbor);
        let mut inserted = false;

        for neighbor in &self.neighbors {
            let result =
                neighbor.fetch_update(Ordering::Release, Ordering::Acquire, |current_neighbor| {
                    if let Some((_, _, current_neighbor_similarity)) =
                        unsafe { current_neighbor.as_ref() }
                    {
                        if neighbor_dist < current_neighbor_similarity.get_value() {
                            return None;
                        }
                    }
                    Some(neighbor_ptr)
                });

            if let Ok(prev_neighbor_ptr) = result {
                if let Some((_, _, prev_neighbor_dist)) = unsafe { prev_neighbor_ptr.as_ref() } {
                    neighbor_dist = prev_neighbor_dist.get_value();
                    neighbor_ptr = prev_neighbor_ptr;
                } else {
                    inserted = true;
                    break;
                }
            }
        }

        if !inserted {
            unsafe {
                drop(Box::from_raw(neighbor_ptr));
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
                    .map(|neighbor| neighbor.1.clone())
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
