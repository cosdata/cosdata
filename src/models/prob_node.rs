use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

use arcshift::ArcShift;

use super::{
    prob_lazy_load::{lazy_item::ProbLazyItem, lazy_item_array::ProbLazyItemArray},
    types::{HNSWLevel, MetricResult, PropState, VectorId},
};

pub const NEIGHBORS_COUNT: usize = 16;

pub type SharedNode = Arc<ProbLazyItem<ProbNode>>;

pub struct ProbNode {
    pub hnsw_level: HNSWLevel,
    pub prop: ArcShift<PropState>,
    neighbors: [AtomicPtr<Arc<(SharedNode, MetricResult)>>; NEIGHBORS_COUNT],
    parent: AtomicPtr<SharedNode>,
    child: AtomicPtr<SharedNode>,
    pub versions: ProbLazyItemArray<ProbNode, 4>,
}

unsafe impl Send for ProbNode {}
unsafe impl Sync for ProbNode {}

impl ProbNode {
    pub fn new(
        hnsw_level: HNSWLevel,
        prop: ArcShift<PropState>,
        parent: Option<SharedNode>,
        child: Option<SharedNode>,
    ) -> Self {
        let neighbors = std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut()));

        Self {
            hnsw_level,
            prop,
            neighbors,
            parent: AtomicPtr::new(
                parent.map_or_else(|| ptr::null_mut(), |parent| Box::into_raw(Box::new(parent))),
            ),
            child: AtomicPtr::new(
                child.map_or_else(|| ptr::null_mut(), |child| Box::into_raw(Box::new(child))),
            ),
            versions: ProbLazyItemArray::new(),
        }
    }

    pub fn new_with_neighbors(
        hnsw_level: HNSWLevel,
        prop: ArcShift<PropState>,
        neighbors: [AtomicPtr<Arc<(SharedNode, MetricResult)>>; NEIGHBORS_COUNT],
        parent: Option<SharedNode>,
        child: Option<SharedNode>,
    ) -> Self {
        Self {
            hnsw_level,
            prop,
            neighbors,
            parent: AtomicPtr::new(
                parent.map_or_else(|| ptr::null_mut(), |parent| Box::into_raw(Box::new(parent))),
            ),
            child: AtomicPtr::new(
                child.map_or_else(|| ptr::null_mut(), |child| Box::into_raw(Box::new(child))),
            ),
            versions: ProbLazyItemArray::new(),
        }
    }

    pub fn new_with_neighbors_and_versions(
        hnsw_level: HNSWLevel,
        prop: ArcShift<PropState>,
        neighbors: [AtomicPtr<Arc<(SharedNode, MetricResult)>>; NEIGHBORS_COUNT],
        parent: Option<SharedNode>,
        child: Option<SharedNode>,
        versions: ProbLazyItemArray<ProbNode, 4>,
    ) -> Self {
        Self {
            hnsw_level,
            prop,
            neighbors,
            parent: AtomicPtr::new(
                parent.map_or_else(|| ptr::null_mut(), |parent| Box::into_raw(Box::new(parent))),
            ),
            child: AtomicPtr::new(
                child.map_or_else(|| ptr::null_mut(), |child| Box::into_raw(Box::new(child))),
            ),
            versions,
        }
    }

    pub fn get_parent(&self) -> Option<SharedNode> {
        unsafe {
            let ptr = self.parent.load(Ordering::SeqCst);
            if ptr.is_null() {
                None
            } else {
                Some((*ptr).clone())
            }
        }
    }

    pub fn set_parent(&self, parent: SharedNode) {
        unsafe {
            let old = self
                .parent
                .swap(Box::into_raw(Box::new(parent)), Ordering::SeqCst);
            if !old.is_null() {
                drop(Box::from_raw(old));
            }
        }
    }

    pub fn get_child(&self) -> Option<SharedNode> {
        unsafe {
            let ptr = self.child.load(Ordering::SeqCst);
            if ptr.is_null() {
                None
            } else {
                Some((*ptr).clone())
            }
        }
    }

    pub fn set_child(&self, child: SharedNode) {
        unsafe {
            let old = self
                .child
                .swap(Box::into_raw(Box::new(child)), Ordering::SeqCst);
            if !old.is_null() {
                drop(Box::from_raw(old));
            }
        }
    }

    pub fn get_id(&self) -> Option<VectorId> {
        let mut prop_arc = self.prop.clone();
        match prop_arc.get() {
            PropState::Pending(_) => None,
            PropState::Ready(prop) => Some(prop.id.clone()),
        }
    }

    pub fn add_neighbor(
        &self,
        neighbor_node: SharedNode,
        neighbor_id: VectorId,
        dist: MetricResult,
    ) {
        let idx = ((self.get_id().unwrap().get_hash() ^ neighbor_id.get_hash())
            % NEIGHBORS_COUNT as u64) as usize;
        let neighbor = Box::new(Arc::new((neighbor_node, dist)));

        let neighbor_ptr = Box::into_raw(neighbor);

        let result = self.neighbors[idx].fetch_update(
            Ordering::SeqCst,
            Ordering::SeqCst,
            |current_neighbor| {
                if current_neighbor.is_null() {
                    Some(neighbor_ptr)
                } else {
                    unsafe {
                        if dist.get_value() > (*current_neighbor).1.get_value() {
                            Some(neighbor_ptr)
                        } else {
                            None
                        }
                    }
                }
            },
        );

        unsafe {
            if let Ok(prev_neighbor) = result {
                if !prev_neighbor.is_null() {
                    drop(Box::from_raw(prev_neighbor));
                }
            } else {
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
                    .map(|neighbor| neighbor.0.clone())
            })
            .collect()
    }

    pub fn clone_neighbors(&self) -> [AtomicPtr<Arc<(SharedNode, MetricResult)>>; NEIGHBORS_COUNT] {
        std::array::from_fn(|i| unsafe {
            AtomicPtr::new(
                self.neighbors[i]
                    .load(Ordering::SeqCst)
                    .as_ref()
                    .map_or_else(
                        || ptr::null_mut(),
                        |neighbor| Box::into_raw(Box::new(neighbor.clone())),
                    ),
            )
        })
    }

    pub fn get_neighbors_raw(
        &self,
    ) -> &[AtomicPtr<Arc<(SharedNode, MetricResult)>>; NEIGHBORS_COUNT] {
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