use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

use parking_lot::{RwLock, RwLockReadGuard};

use super::{
    cache_loader::HNSWIndexCache,
    prob_lazy_load::lazy_item::ProbLazyItem,
    types::{DistanceMetric, HNSWLevel, InternalId, MetricResult, NodePropMetadata, NodePropValue},
};

pub type SharedNode = *mut ProbLazyItem<ProbNode>;
pub type Neighbors = Box<[AtomicPtr<(InternalId, SharedNode, MetricResult)>]>;

pub struct ProbNode {
    pub hnsw_level: HNSWLevel,
    pub prop_value: Arc<NodePropValue>,
    pub prop_metadata: Option<Arc<NodePropMetadata>>,
    // Each neighbor is represented as (neighbor_id, neighbor_node, distance)
    neighbors: Neighbors,
    parent: AtomicPtr<ProbLazyItem<ProbNode>>,
    child: AtomicPtr<ProbLazyItem<ProbNode>>,
    lowest_index: RwLock<(u8, MetricResult)>,
    // Tracks versioning of this node:
    // - If this node is the *base* (original) version, `root_version` points to the latest version.
    // - If this node is a *non-base* (derived) version, `root_version` points back to the base version.
    // The accompanying `bool` flag indicates what `root_version` points to:
    // - `true`: points to the root (base) version.
    // - `false`: points to the latest version (this node is the base version).
    // If the pointer is null, there is only one version (this node itself), and the flag can be ignored.
    pub root_version: RwLock<(SharedNode, bool)>,
}

unsafe impl Send for ProbNode {}
unsafe impl Sync for ProbNode {}

impl ProbNode {
    pub fn new(
        hnsw_level: HNSWLevel,
        prop_value: Arc<NodePropValue>,
        prop_metadata: Option<Arc<NodePropMetadata>>,
        parent: SharedNode,
        child: SharedNode,
        neighbors_count: usize,
        dist_metric: DistanceMetric,
    ) -> Self {
        let mut neighbors = Vec::with_capacity(neighbors_count);

        for _ in 0..neighbors_count {
            neighbors.push(AtomicPtr::new(ptr::null_mut()));
        }

        Self {
            hnsw_level,
            prop_value,
            prop_metadata,
            neighbors: neighbors.into_boxed_slice(),
            parent: AtomicPtr::new(parent),
            child: AtomicPtr::new(child),
            lowest_index: RwLock::new((0, MetricResult::min(dist_metric))),
            root_version: RwLock::new((ptr::null_mut(), false)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_neighbors_and_versions_and_root_version(
        hnsw_level: HNSWLevel,
        prop_value: Arc<NodePropValue>,
        prop_metadata: Option<Arc<NodePropMetadata>>,
        neighbors: Neighbors,
        parent: SharedNode,
        child: SharedNode,
        root_version: SharedNode,
        root_version_tag: bool,
        dist_metric: DistanceMetric,
    ) -> Self {
        let mut lowest_idx = 0;
        let mut lowest_sim = MetricResult::max(dist_metric);

        for (idx, neighbor) in neighbors.iter().enumerate() {
            let neighbor = unsafe { neighbor.load(Ordering::Relaxed).as_ref() };
            let Some((_, _, sim)) = neighbor else {
                lowest_idx = idx;
                lowest_sim = MetricResult::min(dist_metric);
                break;
            };
            if sim < &lowest_sim {
                lowest_idx = idx;
                lowest_sim = *sim;
            }
        }

        Self {
            hnsw_level,
            prop_value,
            prop_metadata,
            neighbors,
            parent: AtomicPtr::new(parent),
            child: AtomicPtr::new(child),
            lowest_index: RwLock::new((lowest_idx as u8, lowest_sim)),
            root_version: RwLock::new((root_version, root_version_tag)),
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

    pub fn get_id(&self) -> InternalId {
        self.prop_value.id
    }

    pub fn freeze(&self) -> RwLockReadGuard<'_, (u8, MetricResult)> {
        self.lowest_index.read()
    }

    pub fn add_neighbor(
        &self,
        neighbor_id: InternalId,
        neighbor_node: SharedNode,
        dist: MetricResult,
        cache: &HNSWIndexCache,
        dist_metric: DistanceMetric,
    ) -> Option<u8> {
        // First find an empty slot or the slot with lowest similarity
        let mut lowest_idx_guard = self.lowest_index.write();
        let (lowest_idx, lowest_sim) = *lowest_idx_guard;

        // If we didn't find an empty slot and new neighbor isn't better, return
        if dist <= lowest_sim {
            return None;
        }

        // Create new neighbor and attempt atomic update
        let new_neighbor = Box::new((neighbor_id, neighbor_node, dist));
        let new_ptr = Box::into_raw(new_neighbor);

        let result = self.neighbors[lowest_idx as usize].fetch_update(
            Ordering::Release,
            Ordering::Acquire,
            |current| match unsafe { current.as_ref() } {
                None => Some(new_ptr),
                Some((_, _, curr_dist)) if &dist > curr_dist => Some(new_ptr),
                _ => None,
            },
        );

        let mut new_lowest_idx = 0;
        let mut new_lowest_sim = MetricResult::max(dist_metric);

        for (idx, nbr) in self.neighbors.iter().enumerate() {
            let nbr = unsafe { nbr.load(Ordering::Relaxed).as_ref() };
            let Some((_, _, nbr_sim)) = nbr else {
                new_lowest_sim = MetricResult::min(dist_metric);
                new_lowest_idx = idx;
                break;
            };
            if nbr_sim < &new_lowest_sim {
                new_lowest_sim = *nbr_sim;
                new_lowest_idx = idx;
            }
        }

        *lowest_idx_guard = (new_lowest_idx as u8, new_lowest_sim);

        drop(lowest_idx_guard);

        match result {
            Ok(old_ptr) => {
                // Successful update
                unsafe {
                    if let Some((_, node, _)) = old_ptr.as_ref() {
                        (**node)
                            .try_get_data(cache)
                            .unwrap()
                            .remove_neighbor_by_id(self.get_id());
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
        }
    }

    pub fn remove_neighbor_by_id(&self, id: InternalId) {
        let _lock = self.freeze();
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

    pub fn remove_neighbor_by_index_and_id(&self, index: u8, id: InternalId) {
        let _lock = self.freeze();
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

    pub fn clone_neighbors(&self) -> Neighbors {
        let _guard = self.freeze();
        self.neighbors
            .iter()
            .map(|neighbor| unsafe {
                AtomicPtr::new(
                    neighbor
                        .load(Ordering::SeqCst)
                        .as_ref()
                        .map_or_else(ptr::null_mut, |neighbor| Box::into_raw(Box::new(*neighbor))),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn get_neighbors_raw(&self) -> &Neighbors {
        &self.neighbors
    }

    /// See [`crate::models::serializer::hnsw::node`] for how its calculated
    pub fn get_serialized_size(neighbors_len: usize) -> usize {
        neighbors_len * 19 + 49
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
