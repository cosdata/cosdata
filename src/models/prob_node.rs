use std::{
    ptr, sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, Mutex, MutexGuard,
    }
};

use super::{
    cache_loader::HNSWIndexCache,
    prob_lazy_load::{lazy_item::ProbLazyItem, lazy_item_array::ProbLazyItemArray},
    types::{DistanceMetric, HNSWLevel, MetricResult, NodePropMetadata, NodePropValue, VectorId},
};

pub type SharedNode = *mut ProbLazyItem<ProbNode>;
pub type Neighbors = Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]>;

pub struct ProbNode {
    pub hnsw_level: HNSWLevel,
    pub prop_value: Arc<NodePropValue>,
    pub prop_metadata: Option<Arc<NodePropMetadata>>,
    // (neighbor_id, neighbor_node, distance)
    // even though `VectorId` is an u64 we don't need the full range here.
    neighbors: Neighbors,
    parent: AtomicPtr<ProbLazyItem<ProbNode>>,
    child: AtomicPtr<ProbLazyItem<ProbNode>>,
    pub versions: ProbLazyItemArray<ProbNode, 8>,
    lowest_index: Mutex<(u8, MetricResult)>,
    pub root_version: SharedNode,
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
            versions: ProbLazyItemArray::new(),
            lowest_index: Mutex::new((0, MetricResult::min(dist_metric))),
            root_version: ptr::null_mut(),
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
        versions: ProbLazyItemArray<ProbNode, 8>,
        root_version: SharedNode,
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

    pub fn get_id(&self) -> VectorId {
        match &self.prop_metadata {
            Some(m_prop) => {
                let metadata_id = (m_prop.id.0 as u64) << 56;
                let vector_id = metadata_id | self.prop_value.id.0;
                VectorId(vector_id)
            }
            None => self.prop_value.id.clone()
        }
    }

    /// Returns two ids for the same prob node (original id, node_id)
    ///
    /// In case of metadata fields, one vector can have multiple
    /// replicas. In that case, original_id = user provided vector id
    /// and node_id = internal node id computed from vector id and
    /// metadata id
    ///
    /// In case there are no metadata fields, original_id will be the
    /// same as node_id
    pub fn get_ids(&self) -> (VectorId, VectorId) {
        if self.prop_metadata.is_some() {
            (self.prop_value.id.clone(), self.get_id())
        } else {
            (self.prop_value.id.clone(), self.prop_value.id.clone())
        }
    }

    pub fn lock_lowest_index(&self) -> MutexGuard<'_, (u8, MetricResult)> {
        self.lowest_index.lock().unwrap()
    }

    pub fn add_neighbor(
        &self,
        neighbor_id: u32,
        neighbor_node: SharedNode,
        dist: MetricResult,
        cache: &HNSWIndexCache,
        dist_metric: DistanceMetric,
    ) -> Option<u8> {
        // First find an empty slot or the slot with lowest similarity
        let mut lowest_idx_guard = self.lock_lowest_index();
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
        }
    }

    pub fn remove_neighbor_by_id(&self, id: u32) {
        let _lock = self.lock_lowest_index();
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
        let _lock = self.lock_lowest_index();
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

    /// See [`crate::models::serializer::dense::node`] for how its calculated
    pub fn get_serialized_size(neighbors_len: usize) -> usize {
        neighbors_len * 19 + 129
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
