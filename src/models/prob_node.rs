use std::{
    collections::hash_map::Entry,
    ptr,
    sync::{
        atomic::{AtomicPtr, AtomicU32, Ordering},
        Arc,
    },
};

use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;

use crate::indexes::hnsw::offset_counter::{self, IndexFileId};

use self::offset_counter::HNSWIndexFileOffsetCounter;

use super::{
    buffered_io::{BufIoError, BufferManager},
    cache_loader::HNSWIndexCache,
    common::TSHashTable,
    lazy_item::{FileIndex, LazyItem},
    serializer::hnsw::RawDeserialize,
    types::{
        DistanceMetric, FileOffset, HNSWLevel, InternalId, MetricResult, NodePropMetadata,
        NodePropValue, ReplicaNodeKind, VectorData,
    },
    versioning::VersionNumber,
};

pub type SharedNode = *mut LazyItem<ProbNode, IndexFileId>;

pub struct LatestNode {
    pub latest: SharedNode,
    pub file_offset: FileOffset,
}

impl LatestNode {
    pub fn new(latest: SharedNode, file_offset: FileOffset) -> SharedLatestNode {
        Box::into_raw(Box::new(Self {
            latest,
            file_offset,
        }))
    }

    #[allow(invalid_reference_casting)]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn get_or_create_version(
        this: *mut Self,
        version: VersionNumber,
        synchronization_map: &TSHashTable<SharedLatestNode, ()>,
        cache: &HNSWIndexCache,
        file_id: IndexFileId,
        offset_counter: &HNSWIndexFileOffsetCounter,
    ) -> Result<(SharedNode, bool), BufIoError> {
        let self_ = unsafe { &*this };
        let latest = unsafe { &*self_.latest };
        let node = latest.try_get_data(cache)?;
        if node.version.load(Ordering::Relaxed) == *version {
            return Ok((self_.latest, false));
        }
        synchronization_map.lock_key_and_try(this, || {
            if node.version.load(Ordering::Relaxed) == *version {
                return Ok((self_.latest, false));
            }

            let offset = if node.hnsw_level.0 == 0 {
                offset_counter.next_level_0_offset()
            } else {
                offset_counter.next_offset()
            };

            unsafe {
                *(&latest.file_index as *const _ as *mut _) = FileIndex { file_id, offset };
            }

            node.version.store(*version, Ordering::Release);

            let cursor = cache.latest_version_links_bufman.open_cursor()?;
            cache
                .latest_version_links_bufman
                .seek_with_cursor(cursor, self_.file_offset.0 as u64)?;
            cache
                .latest_version_links_bufman
                .update_u32_with_cursor(cursor, offset.0)?;
            cache
                .latest_version_links_bufman
                .update_u32_with_cursor(cursor, *file_id)?;
            cache.latest_version_links_bufman.close_cursor(cursor)?;

            Ok((self_.latest, true))
        })
    }
}

pub type SharedLatestNode = *mut LatestNode;

pub type Neighbors = Box<[AtomicPtr<(InternalId, SharedLatestNode, MetricResult)>]>;

pub struct ProbNode {
    pub hnsw_level: HNSWLevel,
    pub version: AtomicU32,
    pub prop_value: Arc<NodePropValue>,
    pub prop_metadata: Option<Arc<NodePropMetadata>>,
    // Each neighbor is represented as (neighbor_id, neighbor_node, distance)
    neighbors: Neighbors,
    parent: AtomicPtr<LatestNode>,
    child: AtomicPtr<LatestNode>,
    pub lowest_index: RwLock<(u8, MetricResult)>,
}

unsafe impl Send for ProbNode {}
unsafe impl Sync for ProbNode {}

impl ProbNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hnsw_level: HNSWLevel,
        version: VersionNumber,
        prop_value: Arc<NodePropValue>,
        prop_metadata: Option<Arc<NodePropMetadata>>,
        parent: SharedLatestNode,
        child: SharedLatestNode,
        neighbors_count: usize,
        dist_metric: DistanceMetric,
    ) -> Self {
        let mut neighbors = Vec::with_capacity(neighbors_count);

        for _ in 0..neighbors_count {
            neighbors.push(AtomicPtr::new(ptr::null_mut()));
        }

        Self {
            hnsw_level,
            version: AtomicU32::new(*version),
            prop_value,
            prop_metadata,
            neighbors: neighbors.into_boxed_slice(),
            parent: AtomicPtr::new(parent),
            child: AtomicPtr::new(child),
            lowest_index: RwLock::new((0, MetricResult::min(dist_metric))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_neighbors_and_versions(
        hnsw_level: HNSWLevel,
        version: VersionNumber,
        prop_value: Arc<NodePropValue>,
        prop_metadata: Option<Arc<NodePropMetadata>>,
        neighbors: Neighbors,
        parent: SharedLatestNode,
        child: SharedLatestNode,
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
            version: AtomicU32::new(*version),
            prop_value,
            prop_metadata,
            neighbors,
            parent: AtomicPtr::new(parent),
            child: AtomicPtr::new(child),
            lowest_index: RwLock::new((lowest_idx as u8, lowest_sim)),
        }
    }

    pub fn get_parent(&self) -> SharedLatestNode {
        self.parent.load(Ordering::Acquire)
    }

    pub fn set_parent(&self, parent: SharedLatestNode) {
        self.parent.store(parent, Ordering::Release);
    }

    pub fn get_child(&self) -> SharedLatestNode {
        self.child.load(Ordering::Acquire)
    }

    pub fn set_child(&self, child: SharedLatestNode) {
        self.child.store(child, Ordering::Release);
    }

    pub fn get_id(&self) -> InternalId {
        match &self.prop_metadata {
            Some(metadata) => metadata.replica_id,
            None => self.prop_value.id,
        }
    }

    pub fn freeze(&self) -> RwLockReadGuard<'_, (u8, MetricResult)> {
        self.lowest_index.read()
    }

    pub fn add_neighbor(
        &self,
        neighbor_id: InternalId,
        neighbor_node: SharedLatestNode,
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
                        (*(**node).latest)
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

    pub fn remove_neighbor_by_id(&self, id: InternalId) -> bool {
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
                return true;
            }
        }
        false
    }

    pub fn is_neighbors_empty(&self) -> bool {
        let _lock = self.freeze();
        self.neighbors
            .iter()
            .all(|neighbor| neighbor.load(Ordering::Relaxed).is_null())
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
        neighbors_len * 13 + 31
    }

    pub fn build_from_raw(
        raw: <Self as RawDeserialize>::Raw,
        cache: &HNSWIndexCache,
        dummy_bufman: &BufferManager,
        pending_items: &mut FxHashMap<FileIndex<IndexFileId>, SharedNode>,
        latest_version_links: &mut FxHashMap<FileOffset, SharedLatestNode>,
        latest_version_links_cursor: u64,
    ) -> Result<Self, BufIoError> {
        let (
            hnsw_level,
            version,
            prop_value,
            prop_metadata,
            neighbors,
            parent_offset,
            child_offset,
        ) = raw;

        let parent = if parent_offset.0 != u32::MAX {
            match latest_version_links.entry(parent_offset) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let file_index = SharedLatestNode::deserialize_raw(
                        dummy_bufman, // not used
                        &cache.latest_version_links_bufman,
                        u64::MAX, // not used
                        latest_version_links_cursor,
                        parent_offset,
                        IndexFileId::invalid(),
                        cache,
                    )?;
                    let item = cache
                        .registry
                        .get(&HNSWIndexCache::combine_index(&file_index))
                        .unwrap_or_else(|| {
                            *pending_items
                                .entry(file_index)
                                .or_insert_with(|| LazyItem::new_pending(file_index))
                        });
                    let ptr = LatestNode::new(item, parent_offset);
                    entry.insert(ptr);
                    ptr
                }
            }
        } else {
            ptr::null_mut()
        };

        let child = if child_offset.0 != u32::MAX {
            match latest_version_links.entry(child_offset) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let file_index = SharedLatestNode::deserialize_raw(
                        dummy_bufman, // not used
                        &cache.latest_version_links_bufman,
                        u64::MAX, // not used
                        latest_version_links_cursor,
                        child_offset,
                        IndexFileId::invalid(),
                        cache,
                    )?;
                    let item = cache
                        .registry
                        .get(&HNSWIndexCache::combine_index(&file_index))
                        .unwrap_or_else(|| {
                            *pending_items
                                .entry(file_index)
                                .or_insert_with(|| LazyItem::new_pending(file_index))
                        });
                    let ptr = LatestNode::new(item, child_offset);
                    entry.insert(ptr);
                    ptr
                }
            }
        } else {
            ptr::null_mut()
        };

        let neighbors = neighbors
            .into_iter()
            .map(|neighbor| {
                let ptr = if let Some((id, neighbor_offset, score)) = neighbor {
                    Box::into_raw(Box::new((
                        id,
                        match latest_version_links.entry(neighbor_offset) {
                            Entry::Occupied(entry) => *entry.get(),
                            Entry::Vacant(entry) => {
                                let file_index = SharedLatestNode::deserialize_raw(
                                    dummy_bufman, // not used
                                    &cache.latest_version_links_bufman,
                                    u64::MAX, // not used
                                    latest_version_links_cursor,
                                    neighbor_offset,
                                    IndexFileId::invalid(),
                                    cache,
                                )?;
                                let item = cache
                                    .registry
                                    .get(&HNSWIndexCache::combine_index(&file_index))
                                    .unwrap_or_else(|| {
                                        *pending_items
                                            .entry(file_index)
                                            .or_insert_with(|| LazyItem::new_pending(file_index))
                                    });
                                let ptr = LatestNode::new(item, neighbor_offset);
                                entry.insert(ptr);
                                ptr
                            }
                        },
                        score,
                    )))
                } else {
                    ptr::null_mut()
                };
                Ok::<_, BufIoError>(AtomicPtr::new(ptr))
            })
            .collect::<Result<Neighbors, _>>()?;

        Ok(Self::new_with_neighbors_and_versions(
            hnsw_level,
            version,
            prop_value,
            prop_metadata,
            neighbors,
            parent,
            child,
            *cache.distance_metric.read().unwrap(),
        ))
    }

    /// Returns the kind of node it is
    pub fn replica_node_kind(&self) -> ReplicaNodeKind {
        let metadata = self.prop_metadata.as_ref().map(|pm| &*pm.vec);
        let internal_id = self.get_id();
        let vector_data = VectorData {
            id: Some(&internal_id),
            quantized_vec: &self.prop_value.vec,
            metadata,
        };
        vector_data.replica_node_kind()
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
