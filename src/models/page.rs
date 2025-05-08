use super::{types::FileOffset, versioning::VersionHash};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
    vec::IntoIter,
};

#[derive(Debug, Clone)]
pub struct VersionedPagepool<const LEN: usize> {
    pub current_version: VersionHash,
    pub serialized_at: Arc<RwLock<Option<FileOffset>>>,
    pub pagepool: Arc<Pagepool<LEN>>,
    pub next: Arc<RwLock<Option<VersionedPagepool<LEN>>>>,
}

#[cfg(test)]
impl<const LEN: usize> PartialEq for VersionedPagepool<LEN> {
    fn eq(&self, other: &Self) -> bool {
        self.current_version == other.current_version
            && *self.serialized_at.read().unwrap() == *other.serialized_at.read().unwrap()
            && self.pagepool == other.pagepool
            && *self.next.read().unwrap() == *other.next.read().unwrap()
    }
}

impl<const LEN: usize> VersionedPagepool<LEN> {
    pub fn new(version: VersionHash) -> Self {
        Self {
            current_version: version,
            serialized_at: Arc::new(RwLock::new(None)),
            pagepool: Arc::new(Pagepool::default()),
            next: Arc::new(RwLock::new(None)),
        }
    }

    pub fn push(&self, version: VersionHash, id: u32) {
        if self.current_version != version {
            let next_read_guard = self.next.read().unwrap();
            if let Some(next) = &*next_read_guard {
                return next.push(version, id);
            }
            drop(next_read_guard);
            let mut next_write_guard = self.next.write().unwrap();
            if let Some(next) = &mut *next_write_guard {
                return next.push(version, id);
            }

            let new_next = Self::new(version);
            new_next.push(version, id);
            *next_write_guard = Some(new_next);
            return;
        }
        self.pagepool.push(id);
    }

    pub fn len(&self) -> usize {
        self.pagepool
            .inner
            .read()
            .unwrap()
            .iter()
            .map(|page| page.len)
            .sum::<usize>()
            + self
                .next
                .read()
                .unwrap()
                .as_ref()
                .map_or(0, |next| next.len())
    }

    #[allow(unused)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[allow(unused)]
    pub fn contains(&self, id: &u32) -> bool {
        self.pagepool.contains(id)
            || self
                .next
                .read()
                .unwrap()
                .as_ref()
                .is_some_and(|pool| pool.contains(id))
    }
}

pub struct VersionedPagepoolIter<const LEN: usize> {
    current_pool: Option<Arc<VersionedPagepool<LEN>>>,
    page_iter: Option<IntoIter<Page<LEN>>>,
    data_iter: Option<std::array::IntoIter<u32, LEN>>,
    current_page_len: usize,
    current_page_idx: usize,
}

impl<const LEN: usize> Iterator for VersionedPagepoolIter<LEN> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have a `data_iter`, try getting the next `u32`
            if let Some(ref mut data_iter) = self.data_iter {
                if self.current_page_idx < self.current_page_len {
                    if let Some(value) = data_iter.next() {
                        self.current_page_idx += 1;
                        return Some(value);
                    }
                }
                // Exhausted this page, move to the next one
                self.data_iter = None;
            }

            // If we have a `page_iter`, get the next page
            if let Some(ref mut page_iter) = self.page_iter {
                if let Some(page) = page_iter.next() {
                    self.data_iter = Some(page.data.into_iter());
                    self.current_page_len = page.len;
                    self.current_page_idx = 0;
                    continue;
                }
                // Exhausted all pages, move to the next VersionedPagepool
                self.page_iter = None;
            }

            // If we have a `current_pool`, move to the next one
            if let Some(pool) = self.current_pool.take() {
                if let Ok(next_pool) = pool.next.read() {
                    if let Some(ref next) = *next_pool {
                        self.current_pool = Some(Arc::new(next.clone()));
                        if let Ok(pages) = next.pagepool.inner.read() {
                            self.page_iter = Some(pages.clone().into_iter());
                        }
                        continue;
                    }
                }
            }

            // No more data
            return None;
        }
    }
}

impl<const LEN: usize> VersionedPagepool<LEN> {
    pub fn iter(&self) -> VersionedPagepoolIter<LEN> {
        let page_iter = self
            .pagepool
            .inner
            .read()
            .ok()
            .map(|pages| pages.clone().into_iter());

        VersionedPagepoolIter {
            current_pool: Some(Arc::new(self.clone())),
            page_iter,
            data_iter: None,
            current_page_len: 0,
            current_page_idx: 0,
        }
    }
}

#[derive(Default, Debug)]
pub struct Pagepool<const LEN: usize> {
    pub inner: RwLock<Vec<Page<LEN>>>,
}

#[cfg(test)]
impl<const LEN: usize> PartialEq for Pagepool<LEN> {
    fn eq(&self, other: &Self) -> bool {
        *self.inner.read().unwrap() == *other.inner.read().unwrap()
    }
}

impl<const LEN: usize> Pagepool<LEN> {
    pub fn push(&self, data: u32) {
        let mut inner = self.inner.write().unwrap();
        if let Some(last) = inner.last_mut() {
            if !last.is_full() {
                last.push(data);
                return;
            }
        }
        let mut page = Page::<LEN>::new();
        page.push(data);
        inner.push(page);
    }

    #[allow(unused)]
    pub fn contains(&self, data: &u32) -> bool {
        self.inner
            .read()
            .unwrap()
            .iter()
            .any(|p| p.data.contains(data))
    }
}

#[derive(Debug, Clone)]
pub struct Page<const LEN: usize> {
    pub data: [u32; LEN],
    pub len: usize,
    pub serialized_at: Arc<RwLock<Option<FileOffset>>>,
    pub dirty: Arc<AtomicBool>,
}

impl<const LEN: usize> PartialEq for Page<LEN> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.len == other.len
    }
}

impl<const LEN: usize> std::ops::Deref for Page<LEN> {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        &self.data[..self.len]
    }
}

impl<const LEN: usize> Default for Page<LEN> {
    fn default() -> Self {
        Self {
            data: [u32::MAX; LEN],
            len: 0,
            serialized_at: Arc::new(RwLock::new(None)),
            dirty: Arc::new(AtomicBool::new(true)),
        }
    }
}

impl<const LEN: usize> Page<LEN> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, data: u32) {
        self.data[self.len] = data;
        self.len += 1;
        self.dirty.store(true, Ordering::Release);
    }

    fn is_full(&self) -> bool {
        self.len == LEN
    }
}

impl<const LEN: usize> AsRef<[u32; LEN]> for Page<LEN> {
    fn as_ref(&self) -> &[u32; LEN] {
        &self.data
    }
}

// #[cfg(test)]
// mod page_tests {
//     use super::*;

//     use std::collections::HashSet;

//     use crate::models::{
//         buffered_io::{BufferManager, BufferManagerFactory},
//         cache_loader::NodeRegistry,
//         lazy_load::FileIndex,
//         serializer::CustomSerialize,
//         types::FileOffset,
//         versioning::Hash,
//     };
//     use std::sync::Arc;

//     use tempfile::{tempdir, TempDir};

//     fn setup_test() -> (
//         Arc<BufferManagerFactory>,
//         Arc<BufferManager>,
//         u64,
//         TempDir,
//         Arc<NodeRegistry>,
//     ) {
//         let root_version_id = Hash::from(0);

//         let dir = tempdir().unwrap();
//         let bufmans = Arc::new(BufferManagerFactory::new(
//             dir.as_ref().into(),
//             |root, ver| root.join(format!("{}.index", **ver)),
//         ));

//         let cache = Arc::new(NodeRegistry::new(1000, bufmans.clone()));
//         let bufman = bufmans.get(&root_version_id).unwrap();
//         let cursor = bufman.open_cursor().unwrap();
//         (bufmans, bufman, cursor, dir, cache)
//     }

//     #[test]
//     fn test_serialize_deserialize_page() {
//         let mut page_pool = Pagepool::<10>::default();
//         let mut skipm: HashSet<u64> = HashSet::new();

//         for i in 0..10 * 10_u32 {
//             page_pool.push(i);
//         }

//         let root_version_id = Hash::from(0);
//         let root_version_number = 0;

//         let (bufmgr_factory, bufmg, cursor, temp_dir, cache) = setup_test();
//         let offset = page_pool.serialize(bufmgr_factory.clone(), root_version_id, cursor);

//         assert!(offset.is_ok());

//         let offset = offset.unwrap();
//         bufmg.close_cursor(cursor).unwrap();

//         let deser = Pagepool::<10>::deserialize(
//             bufmgr_factory.clone(),
//             FileIndex::Valid {
//                 offset: FileOffset(offset),
//                 version_id: root_version_id,
//                 version_number: root_version_number,
//             },
//             cache.clone(),
//             0_u16,
//             &mut skipm,
//         );
//         assert!(deser.is_ok());
//         let deser = deser.unwrap();

//         assert_eq!(page_pool, deser);
//     }
// }
