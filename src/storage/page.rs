use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};

use crate::models::{types::FileOffset, versioning::Hash};

#[derive(Debug, Clone)]
pub struct VersionedPagepool<const LEN: usize> {
    pub current_version: Hash,
    pub serialized_at: Arc<RwLock<Option<FileOffset>>>,
    pub pagepool: Pagepool<LEN>,
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
    pub fn new(version: Hash) -> Self {
        Self {
            current_version: version,
            serialized_at: Arc::new(RwLock::new(None)),
            pagepool: Pagepool::default(),
            next: Arc::new(RwLock::new(None)),
        }
    }

    pub fn push(&mut self, version: Hash, vector_id: u32) {
        if self.current_version != version {
            let mut next_write_guard = self.next.write().unwrap();
            if let Some(next) = &mut *next_write_guard {
                return next.push(version, vector_id);
            }

            let mut new_next = Self::new(version);
            new_next.push(version, vector_id);
            *next_write_guard = Some(new_next);
            return;
        }
        self.pagepool.push(vector_id);
    }

    pub fn len(&self) -> usize {
        self.pagepool
            .inner
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
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Pagepool<const LEN: usize> {
    pub inner: Vec<Page<LEN>>,
}

impl<const LEN: usize> Pagepool<LEN> {
    pub fn push(&mut self, data: u32) {
        if let Some(last) = self.inner.last_mut() {
            if !last.is_full() {
                last.push(data);
                return;
            }
        }
        let mut page = Page::<LEN>::new();
        page.push(data);
        self.inner.push(page);
    }

    pub fn push_chunk(&mut self, chunk: [u32; LEN]) {
        self.inner.push(Page::<LEN>::from_data(chunk))
    }

    pub fn contains(&self, data: u32) -> bool {
        self.inner.iter().any(|p| p.data.contains(&data))
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

impl<const LEN: usize> Page<LEN> {
    pub fn new() -> Self {
        Self {
            data: [u32::MAX; LEN],
            len: 0,
            serialized_at: Arc::new(RwLock::new(None)),
            dirty: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn push(&mut self, data: u32) {
        self.data[self.len] = data;
        self.len += 1;
        self.dirty.store(true, Ordering::Release);
    }

    fn from_data(data: [u32; LEN]) -> Self {
        Self {
            data,
            len: 0,
            serialized_at: Arc::new(RwLock::new(None)),
            dirty: Arc::new(AtomicBool::new(true)),
        }
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
