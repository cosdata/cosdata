use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[allow(unused)]
struct LRUCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    map: DashMap<K, (V, AtomicU64)>,
    capacity: usize,
}

#[allow(unused)]
impl<K, V> LRUCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    fn new(capacity: usize) -> Self {
        LRUCache {
            map: DashMap::new(),
            capacity,
        }
    }

    fn get(&self, key: &K) -> Option<V> {
        if let Some(mut entry) = self.map.get_mut(key) {
            let (value, timestamp) = entry.value_mut();
            timestamp.store(Self::current_time(), Ordering::SeqCst);
            Some(value.clone())
        } else {
            None
        }
    }

    fn insert(&self, key: K, value: V) {
        if self.map.len() >= self.capacity {
            self.evict_lru();
        }
        self.map.insert(key, (value, AtomicU64::new(Self::current_time())));
    }

    fn evict_lru(&self) {
        let mut oldest_key = None;
        let mut oldest_time = u64::MAX;

        for entry in self.map.iter() {
            let (key, (_, timestamp)) = entry.pair();
            let time = timestamp.load(Ordering::SeqCst);
            if time < oldest_time {
                oldest_time = time;
                oldest_key = Some(key.clone());
            }
        }

        if let Some(key) = oldest_key {
            self.map.remove(&key);
        }
    }

    fn current_time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_basic_usage() {
        let cache = LRUCache::new(2);

        cache.insert("key1", "value1");
        cache.insert("key2", "value2");

        match cache.get(&"key1") {
            Some(v) => assert_eq!("value1", v),
            None => assert!(false),
        }

        cache.insert("key3", "value3"); // This should evict key2

        match cache.get(&"key3") {
            Some(v) => assert_eq!("value3", v),
            None => assert!(false),
        }
    }
}
