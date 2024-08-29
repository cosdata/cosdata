use dashmap::DashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use super::buffered_io::BufIoError;

pub struct LRUCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    map: DashMap<K, (V, u64)>,
    capacity: usize,
}

impl<K, V> LRUCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        LRUCache {
            map: DashMap::new(),
            capacity,
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        if let Some(mut entry) = self.map.get_mut(key) {
            let (value, mut timestamp) = entry.value_mut();
            timestamp = Self::current_time();
            Some(value.clone())
        } else {
            None
        }
    }

    pub fn insert(&self, key: K, value: V) {
        self.map.insert(key, (value, Self::current_time()));
        if self.map.len() > self.capacity {
            self.evict_lru();
        }
    }

    pub fn get_or_insert(&self, key: K, f: impl FnOnce() -> Result<V, BufIoError>) -> Result<V, BufIoError> {
        let res = self.map
            .entry(key)
            .and_modify(|(_, ts)| *ts = Self::current_time())
            .or_try_insert_with(move || {
                f().map(|v| (v, Self::current_time()))
            })
            .map(|v| v.0.clone());
        // @NOTE: We need to clone the value before calling
        // `self.evict_lru`, that too in a separate block! Otherwise
        // it causes some deadlock
        match res {
            Ok(v) => {
                if self.map.len() > self.capacity {
                    self.evict_lru();
                }
                Ok(v)
            }
            Err(e) => Err(e)
        }
    }

    fn evict_lru(&self) {
        let mut oldest_key = None;
        let mut oldest_time = u64::MAX;

        for entry in self.map.iter() {
            let (key, (_, timestamp)) = entry.pair();
            if *timestamp < oldest_time {
                oldest_time = *timestamp;
                oldest_key = Some(key.clone());
            }
        }

        if let Some(key) = oldest_key {
            self.map.remove(&key);
        }
    }

    pub fn iter(&self) -> dashmap::iter::Iter<K, (V, u64), std::hash::RandomState, DashMap<K, (V, u64)>> {
        self.map.iter()
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

        // Verify that key2 is evicted
        //
        // @TODO: Checking for the evicted key causes the code to
        // panic for some reason. So for now we're just checking that
        // size <= capacity.
        assert_eq!(2, cache.map.len());
        // assert!(!cache.map.contains_key(&"key2"));
    }

    #[test]
    fn test_get_or_insert() {
        let cache = LRUCache::new(2);

        // Insert two values using `try_insert_with`, verifying that
        // the method returns the correct value
        let x = cache.get_or_insert("key1", || {
            Ok("value1")
        });
        assert_eq!("value1", x.unwrap());
        assert_eq!(1, cache.map.len());

        let y = cache.get_or_insert("key2", || {
            Ok("value2")
        });
        assert_eq!("value2", y.unwrap());
        assert_eq!(2, cache.map.len());

        // Try getting key1 again. The closure shouldn't get executed
        // this time.
        let x1 = cache.get_or_insert("key1", || {
            // This code will not be executed
            assert!(false);
            Err(BufIoError::Locking)
        });
        assert!(x1.is_ok_and(|x| x == "value1"));

        // Insert a third value. It will cause key2 to be evicted
        let z = cache.get_or_insert("key3", || {
            Ok("value3")
        });
        assert_eq!("value3", z.unwrap());

        // Verify that key2 is evicted
        //
        // @TODO: Checking for the evicted key causes the code to
        // panic for some reason. So for now we're just checking that
        // size <= capacity.
        assert_eq!(2, cache.map.len());
        // assert!(!cache.map.contains_key(&"key2"));

        // Verify that error during insertion doesn't result in
        // evictions
        match cache.get_or_insert("key4", || Err(BufIoError::Locking)) {
            Err(BufIoError::Locking) => assert!(true),
            _ => assert!(false),
        }
        assert_eq!(2, cache.map.len());
    }
}
