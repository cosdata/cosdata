use dashmap::DashMap;
use half::f16;
use rand::Rng;
use std::iter::Iterator;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// Calculates counter age, while considering a possibility of
// wraparound (with the assumption that wraparound will happen at most
// once)
//
// `global_value` > `item_value` indicates that wraparound has
// happened. In that case, the age is calculated taking that into
// consideration.
//
fn counter_age(global_value: u32, item_value: u32) -> u32 {
    if global_value >= item_value {
        global_value - item_value
    } else {
        // until wrap around + after wraparound
        (u32::MAX - item_value) + global_value
    }
}

pub struct EvictionIndex {
    inner: [AtomicU64; 256],
}

impl EvictionIndex {
    fn new() -> Self {
        Self {
            // @NOTE: Uses inline constants; will only work for msrv =
            // 1.79.0
            inner: core::array::from_fn(|_| AtomicU64::new(u64::MAX)),
        }
    }

    fn idx(counter: u32) -> usize {
        counter as usize % 256
    }

    fn is_empty(&self, idx: usize) -> bool {
        self.inner[idx as usize].load(Ordering::SeqCst) == u64::MAX
    }

    fn clear(&self, idx: usize) {
        self.inner[idx].store(u64::MAX, Ordering::SeqCst)
    }

    fn on_cache_miss(&self, counter: u32, key: u64) {
        let i = Self::idx(counter);
        if self.is_empty(i) {
            self.inner[i].store(key, Ordering::SeqCst);
        }
    }

    fn on_cache_hit(&self, old_counter: u32, new_counter: u32, key: u64) {
        let i = Self::idx(old_counter);
        if self.inner[i].load(Ordering::SeqCst) == key {
            self.clear(i);
            let j = Self::idx(new_counter);
            if self.is_empty(j) {
                self.inner[j].store(key, Ordering::SeqCst);
            }
        }
    }

    fn get_keys(&self, max: u8) -> Vec<(u8, u64)> {
        let mut result = vec![];
        for (i, x) in self.inner.iter().enumerate() {
            if result.len() > max as usize {
                break;
            }
            let key = x.load(Ordering::SeqCst);
            if key < u64::MAX {
                result.push((i as u8, key));
            }
        }
        result
    }

    fn remove(&self, idx: u8) {
        self.clear(idx as usize)
    }
}

pub struct ProbEviction {
    // Probability of eviction per call. E.g. A value of 0.1 means
    // eviction will be randomly triggered with 10% probability on each call
    prob: f16,
    // Parameter to tune the "aggressiveness" of eviction i.e. higher
    // value means more aggressive
    lambda: f16,
}

impl ProbEviction {
    pub fn new(prob: f16) -> Self {
        Self {
            prob,
            lambda: f16::from_f32_const(0.01),
        }
    }

    fn should_trigger(&self) -> bool {
        self.prob > f16::from_f32(rand::thread_rng().gen())
    }

    fn eviction_probability(&self, global_counter: u32, counter_value: u32) -> f32 {
        let age = counter_age(global_counter, counter_value);
        let recency_prob = (-self.lambda.to_f32() * age as f32).exp();
        let eviction_prob = 1.0 - recency_prob;
        eviction_prob
    }

    fn should_evict(&self, global_counter: u32, counter_value: u32) -> bool {
        let eviction_prob = self.eviction_probability(global_counter, counter_value);
        eviction_prob > rand::thread_rng().gen()
    }
}

#[allow(unused)]
pub enum EvictStrategy {
    // Eviction will happen immediately after insertion
    Immediate,
    // All extra items will be evicted together at a probabilistically
    // calculated frequency
    Probabilistic(ProbEviction),
}

pub struct LRUCache<K, V>
where
    K: Eq + std::hash::Hash + Clone + Into<u64> + From<u64>,
    V: Clone,
{
    // Store value and counter value
    map: DashMap<K, (V, u32)>,
    capacity: usize,
    // Global counter
    counter: AtomicU32,
    evict_strategy: EvictStrategy,
    index: EvictionIndex,
    evict_hook: Option<fn(&V)>,
}

/// Wrapper for the value that's returned from the LRUCache when
/// trying to get_or_insert in a single operation. Useful for
/// indicating whether there was a cache hit or a miss.
pub enum CachedValue<V> {
    Hit(V),
    Miss(V),
}

impl<V> CachedValue<V> {
    pub fn inner(self) -> V {
        match self {
            Self::Hit(v) => v,
            Self::Miss(v) => v,
        }
    }
}

impl<K, V> LRUCache<K, V>
where
    K: Eq + std::hash::Hash + Clone + Into<u64> + From<u64>,
    V: Clone,
{
    pub fn new(capacity: usize, evict_strategy: EvictStrategy) -> Self {
        LRUCache {
            map: DashMap::new(),
            counter: AtomicU32::new(0),
            index: EvictionIndex::new(),
            evict_hook: None,
            capacity,
            evict_strategy,
        }
    }

    // Constructs a new LRUCache with probabilistic eviction strategy
    pub fn with_prob_eviction(capacity: usize, prob: f32) -> Self {
        let strategy = EvictStrategy::Probabilistic(ProbEviction::new(f16::from_f32_const(prob)));
        Self::new(capacity, strategy)
    }

    pub fn set_evict_hook(&mut self, hook: Option<fn(&V)>) {
        self.evict_hook = hook;
    }

    /// Returns an entry from the cache
    ///
    /// None will be returned if the cache doesn't contain the key
    pub fn get(&self, key: &K) -> Option<V> {
        if let Some(mut entry) = self.map.get_mut(key) {
            let (value, counter_val) = entry.value_mut();
            let old_counter = *counter_val;
            let new_counter = self.increment_counter();
            *counter_val = new_counter;
            self.index
                .on_cache_hit(old_counter, new_counter, key.clone().into());
            Some(value.clone())
        } else {
            None
        }
    }

    /// Inserts an entry into the cache
    ///
    /// Note that if the entry is already present in cache, it will be
    /// overwritten
    pub fn insert(&self, key: K, value: V) {
        let counter = self.increment_counter();
        self.map.insert(key.clone(), (value, counter));
        self.index.on_cache_miss(counter, key.into());
        // self.evict();
    }

    /// Gets the value from the cache if it exists, else tries to
    /// insert the result of the fn `f` into the cache and returns the
    /// same
    pub fn get_or_insert<E>(
        &self,
        key: K,
        f: impl FnOnce() -> Result<V, E>,
    ) -> Result<CachedValue<V>, E> {
        let mut inserted = false;
        let k1 = key.clone();
        let k2 = key.clone();
        let res = self
            .map
            .entry(key)
            .and_modify(|(_, counter)| {
                let old_counter = counter.clone();
                let new_counter = self.increment_counter();
                self.index.on_cache_hit(old_counter, new_counter, k1.into());
                *counter = new_counter;
            })
            .or_try_insert_with(|| {
                inserted = true;
                let counter = self.increment_counter();
                self.index.on_cache_miss(counter, k2.into());
                f().map(|v| (v, counter))
            })
            .map(|v| v.0.clone());
        // @NOTE: We need to clone the value before calling
        // `self.evict`, that too in a separate block! Otherwise
        // it causes some deadlock
        match res {
            Ok(v) => {
                if inserted {
                    // self.evict();
                    Ok(CachedValue::Miss(v))
                } else {
                    Ok(CachedValue::Hit(v))
                }
            }
            Err(e) => Err(e),
        }
    }

    fn evict(&self) {
        if self.map.len() > self.capacity {
            match &self.evict_strategy {
                EvictStrategy::Immediate => self.evict_lru(),
                EvictStrategy::Probabilistic(prob) => {
                    if self.map.len() > self.capacity && prob.should_trigger() {
                        self.evict_lru_probabilistic(&prob);
                    }
                }
            }
        }
    }

    fn evict_lru(&self) {
        let mut oldest_pair = None;
        let mut oldest_counter = u32::MAX;

        for entry in self.map.iter() {
            let (key, (value, counter_val)) = entry.pair();
            if *counter_val < oldest_counter {
                oldest_counter = *counter_val;
                oldest_pair = Some((key.clone(), value.clone()));
            }
        }

        if let Some((key, value)) = oldest_pair {
            // If item didn't exist it will return None. This can
            // happen if another thread finds the same item to evict
            // and "wins". This implies for temporarily the dashmap
            // size could exceed max capacity. It's fine for now but
            // needs to be fixed.
            if let Some(evict_hook) = self.evict_hook {
                evict_hook(&value);
            }
            let removed = self.map.remove(&key);
            if removed.is_none() {
                log::warn!("Item already evicted by another thread");
            }
        }
    }

    fn evict_lru_probabilistic(&self, strategy: &ProbEviction) {
        let num_to_evict = (1.0_f32 / strategy.prob.to_f32()) as u8;
        if num_to_evict > 0 {
            let global_counter = self.counter.load(Ordering::SeqCst);
            let mut pairs_to_evict = Vec::with_capacity(num_to_evict as usize);
            // @TODO: What if num_to_evict is > 256?
            for (idx, key) in self.index.get_keys(num_to_evict as u8) {
                if pairs_to_evict.len() as u8 >= num_to_evict {
                    break;
                }
                if let Some(entry) = self.map.get(&K::from(key)) {
                    let (key, (value, counter_val)) = entry.pair();
                    if strategy.should_evict(global_counter, *counter_val) {
                        // @NOTE: We need to collect the pairs in a
                        // vector and remove the keys from the dashmap
                        // later whereas values are used for calling
                        // `evict_hook` (if specified). Directly
                        // calling the `remove` method here causes a
                        // deadlock because of the existing reference
                        // into the dashmap. See `DashMap.remove` docs
                        // for more info.
                        pairs_to_evict.push((idx, key.clone(), value.clone()));
                    }
                }
            }
            for (idx, key, value) in pairs_to_evict {
                if let Some(evict_hook) = self.evict_hook {
                    evict_hook(&value)
                }
                self.map.remove(&key);
                self.index.remove(idx);
            }
        }
    }

    pub fn iter(
        &self,
    ) -> dashmap::iter::Iter<K, (V, u32), std::hash::RandomState, DashMap<K, (V, u32)>> {
        self.map.iter()
    }

    pub fn values(&self) -> Values<K, V> {
        Values {
            iter: self.map.iter(),
        }
    }

    fn increment_counter(&self) -> u32 {
        self.counter.fetch_add(1, Ordering::SeqCst)
    }
}

pub struct Values<'a, K: 'a, V: 'a> {
    iter: dashmap::iter::Iter<'a, K, (V, u32), std::hash::RandomState, DashMap<K, (V, u32)>>,
}

impl<'a, K, V> Iterator for Values<'a, K, V>
where
    K: 'a + Eq + std::hash::Hash + Clone,
    V: 'a + Clone,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        let tuple = self.iter.next();
        tuple.map(|entry| entry.value().0.clone())
    }
}

#[cfg(test)]
mod tests {

    use std::{collections::HashMap, sync::Arc, thread};

    use super::*;

    // #[test]
    // fn test_basic_usage() {
    //     let cache: LRUCache<u64, &'static str> = LRUCache::new(2, EvictStrategy::Immediate);

    //     cache.insert(1, "value1");
    //     cache.insert(2, "value2");

    //     match cache.get(&1) {
    //         Some(v) => assert_eq!("value1", v),
    //         None => assert!(false),
    //     }

    //     cache.insert(3, "value3"); // This should evict key2

    //     match cache.get(&3) {
    //         Some(v) => assert_eq!("value3", v),
    //         None => assert!(false),
    //     }

    //     // Verify that key2 is evicted
    //     assert_eq!(2, cache.map.len());
    //     assert!(!cache.map.contains_key(&2));
    // }

    // #[derive(Debug)]
    // struct FakeError(&'static str);

    // #[test]
    // fn test_get_or_insert() {
    //     let cache: LRUCache<u64, &'static str> = LRUCache::new(2, EvictStrategy::Immediate);

    //     // Insert two values using `try_insert_with`, verifying that
    //     // the method returns the correct value
    //     let x = cache
    //         .get_or_insert::<FakeError>(1, || Ok("value1"))
    //         .map(|entry| entry.inner());
    //     assert_eq!("value1", x.unwrap());
    //     assert_eq!(1, cache.map.len());

    //     let y = cache
    //         .get_or_insert::<FakeError>(2, || Ok("value2"))
    //         .map(|entry| entry.inner());
    //     assert_eq!("value2", y.unwrap());
    //     assert_eq!(2, cache.map.len());

    //     // Try getting key1 again. The closure shouldn't get executed
    //     // this time.
    //     let x1 = cache
    //         .get_or_insert(1, || {
    //             // This code will not be executed
    //             assert!(false);
    //             Err(FakeError("must not be called"))
    //         })
    //         .map(|entry| entry.inner());
    //     assert!(x1.is_ok_and(|x| x == "value1"));

    //     // Insert a third value. It will cause key2 to be evicted
    //     let z = cache
    //         .get_or_insert::<FakeError>(3, || Ok("value3"))
    //         .map(|entry| entry.inner());
    //     assert_eq!("value3", z.unwrap());

    //     // Verify that key2 is evicted
    //     assert_eq!(2, cache.map.len());
    //     assert!(!cache.map.contains_key(&2));

    //     // Verify that error during insertion doesn't result in
    //     // evictions
    //     match cache.get_or_insert::<FakeError>(4, || Err(FakeError("something went wrong"))) {
    //         Err(FakeError(msg)) => assert_eq!("something went wrong", msg),
    //         _ => assert!(false),
    //     }
    //     assert_eq!(2, cache.map.len());
    // }

    // #[test]
    // fn test_conc_get_or_insert() {
    //     let inner: LRUCache<u64, &'static str> = LRUCache::new(2, EvictStrategy::Immediate);
    //     let cache = Arc::new(inner);

    //     // Try concurrently inserting the same entry from 2 threads
    //     let t1 = {
    //         let c = cache.clone();
    //         thread::spawn(move || {
    //             let x = c
    //                 .get_or_insert::<FakeError>(1, || Ok("value1"))
    //                 .map(|entry| entry.inner());
    //             assert_eq!("value1", x.unwrap());
    //         })
    //     };

    //     let t2 = {
    //         let c = cache.clone();
    //         thread::spawn(move || {
    //             let x = c
    //                 .get_or_insert::<FakeError>(1, || Ok("value1"))
    //                 .map(|entry| entry.inner());
    //             assert_eq!("value1", x.unwrap());
    //         })
    //     };

    //     t1.join().unwrap();
    //     t2.join().unwrap();

    //     assert_eq!(1, cache.map.len());

    //     // Insert 2nd entry
    //     let y = cache
    //         .get_or_insert::<FakeError>(2, || Ok("value2"))
    //         .map(|entry| entry.inner());
    //     assert_eq!("value2", y.unwrap());
    //     assert_eq!(2, cache.map.len());

    //     // Insert 3rd and 4th entries in separate threads
    //     let t3 = {
    //         let c = cache.clone();
    //         thread::spawn(move || {
    //             let x = c
    //                 .get_or_insert::<FakeError>(3, || Ok("value3"))
    //                 .map(|entry| entry.inner());
    //             assert_eq!("value3", x.unwrap());
    //         })
    //     };

    //     let t4 = {
    //         let c = cache.clone();
    //         thread::spawn(move || {
    //             let x = c
    //                 .get_or_insert::<FakeError>(4, || Ok("value4"))
    //                 .map(|entry| entry.inner());
    //             assert_eq!("value4", x.unwrap());
    //         })
    //     };

    //     t3.join().unwrap();
    //     t4.join().unwrap();

    //     // Verify cache eviction
    //     //
    //     // @NOTE: Sometimes only one item is evicted instead of
    //     // two. This because the two threads find the same item to
    //     // evict and only one of them succeeds at actually removing it
    //     // from the the map. To be fixed later.
    //     let size = cache.map.len();
    //     // assert_eq!(2, size);
    //     assert!(size == 2 || size == 3);
    // }

    #[test]
    fn test_values_iterator() {
        let cache: LRUCache<u64, &'static str> = LRUCache::new(4, EvictStrategy::Immediate);

        cache.insert(1, "value1");
        cache.insert(2, "value2");
        cache.insert(3, "value3");
        cache.insert(4, "value4");

        let mut values = cache.values().collect::<Vec<&'static str>>();
        values.sort();
        assert_eq!(vec!["value1", "value2", "value3", "value4"], values);
    }

    fn gen_rand_nums(rng: &mut rand::rngs::ThreadRng, n: u64, min: u32, max: u32) -> Vec<u32> {
        (0..n).map(|_| rng.gen_range(min..max)).collect()
    }

    #[test]
    fn test_eviction_probability() {
        let prob = ProbEviction::new(f16::from_f32_const(0.03125));

        // Without wraparound
        let global_counter = 1000;
        let results = (1..=global_counter)
            .map(|n| prob.eviction_probability(global_counter, n))
            .collect::<Vec<f32>>();
        // Check that the eviction probability reduces with decrease
        // in counter age, i.e. the results vector is sorted in
        // descending order.
        assert!(results.as_slice().windows(2).all(|w| w[0] >= w[1]));

        // With wraparound
        let global_counter = 2147483647_u32;
        let mut rng = rand::thread_rng();
        // Generate some counter values before wraparound. These will
        // be > global_counter and < u32::MAX
        let mut counter_vals: Vec<u32> = gen_rand_nums(&mut rng, 100, global_counter, u32::MAX);
        counter_vals.sort();
        // Generate some counter values after wraparound. These will
        // be > 0 and < global_counter
        let mut after_wraparound: Vec<u32> = gen_rand_nums(&mut rng, 100, 0, global_counter);
        after_wraparound.sort();
        // As the global counter is very large, add some known values
        // closer to the global counter
        let mut recent: Vec<u32> = (1..=100).map(|n| global_counter - n).collect();
        recent.sort();

        // Concatenate the above inputs in order
        counter_vals.append(&mut after_wraparound);
        counter_vals.append(&mut recent);

        let results = counter_vals
            .into_iter()
            .map(|n| prob.eviction_probability(global_counter, n))
            .collect::<Vec<f32>>();
        // Check that the eviction probability reduces with increase
        // in counter value, i.e. the results vector is sorted in
        // descending order.
        assert!(results.as_slice().windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn test_eviction_index() {
        let index: EvictionIndex = EvictionIndex::new();

        let mut global_counter = 0;

        // A mapping of keys -> counters
        let mut m: HashMap<u64, u32> = HashMap::new();

        // Simulate insertion of 256 items
        for i in 1..=256 {
            index.on_cache_miss(global_counter, i);
            m.insert(i, global_counter);
            global_counter += 1;
        }

        let expected = (1..=256).collect::<Vec<u64>>();
        let actual = index
            .inner
            .iter()
            .map(|x| AtomicU64::load(x, Ordering::SeqCst))
            .collect::<Vec<u64>>();

        // Verify the contents of the index
        assert_eq!(expected, actual);

        // Simulate insertion of 10 more items
        for i in 257..267 {
            index.on_cache_miss(global_counter, i);
            m.insert(i, global_counter);
            global_counter += 1;
        }

        let actual2 = index
            .inner
            .iter()
            .map(|x| AtomicU64::load(x, Ordering::SeqCst))
            .collect::<Vec<u64>>();

        // Assert that the index contents are unchanged
        assert_eq!(expected, actual2);

        // Simulate `get` for a known item
        {
            let x = 147_u64;
            let new_counter = global_counter;
            let old_counter = m.insert(x, new_counter).unwrap();
            let old_id = old_counter as usize % 256;

            assert_eq!(146, old_id);

            // Find value in slot corresponding to new counter
            let new_id = new_counter as usize % 256;
            let new_slot_val = index.inner[new_id].load(Ordering::SeqCst);

            index.on_cache_hit(old_counter, new_counter, x);

            // Slot corresponding to old counter is cleared
            assert_eq!(u64::MAX, index.inner[old_id].load(Ordering::SeqCst));

            // Slot corresponding to new counter is unchanged
            assert_eq!(new_slot_val, index.inner[new_id].load(Ordering::SeqCst));
        }

        // Simulate `get` for another known item such that the new
        // counter is 256 + 147, so that it matches the same slot in
        // the index that was emptied earlier
        {
            let y = 202_u64;
            let new_counter = 256 + 146;
            let old_counter = m.insert(y, new_counter).unwrap();
            index.on_cache_hit(old_counter, new_counter, y);

            // Slot corresponding to old counter is cleared
            assert_eq!(
                u64::MAX,
                index.inner[old_counter as usize % 256].load(Ordering::SeqCst)
            );

            // Slot corresponding to new counter has this value
            assert_eq!(
                202,
                index.inner[new_counter as usize % 256].load(Ordering::SeqCst)
            );
        }
    }

    // #[test]
    // fn test_evict_hook() {
    //     let mut cache: LRUCache<u64, &'static str> = LRUCache::new(2, EvictStrategy::Immediate);
    //     cache.set_evict_hook(Some(|&value| {
    //         assert_eq!("value2", value);
    //     }));

    //     cache.insert(1, "value1");
    //     cache.insert(2, "value2");

    //     match cache.get(&1) {
    //         Some(v) => assert_eq!("value1", v),
    //         None => assert!(false),
    //     }

    //     cache.insert(3, "value3"); // This should evict key2

    //     match cache.get(&3) {
    //         Some(v) => assert_eq!("value3", v),
    //         None => assert!(false),
    //     }

    //     // Verify that key2 is evicted
    //     assert_eq!(2, cache.map.len());
    //     assert!(!cache.map.contains_key(&2));
    // }
}
