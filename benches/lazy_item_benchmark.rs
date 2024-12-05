use cosdata::models::identity_collections::IdentityMapKey;
use cosdata::models::{
    lazy_load::{LazyItem, LazyItemMap, LazyItemVec},
    versioning::Hash,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;

fn create_lazy_item_vec_list(index_count: u32, lazy_vec_count: u32) -> Vec<LazyItemVec<String>> {
    let mut res: Vec<LazyItemVec<String>> = Vec::new();
    for _ in 0..lazy_vec_count {
        let lazy_item_vec: LazyItemVec<String> = LazyItemVec::new();
        for i in 0..index_count {
            let version_id = Hash::from(i);
            let item = LazyItem::new(version_id, i as u16, i.to_string());
            lazy_item_vec.insert(i as usize, item);
        }
        res.push(lazy_item_vec);
    }
    res
}

fn create_lazy_item_map_list(index_count: u32, lazy_map_count: u32) -> Vec<LazyItemMap<String>> {
    let mut res: Vec<LazyItemMap<String>> = Vec::new();
    for _ in 0..lazy_map_count {
        let lazy_item_map: LazyItemMap<String> = LazyItemMap::new();
        for i in 0..index_count {
            let version_id = Hash::from(i);
            let item = LazyItem::new(version_id, i as u16, i.to_string());
            lazy_item_map.insert(IdentityMapKey::Int(i), item);
        }
        res.push(lazy_item_map);
    }
    res
}

fn get_biased_random_index() -> usize {
    let mut rng = rand::thread_rng();
    if rng.gen_bool(0.75) {
        // 75% chance to pick from 0..8
        rng.gen_range(0..8)
    } else {
        // 25% chance to pick from 8..16
        rng.gen_range(8..16)
    }
}

fn lazy_item_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lazy item");
    group.sample_size(10);

    let list_lazy_vec = create_lazy_item_vec_list(16, 1000);
    let list_lazy_map = create_lazy_item_map_list(16, 1000);

    //Retrieving random value between (0..16) from each Lazy vector from a List of 1000 LazyItemVec, with 75% probability to pick from (0..8)
    group.bench_function(BenchmarkId::new("LazyItemVec_Retrieval", 1000), |b| {
        b.iter(|| {
            for lazy_vec in list_lazy_vec.iter() {
                lazy_vec.get(get_biased_random_index());
            }
        });
    });

    //Retrieving random value between (0..16) from each Lazy map from a List of 1000 LazyItemMap, with 75% probability to pick from (0..8)
    group.bench_function(BenchmarkId::new("LazyItemMap_Retrieval", 1000), |b| {
        b.iter(|| {
            for lazy_map in list_lazy_map.iter() {
                let index = get_biased_random_index();
                lazy_map.get(&IdentityMapKey::Int(index as u32));
            }
        });
    });

    let index_count = 16;
    //Inserting index value of (0..16) as String in each Lazy_vector and then adding them to create a list of 1000 Lazy_vectors.
    group.bench_function(BenchmarkId::new("LazyItemVec_Insertion", 1000), |b| {
        b.iter(|| {
            let mut lazy_vec_list: Vec<LazyItemVec<String>> = Vec::new();
            for _ in 0..1000 {
                let lazy_vec: LazyItemVec<String> = LazyItemVec::new();
                for i in 0..index_count {
                    let version_id = Hash::from(i);
                    let item = LazyItem::new(version_id, i as u16, i.to_string());
                    lazy_vec.insert(i as usize, item);
                }
                lazy_vec_list.push(lazy_vec);
            }
        });
    });

    //Inserting index value of (0..16) as String in each Lazy_item_map and then adding them to create a list of 1000 Lazy_map.
    group.bench_function(BenchmarkId::new("LazyItemMap_Insertion", 1000), |b| {
        b.iter(|| {
            let mut lazy_map_list: Vec<LazyItemMap<String>> = Vec::new();
            for _ in 0..1000 {
                let lazy_map: LazyItemMap<String> = LazyItemMap::new();
                for i in 0..index_count {
                    let version_id = Hash::from(i);
                    let item = LazyItem::new(version_id, i as u16, i.to_string());
                    lazy_map.insert(IdentityMapKey::Int(i), item);
                }
                lazy_map_list.push(lazy_map);
            }
        });
    });
}

criterion_group!(benches, lazy_item_benchmark);
criterion_main!(benches);
