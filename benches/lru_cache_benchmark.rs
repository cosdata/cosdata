use cosdata::models::lru_cache::{EvictStrategy, LRUCache, ProbEviction};
use criterion::{criterion_group, criterion_main, Criterion};
use half::f16;
use rand::Rng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru cache");

    let cache1: LRUCache<u64, u64> = LRUCache::new(10000, EvictStrategy::Immediate);

    let evict_strategy =
        EvictStrategy::Probabilistic(ProbEviction::new(f16::from_f32_const(0.03125)));
    let cache2: LRUCache<u64, u64> = LRUCache::new(10000, evict_strategy);

    let mut rng = rand::thread_rng();

    group.bench_function("immediate eviction", |b| {
        b.iter(|| {
            let x = rng.gen_range(u64::MIN..u64::MAX);
            cache1.get_or_insert(x, || Ok::<u64, Box<dyn std::error::Error>>(x))
        })
    });

    group.bench_function("probabilistic eviction", |b| {
        b.iter(|| {
            let x = rng.gen_range(u64::MIN..u64::MAX);
            cache2.get_or_insert(x, || Ok::<u64, Box<dyn std::error::Error>>(x))
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
