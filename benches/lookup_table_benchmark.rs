use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

fn shift_and_accumulate_u16(value: u16) -> u32 {
    let mut result: u32 = 0;
    result += x_function(15 & (value as u32 >> 0));
    result += x_function(15 & (value as u32 >> 4));
    result += x_function(15 & (value as u32 >> 8));
    result += x_function(15 & (value as u32 >> 12));
    result
}

// x_function remains the same
fn x_function(value: u32) -> u32 {
    match value {
        0 => 0,
        1 => 1,
        2 => 1,
        3 => 2,
        4 => 1,
        5 => 2,
        6 => 2,
        7 => 3,
        8 => 1,
        9 => 2,
        10 => 2,
        11 => 3,
        12 => 2,
        13 => 3,
        14 => 3,
        15 => 4,
        _ => 0, // Invalid input
    }
}

fn bench_lookup_table(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    for _ in 0..5 {
        let num = rng.gen();

        c.bench_function("shift_and_accumulate", |b| {
            b.iter(|| {
                let result = shift_and_accumulate_u16(black_box(num));
                black_box(result)
            })
        });

        c.bench_function("count_ones", |b| {
            b.iter(|| {
                let result = black_box(num).count_ones();
                black_box(result)
            })
        });
    }
}

criterion_group!(benches, bench_lookup_table);
criterion_main!(benches);
