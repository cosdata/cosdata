use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;

fn bin_quant(v: &[f32]) -> Vec<u32> {
    v.chunks(32)
        .map(|c| {
            let mut b = 0;
            for (i, &x) in c.iter().enumerate() {
                if x >= 0.5 {
                    b |= 1 << i;
                }
            }
            b
        })
        .collect()
}

fn oct_quant(v: &[f32]) -> Vec<u8> {
    v.chunks(2)
        .map(|c| {
            let mut o = 0;
            for (i, &x) in c.iter().enumerate() {
                o |= (((x * 15.0).round() as u8) & 0b1111) << (4 * (1 - i));
            }
            o
        })
        .collect()
}

fn simp_quant(v: &[f32]) -> Result<Vec<u8>, String> {
    let (out_of_range, has_negative) = v.iter().fold((false, false), |(oor, neg), &x| {
        (oor || x > 1.0 || x < -1.0, neg || x < 0.0)
    });
    if out_of_range {
        return Err(String::from(
            "Values sent in vector for simp_quant are out of range [-1,+1]",
        ));
    }

    let res: Vec<u8> = v
        .iter()
        .map(|&x| {
            let y = if has_negative { x + 1.0 } else { x };
            (y * 255.0).round() as u8
        })
        .collect();
    Ok(res)
}

fn cos_sim_binary(a: &[u32], b: &[u32]) -> f32 {
    let (mut dp, mut na, mut nb) = (0u64, 0u64, 0u64);
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dp += (ai & bi).count_ones() as u64;
        na += ai.count_ones() as u64;
        nb += bi.count_ones() as u64;
    }
    dp as f32 / ((na * nb) as f32).sqrt()
}

fn cos_sim_octal(a: &[u8], b: &[u8]) -> f32 {
    let (mut dp, mut na, mut nb) = (0u64, 0u64, 0u64);
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        for i in 0..2 {
            let m = 0b1111 << (4 * (1 - i));
            let (ap, bp) = ((ai & m) >> (4 * (1 - i)), (bi & m) >> (4 * (1 - i)));
            dp += ap as u64 * bp as u64;
            na += ap as u64 * ap as u64;
            nb += bp as u64 * bp as u64;
        }
    }
    dp as f32 / ((na * nb) as f32).sqrt()
}

fn cos_sim_u8(a: &[u8], b: &[u8]) -> f32 {
    let (mut d, mut na, mut nb) = (0u64, 0u64, 0u64);
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        d += ai as u64 * bi as u64;
        na += ai as u64 * ai as u64;
        nb += bi as u64 * bi as u64;
    }
    d as f32 / ((na * nb) as f32).sqrt()
}

fn generate_random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn benchmark_binary(c: &mut Criterion) {
    let vectors = generate_random_vectors(100, 1000);
    let mut rng = rand::thread_rng();

    c.bench_function("binary_quantization_and_cosine", |b| {
        b.iter(|| {
            let i1 = rng.gen_range(0..100);
            let i2 = rng.gen_range(0..100);
            let v1 = black_box(&vectors[i1]);
            let v2 = black_box(&vectors[i2]);
            let bv1 = bin_quant(v1);
            let bv2 = bin_quant(v2);
            cos_sim_binary(&bv1, &bv2)
        })
    });
}

fn benchmark_octal(c: &mut Criterion) {
    let vectors = generate_random_vectors(100, 1000);
    let mut rng = rand::thread_rng();

    c.bench_function("octal_quantization_and_cosine", |b| {
        b.iter(|| {
            let i1 = rng.gen_range(0..100);
            let i2 = rng.gen_range(0..100);
            let v1 = black_box(&vectors[i1]);
            let v2 = black_box(&vectors[i2]);
            let ov1 = oct_quant(v1);
            let ov2 = oct_quant(v2);
            cos_sim_octal(&ov1, &ov2)
        })
    });
}

fn benchmark_u8(c: &mut Criterion) {
    let vectors = generate_random_vectors(100, 1000);
    let mut rng = rand::thread_rng();

    c.bench_function("u8_quantization_and_cosine", |b| {
        b.iter(|| {
            let i1 = rng.gen_range(0..100);
            let i2 = rng.gen_range(0..100);
            let v1 = black_box(&vectors[i1]);
            let v2 = black_box(&vectors[i2]);
            let sv1 = simp_quant(v1).inspect_err(|x| println!("{:?}", x)).unwrap();
            let sv2 = simp_quant(v2).inspect_err(|x| println!("{:?}", x)).unwrap();
            cos_sim_u8(&sv1, &sv2)
        })
    });
}

criterion_group!(benches, benchmark_binary, benchmark_octal, benchmark_u8);
criterion_main!(benches);
