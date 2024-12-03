use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};

fn create_file(path: &str, size: u64) -> std::io::Result<File> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let mut rng = rand::thread_rng();
    const BUFFER_SIZE: usize = 8192; // 8 KB buffer
    let mut buffer = [0u8; BUFFER_SIZE];

    let full_chunks = size / BUFFER_SIZE as u64;
    let remainder = (size % BUFFER_SIZE as u64) as usize;

    for _ in 0..full_chunks {
        // Use `rng.fill` for slices instead of arrays
        rng.fill(&mut buffer[..]);
        file.write_all(&buffer)?;
    }

    if remainder > 0 {
        // Use `rng.fill` for slices
        rng.fill(&mut buffer[..remainder]);
        file.write_all(&buffer[..remainder])?;
    }

    file.seek(SeekFrom::Start(0))?; // Reset file pointer to the beginning
    Ok(file)
}

fn write_random_bytes(
    file: &mut File,
    iterations: usize,
    write_size: usize,
    file_size: u64,
) -> std::io::Result<()> {
    let mut rng = rand::thread_rng();
    let buffer: Vec<u8> = (0..write_size).map(|_| rng.gen()).collect();

    for _ in 0..iterations {
        let position = rng.gen_range(0..file_size);
        file.seek(SeekFrom::Start(position))?;
        file.write_all(&buffer)?;
    }

    Ok(())
}

fn benchmark_writes(c: &mut Criterion) {
    let file_size = 100 * 1024 * 1024; // 100 MB
    let file_path = "test_file.bin";
    // Create file outside of the benchmark
    let _file = create_file(file_path, file_size).unwrap();

    let mut group = c.benchmark_group("Random Writes");

    group.bench_function("1000 iterations of 128 bytes on 100MB file", |b| {
        b.iter(|| {
            // Reopen the file for each iteration to ensure consistent starting state
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file_path)
                .unwrap();
            write_random_bytes(&mut file, 1000, 128, file_size).unwrap();
        });
    });

    group.bench_function("100 iterations of 1280 bytes on 100MB file", |b| {
        b.iter(|| {
            // Reopen the file for each iteration to ensure consistent starting state
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file_path)
                .unwrap();
            write_random_bytes(&mut file, 100, 1280, file_size).unwrap();
        });
    });
    group.finish();

    // Clean up the test file
    std::fs::remove_file(file_path).unwrap();
}

criterion_group!(benches, benchmark_writes);
criterion_main!(benches);
