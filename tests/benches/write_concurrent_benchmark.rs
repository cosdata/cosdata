use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;
use std::thread;

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
        rng.fill(&mut buffer[..]);
        file.write_all(&buffer)?;
    }

    if remainder > 0 {
        rng.fill(&mut buffer[..remainder]);
        file.write_all(&buffer[..remainder])?;
    }

    file.seek(SeekFrom::Start(0))?; // Reset file pointer to the beginning
    Ok(file)
}

fn write_random_bytes(
    mut file: &File,
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

fn benchmark_concurrent_writes(c: &mut Criterion) {
    let file_size = 10 * 1024 * 1024; // 10 MB
    let num_files = 10;

    // Create files outside of the benchmark
    let file_paths: Vec<String> = (0..num_files)
        .map(|i| format!("test_file{}.bin", i))
        .collect();

    let files: Arc<Vec<File>> = Arc::new(
        file_paths
            .iter()
            .map(|path| create_file(path, file_size).unwrap())
            .collect(),
    );

    let mut group = c.benchmark_group("Concurrent Random Writes");

    group.bench_function(
        "1000 iterations of 128 bytes across 10 files (each 10MB)",
        |b| {
            b.iter(|| {
                let files = Arc::clone(&files);
                let handles: Vec<_> = (0..num_files)
                    .map(|i| {
                        let files = Arc::clone(&files);
                        thread::spawn(move || {
                            write_random_bytes(&files[i], 1000, 128, file_size).unwrap();
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }
            });
        },
    );

    group.bench_function(
        "100 iterations of 1280 bytes across 10 files (each 10MB)",
        |b| {
            b.iter(|| {
                let files = Arc::clone(&files);
                let handles: Vec<_> = (0..num_files)
                    .map(|i| {
                        let files = Arc::clone(&files);
                        thread::spawn(move || {
                            write_random_bytes(&files[i], 100, 1280, file_size).unwrap();
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }
            });
        },
    );

    group.finish();

    // Clean up the test files
    drop(files); // Ensure all file handles are closed before attempting to remove
    for path in file_paths.iter() {
        if let Err(e) = std::fs::remove_file(path) {
            eprintln!("Error removing file: {:?}", e);
        }
    }
}

criterion_group!(benches, benchmark_concurrent_writes);
criterion_main!(benches);
