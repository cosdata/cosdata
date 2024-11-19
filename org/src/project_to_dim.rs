// Rationale:: 
// Projection of a 2D vector (x, y) to a 3D point where the z-coordinate encodes the angular distance
// between the vector and the x-axis.
//
// The angle θ between the vector (x, y) and the x-axis is given by:
// θ = arctan(y / x)
//
// To normalize the angle to a value between 0 and 1, we divide by π:
// z = θ / π
//
// This would give a 3D point (0, 0, z) where z encodes the angular distance from the x-axis.
//
// It generates random vectors A and B, and creates perturbed versions of A.
// It projects these vectors to the target dimension (16 in this case).
// It calculates cosine similarities between the projected vectors.
// It creates indices for the projected vectors using the make_index function.
// It counts how many perturbed A vectors fall under each index.
// It prints various statistics about the perturbations, similarities, and index counts.

// This creates a normal (Gaussian) distribution with a mean of 0.0 and a standard 
// deviation of 0.1. This distribution will be used to generate random perturbations.
// On average, the perturbed vectors are centered around the original vector A (because the normal distribution is centered at 0).
// Most perturbations are small (about 68% will be within ±10% of the original value, given the standard deviation of 0.1).
// Larger perturbations are possible but less likely, following the bell curve of the normal distribution.

// Note: deviation changed from 0.1 to 0.25 below

use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use std::collections::HashMap;

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot_product / (norm_a * norm_b)
}

fn project_to_3d(x: f64, y: f64) -> f64 {
    let theta = y.atan2(x);
    theta / std::f64::consts::PI
}

fn project_vector_to_x(v: &[f64], x: usize) -> Vec<f64> {
    let mut v = v.to_vec();
    while v.len() > x {
        if v.len() / 2 < x {
            let pairs_to_project = v.len() - x;
            let mut projected: Vec<f64> = (0..pairs_to_project)
                .map(|i| project_to_3d(v[2*i], v[2*i+1]))
                .collect();
            projected.extend_from_slice(&v[pairs_to_project*2..]);
            v = projected;
        } else {
            v = v.chunks(2)
                .map(|chunk| project_to_3d(chunk[0], chunk[1]))
                .collect();
        }
    }
    v
}

fn make_index(v: &[f64]) -> (u16, Vec<u16>) {
    let mut main_index = 0;
    let mut main_mask = 0;
    let mut sway_bits = Vec::new();
    
    for (i, &x) in v.iter().enumerate() {
        if x >= 0.0 {
            main_index |= 1 << i;
        }
        if x >= 0.1 {
            main_mask |= 1 << i;
        } else if x > -0.1 && x < 0.1 {
            sway_bits.push(i as u16);
        }
    }
    
    let mut alt_indices = vec![main_mask];
    for i in 1..2u16.pow(sway_bits.len() as u32) {
        let mut alt_index = main_mask;
        for (j, &bit) in sway_bits.iter().enumerate() {
            if (i & (1 << j)) != 0 {
                alt_index |= 1 << bit;
            }
        }
        alt_indices.push(alt_index);
    }
    
    (main_index, alt_indices)
}

fn standard_deviation(data: &[f64]) -> f64 {
    let mut mean = 0.0;
    let mut sum_squares = 0.0;
    let mut count = 0.0; // Use f64 directly for count to avoid casting

    for &x in data {
        count += 1.0;
        let delta = x - mean;
        mean += delta / count;
        sum_squares += delta * (x - mean);
    }

    // Return 0.0 if data is empty or has only one element
    if count < 2.0 {
        return 0.0;
    }

    // Calculate and return sample standard deviation
    (sum_squares / (count - 1.0)).sqrt()
}


fn main() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);

    // Create two random source vectors of 320 dimensions
    let a: Vec<f64> = (0..320).map(|_| uniform.sample(&mut rng)).collect();
    let b: Vec<f64> = (0..320).map(|_| uniform.sample(&mut rng)).collect();

    // Calculate and print cosine similarity between original A and B
    let similarity_ab_original = cosine_similarity(&a, &b);
    println!("Cosine similarity between original A and B: {:.6}", similarity_ab_original);

    // Number of samples and target dimensions
    let num_vectors = 100;
    let target_dimensions = 16;

    // Generate perturbed versions of A
    let normal = Normal::new(0.0, 0.25).unwrap();
    let perturbed_vectors: Vec<Vec<f64>> = (0..num_vectors)
        .map(|_| a.iter().map(|&x| x * (1.0 + normal.sample(&mut rng))).collect())
        .collect();

    // Print summary of perturbations
    println!("Summary of perturbations:");
    for (i, perturbed) in perturbed_vectors.iter().take(5).enumerate() {
        let diff: Vec<f64> = perturbed.iter().zip(a.iter()).map(|(p, a)| p - a).collect();
        println!("\nPerturbed vector {}:", i + 1);
        println!("  Mean difference: {:.6}", diff.iter().sum::<f64>() / diff.len() as f64);
        println!("  Max difference:  {:.6}", diff.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        println!("  Min difference:  {:.6}", diff.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
    }

    // Project source A and B
    let a_source_projected = project_vector_to_x(&a, target_dimensions);
    let b_projected = project_vector_to_x(&b, target_dimensions);

    // Calculate and print cosine similarity between projected A and B
    let similarity_ab_source = cosine_similarity(&a_source_projected, &b_projected);
    println!("\nProjected A: {:?}", a_source_projected);
    println!("\nProjected B: {:?}", b_projected);
    println!("\nCosine similarity between projected A and B: {:.6}", similarity_ab_source);

    // Calculate and print indices for projected A and B
    let (index_a, alt_indices_a) = make_index(&a_source_projected);
    let (index_b, alt_indices_b) = make_index(&b_projected);
    println!("Index for projected A: {}", index_a);
    println!("Alternative indices for A: {:?}", alt_indices_a);
    println!("Index for projected B: {}", index_b);
    println!("Alternative indices for B: {:?}", alt_indices_b);

    // Main loop
    let mut similarities_ab = Vec::new();
    let mut similarities_aa = Vec::new();
    let mut main_index_counts = HashMap::new();
    let mut alt_index_counts = HashMap::new();

    for a_perturbed in &perturbed_vectors {
        let a_perturbed_projected = project_vector_to_x(a_perturbed, target_dimensions);
        
        let similarity_ab = cosine_similarity(&a_perturbed_projected, &b_projected);
        let similarity_aa = cosine_similarity(&a_perturbed_projected, &a_source_projected);
        
        similarities_ab.push(similarity_ab);
        similarities_aa.push(similarity_aa);

        let (main_index, alt_indices) = make_index(&a_perturbed_projected);
        *main_index_counts.entry(main_index).or_insert(0) += 1;
        for &alt_index in &alt_indices {
            *alt_index_counts.entry(alt_index).or_insert(0) += 1;
        }
    }

    // Print results
    println!("\nCosine similarity between projected perturbed A and projected B:");
    println!("Average: {:.6}", similarities_ab.iter().sum::<f64>() / similarities_ab.len() as f64);
    println!("Standard deviation: {:.6}", standard_deviation(&similarities_ab));
    println!("Min: {:.6}", similarities_ab.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
    println!("Max: {:.6}", similarities_ab.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    println!("\nCosine similarity between projected perturbed A and projected source A:");
    println!("Average: {:.6}", similarities_aa.iter().sum::<f64>() / similarities_aa.len() as f64);
    println!("Standard deviation: {:.6}", standard_deviation(&similarities_aa));
    println!("Min: {:.6}", similarities_aa.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
    println!("Max: {:.6}", similarities_aa.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    // Calculate and print perturbation sizes
    let original_norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let perturbation_sizes: Vec<f64> = perturbed_vectors
        .iter()
        .map(|perturbed| {
            let diff: Vec<f64> = perturbed.iter().zip(a.iter()).map(|(p, a)| p - a).collect();
            diff.iter().map(|x| x * x).sum::<f64>().sqrt() / original_norm
        })
        .collect();

    println!("\nPerturbation analysis:");
    println!("Average relative perturbation size: {:.6}", perturbation_sizes.iter().sum::<f64>() / perturbation_sizes.len() as f64);
    println!("Max relative perturbation size: {:.6}", perturbation_sizes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    println!("Min relative perturbation size: {:.6}", perturbation_sizes.iter().fold(f64::INFINITY, |a, &b| a.min(b)));

    // Print main index counts
    println!("\nMain index counts for perturbed A vectors:");
    let mut sorted_main_counts: Vec<_> = main_index_counts.into_iter().collect();
    sorted_main_counts.sort_by_key(|&(k, _)| k);
    for (index, count) in sorted_main_counts {
        println!("Index {}: {} occurrences", index, count);
    }

    // Print alternative index counts
    println!("\nAlternative index counts for perturbed A vectors:");
    let mut sorted_alt_counts: Vec<_> = alt_index_counts.into_iter().collect();
    sorted_alt_counts.sort_by_key(|&(k, _)| k);
    for (index, count) in sorted_alt_counts {
        println!("Index {}: {} occurrences", index, count);
    }
}
