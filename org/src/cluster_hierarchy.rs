use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{self, Write};

type Vector = Vec<f64>;

#[derive(Debug)]
struct SearchResult {
    similarity: f64,
    vector: Vector,
}

impl SearchResult {
    fn new(similarity: f64, vector: Vector) -> Self {
        SearchResult { similarity, vector }
    }
}
impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal) // Reversed comparison
    }
}
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for SearchResult {}

struct HierarchicalCluster {
    centroid: Vector,
    primary: Vector,
    children: Vec<HierarchicalCluster>,
    max_size: usize,
    level: usize,
    levels_below: usize, // New field
}

impl HierarchicalCluster {
    fn new(initial_vector: Vector, max_size: usize, level: usize) -> Self {
        HierarchicalCluster {
            centroid: initial_vector.clone(),
            primary: initial_vector,
            children: Vec::new(),
            max_size,
            level,
            levels_below: 0,
        }
    }

    fn calculate_hypothetical_centroid(&self, vector: &Vector) -> Vector {
        let n_dims = self.centroid.len();
        let total_count = self.count_all_nodes() + 1;

        let mut new_centroid = vec![0.0; n_dims];
        self.sum_all_centroids(&mut new_centroid);

        for i in 0..n_dims {
            new_centroid[i] += vector[i];
            new_centroid[i] /= total_count as f64;
        }

        new_centroid
    }

    fn update_centroid(&mut self) {
        let n_dims = self.centroid.len();
        let total_count = self.count_all_nodes();

        let mut new_centroid = vec![0.0; n_dims];
        self.sum_all_centroids(&mut new_centroid);

        for i in 0..n_dims {
            new_centroid[i] /= total_count as f64;
        }

        self.centroid = new_centroid;
    }

    fn sum_all_centroids(&self, sum: &mut Vector) {
        for i in 0..sum.len() {
            sum[i] += self.centroid[i];
        }

        for child in &self.children {
            child.sum_all_centroids(sum);
        }
    }

    fn count_all_nodes(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(|child| child.count_all_nodes())
            .sum::<usize>()
    }

    fn is_full(&self) -> bool {
        self.count_all_nodes() >= self.max_size
    }

    fn insert_vector(&mut self, vector: Vector) {
        if self.children.len() < 10 {
            self.children.push(HierarchicalCluster::new(
                vector,
                self.max_size,
                self.level + 1,
            ));
            self.levels_below = 1; // Update levels_below
            self.update_centroid();
            return;
        }

        let mut best_similarity = -1.0;
        let mut best_child_idx = 0;

        for (idx, child) in self.children.iter().enumerate() {
            let hypothetical_centroid = child.calculate_hypothetical_centroid(&vector);
            let similarity = cosine_similarity(&vector, &hypothetical_centroid);

            if similarity > best_similarity {
                best_similarity = similarity;
                best_child_idx = idx;
            }
        }

        self.children[best_child_idx].insert_vector(vector);
        // Update levels_below to be 1 + maximum levels among all children
        self.levels_below = 1 + self
            .children
            .iter()
            .map(|child| child.levels_below)
            .max()
            .unwrap_or(0);
        self.update_centroid();
    }

    fn analyze_level(&self, target_level: usize, current_level: usize) -> Vec<usize> {
        if current_level == target_level {
            return vec![self.levels_below];
        }

        let mut results = Vec::new();
        for child in &self.children {
            results.extend(child.analyze_level(target_level, current_level + 1));
        }
        results
    }

    fn find_nearest_neighbors(&self, query: &Vector, k: usize) -> Vec<SearchResult> {
        let mut results = BinaryHeap::new();
        self.search_recursive_with_branching(query, k, &mut results);

        let mut final_results = Vec::new();
        while let Some(result) = results.pop() {
            final_results.push(result);
        }
        final_results.reverse();
        final_results
    }

    fn search_recursive_with_branching(
        &self,
        query: &Vector,
        k: usize,
        results: &mut BinaryHeap<SearchResult>,
    ) {
        // Check similarity with this node's primary vector
        let similarity = cosine_similarity(query, &self.primary);
        update_results(
            results,
            SearchResult::new(similarity, self.primary.clone()),
            k,
        );

        if self.children.is_empty() {
            return;
        }

        // Get all children sorted by similarity to query
        let mut child_similarities: Vec<(usize, f64)> = self
            .children
            .iter()
            .enumerate()
            .map(|(idx, child)| (idx, cosine_similarity(query, &child.centroid)))
            .collect();
        child_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        match self.levels_below {
            0 | 1 => {
                // Just check primary vectors of all children
                for child in &self.children {
                    let sim = cosine_similarity(query, &child.primary);
                    update_results(results, SearchResult::new(sim, child.primary.clone()), k);
                }
            }
            2 => {
                // Search all children depth-first
                for child in &self.children {
                    child.search_recursive_with_branching(query, k, results);
                }
            }
            3 => {
                // Take top 4 most similar children and search them
                for (idx, _) in child_similarities.iter().take(4) {
                    self.children[*idx].search_recursive_with_branching(query, k, results);
                }
                // Check primary vectors of remaining children
                for (idx, _) in child_similarities.iter().skip(4) {
                    let child = &self.children[*idx];
                    let sim = cosine_similarity(query, &child.primary);
                    update_results(results, SearchResult::new(sim, child.primary.clone()), k);
                }
            }
            _ => {
                // More than 3 levels below - just take best child and search
                if let Some((best_idx, _)) = child_similarities.first() {
                    self.children[*best_idx].search_recursive_with_branching(query, k, results);
                }
                // Check primary vectors of other children
                for (idx, _) in child_similarities.iter().skip(1) {
                    let child = &self.children[*idx];
                    let sim = cosine_similarity(query, &child.primary);
                    update_results(results, SearchResult::new(sim, child.primary.clone()), k);
                }
            }
        }
    }

    fn find_nearest_neighbors_simple_old(&self, query: &Vector, k: usize) -> Vec<SearchResult> {
        let mut results = BinaryHeap::new();
        self.search_recursive(query, k, &mut results);

        let mut final_results = Vec::new();
        while let Some(result) = results.pop() {
            final_results.push(result);
        }
        final_results.reverse();
        final_results
    }

    fn search_recursive(&self, query: &Vector, k: usize, results: &mut BinaryHeap<SearchResult>) {
        let similarity = cosine_similarity(query, &self.primary);
        update_results(
            results,
            SearchResult::new(similarity, self.primary.clone()),
            k,
        );

        if self.children.is_empty() {
            return;
        }

        let mut best_child_idx = 0;
        let mut best_similarity = -1.0;

        for (idx, child) in self.children.iter().enumerate() {
            let sim = cosine_similarity(query, &child.centroid);
            if sim > best_similarity {
                best_similarity = sim;
                best_child_idx = idx;
            }
        }

        self.children[best_child_idx].search_recursive(query, k, results);

        for (idx, child) in self.children.iter().enumerate() {
            if idx != best_child_idx {
                let sim = cosine_similarity(query, &child.primary);
                update_results(results, SearchResult::new(sim, child.primary.clone()), k);
            }
        }
    }

    fn print_structure(&self, indent: usize) {
        let indent_str = " ".repeat(indent * 2);

        let mut total_sim = 0.0;
        let mut count = 0;
        self.calculate_average_similarity(&self.centroid, &mut total_sim, &mut count);
        let avg_similarity = if count > 0 {
            total_sim / count as f64
        } else {
            1.0
        };

        println!(
            "{}Level {}: {} direct children, avg similarity to parent: {:.4}",
            indent_str,
            self.level,
            self.children.len(),
            avg_similarity
        );

        for child in &self.children {
            child.print_structure(indent + 1);
        }
    }

    fn calculate_average_similarity(
        &self,
        parent_centroid: &Vector,
        total: &mut f64,
        count: &mut usize,
    ) {
        *total += cosine_similarity(&self.centroid, parent_centroid);
        *count += 1;

        for child in &self.children {
            child.calculate_average_similarity(&self.centroid, total, count);
        }
    }
}

fn cosine_similarity(v1: &Vector, v2: &Vector) -> f64 {
    let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let mag1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag1 == 0.0 || mag2 == 0.0 {
        0.0
    } else {
        dot_product / (mag1 * mag2)
    }
}

fn generate_perturbed_vector(base: &Vector, perturbation_scale: f64) -> Vector {
    let mut rng = rand::thread_rng();
    base.iter()
        .map(|&x| x + rng.gen_range(-perturbation_scale..=perturbation_scale))
        .collect()
}

fn generate_test_data(n_base: usize, n_perturbations: usize, n_dims: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    let mut all_vectors = Vec::with_capacity(n_base * (n_perturbations + 1));
    let base_vectors: Vec<Vector> = (0..n_base)
        .map(|_| (0..n_dims).map(|_| rng.gen_range(-1.0..=1.0)).collect())
        .collect();

    for base in base_vectors {
        all_vectors.push(base.clone());
        for _ in 0..n_perturbations {
            let perturbed = generate_perturbed_vector(&base, 0.1);
            all_vectors.push(perturbed);
        }
    }

    all_vectors.shuffle(&mut rng);
    all_vectors
}

fn update_results(results: &mut BinaryHeap<SearchResult>, result: SearchResult, k: usize) {
    if results.len() < k {
        results.push(result);
    } else if let Some(worst) = results.peek() {
        if result.similarity > worst.similarity {
            results.pop();
            results.push(result);
        }
    }
}

fn exhaustive_search(all_vectors: &[Vector], query: &Vector, k: usize) -> Vec<SearchResult> {
    let mut results = BinaryHeap::new();

    for vector in all_vectors {
        let similarity = cosine_similarity(query, vector);
        if results.len() < k {
            results.push(SearchResult::new(similarity, vector.clone()));
        } else if let Some(worst) = results.peek() {
            if similarity > worst.similarity {
                // Changed from < to >
                results.pop();
                results.push(SearchResult::new(similarity, vector.clone()));
            }
        }
    }

    let mut final_results = Vec::new();
    while let Some(result) = results.pop() {
        final_results.push(result);
    }
    // No need to reverse since we reversed the ordering in Ord implementation
    final_results
}
// Add this function somewhere with the other utility functions
fn compute_vector_hash(vector: &Vector) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for val in vector {
        val.to_bits().hash(&mut hasher); // Convert f64 to bits before hashing
    }
    hasher.finish()
}
fn main() {
    let n_base_vectors = 200;
    let n_perturbations = 50;
    let n_dims = 20;
    let cluster_size = 100;

    println!(
        "\nGenerating {} base vectors with {} perturbations each...",
        n_base_vectors, n_perturbations
    );

    let all_vectors = generate_test_data(n_base_vectors, n_perturbations, n_dims);
    let mut root_cluster = HierarchicalCluster::new(all_vectors[0].clone(), cluster_size, 0);

    println!("Building cluster hierarchy...");
    for (i, vector) in all_vectors[1..].iter().enumerate() {
        root_cluster.insert_vector(vector.clone());
        if (i + 1) % 1000 == 0 {
            println!("Processed {} vectors", i + 1);
        }
    }

    loop {
        println!("\n╔════════════════════════════════════════════╗");
        println!("║     Hierarchical Clustering with ANN       ║");
        println!("║                                            ║");
        println!("║   1. Visualize cluster graph              ║");
        println!("║   2. Search with perturbed vector         ║");
        println!("║   3. Analyze level statistics             ║");
        println!("║   4. Exit                                 ║");
        println!("╚════════════════════════════════════════════╝");
        print!("\nEnter choice (1-4): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let choice: usize = input.trim().parse().unwrap_or(0);

        match choice {
            1 => {
                println!("\n╔═══ Cluster Structure ═══╗\n");
                root_cluster.print_structure(0);
            }
            2 => {
                let mut rng = rand::thread_rng();
                let base_idx = rng.gen_range(0..n_base_vectors);
                let base = &all_vectors[base_idx * (n_perturbations + 1)];
                let query = generate_perturbed_vector(base, 0.1);
                let query_hash = compute_vector_hash(&query);

                println!("\nQuery vector hash: {:016x}", query_hash);

                let ann_results = root_cluster.find_nearest_neighbors(&query, 10);
                println!("\n╔═══ ANN Search Results ═══╗\n");
                for (i, result) in ann_results.iter().enumerate() {
                    let hash = compute_vector_hash(&result.vector);
                    println!(
                        "{}. Cosine similarity: {:.4}, Hash: {:016x}",
                        i + 1,
                        result.similarity,
                        hash
                    );
                }

                let mut exact_results = exhaustive_search(&all_vectors, &query, 10);
                exact_results.reverse();

                println!("\n╔═══ Exhaustive Search Results ═══╗\n");
                for (i, result) in exact_results.iter().enumerate() {
                    let hash = compute_vector_hash(&result.vector);
                    println!(
                        "{}. Cosine similarity: {:.4}, Hash: {:016x}",
                        i + 1,
                        result.similarity,
                        hash
                    );
                }

                let mut matches = 0;
                for ann_result in &ann_results {
                    if exact_results.iter().any(|exact| {
                        compute_vector_hash(&exact.vector)
                            == compute_vector_hash(&ann_result.vector)
                    }) {
                        matches += 1;
                    }
                }
                println!("\nANN found {} out of top 10 exact matches", matches);
            }
            3 => {
                print!(
                    "Enter level to analyze (0 to {}): ",
                    root_cluster.levels_below
                );
                io::stdout().flush().unwrap();

                let mut level_input = String::new();
                io::stdin().read_line(&mut level_input).unwrap();
                let target_level: usize = level_input.trim().parse().unwrap_or(0);

                if target_level > root_cluster.levels_below {
                    println!(
                        "Invalid level! Maximum available level is {}",
                        root_cluster.levels_below
                    );
                    continue;
                }

                let stats = root_cluster.analyze_level(target_level, 0);
                println!("\n╔═══ Level {} Statistics ═══╗\n", target_level);

                for (i, levels_below) in stats.iter().enumerate() {
                    println!("Cluster {}: {} levels below", i + 1, levels_below);
                }

                println!("\nSummary:");
                println!("Total clusters at level {}: {}", target_level, stats.len());
                if let Some(max_depth) = stats.iter().max() {
                    println!("Maximum additional depth: {}", max_depth);
                }
            }
            4 => {
                println!("\nExiting...");
                break;
            }
            _ => println!("Invalid choice! Please enter 1-4"),
        }

        println!("\nPress Enter to continue...");
        let mut temp = String::new();
        io::stdin().read_line(&mut temp).unwrap();
    }
}
