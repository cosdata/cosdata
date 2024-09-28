use rand::Rng;
// Define the lookup table size
const U16_TABLE_SIZE: usize = u16::MAX as usize + 1;

// Create a static lookup table for u16 values
static mut U16_LOOKUP_TABLE: [u32; U16_TABLE_SIZE] = [0; U16_TABLE_SIZE];

#[derive(Debug, Clone)]
pub struct VectorQt {
    pub quant_vec: Vec<Vec<u32>>,
    pub magnitude: usize,
    pub resolution: u8,
}

// Function to initialize the lookup table for u16 values
pub fn initialize_u16_lookup_table() {
    for i in 0..U16_TABLE_SIZE {
        unsafe {
            U16_LOOKUP_TABLE[i] = shift_and_accumulate_u16(i as u16);
        }
    }
}

// Helper function to compute shift_and_accumulate for u16 values
fn shift_and_accumulate_u16(value: u16) -> u32 {
    let mut result: u32 = 0;
    result += x_function(15 & (value as u32 >> 0));
    result += x_function(15 & (value as u32 >> 4));
    result += x_function(15 & (value as u32 >> 8));
    result += x_function(15 & (value as u32 >> 12));
    result
}

// x_function remains the same
pub fn x_function(value: u32) -> u32 {
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

pub fn shift_and_accumulate(value: u32) -> u32 {
    let high = (value >> 16) as u16;
    let low = (value & 0xFFFF) as u16;
    unsafe { U16_LOOKUP_TABLE[high as usize] + U16_LOOKUP_TABLE[low as usize] }
}

fn main() {
    // Initialize the lookup table once
    initialize_u16_lookup_table();
    println!("done");

    let size = 64;
    let min = 0.0;
    let max = 1.0;
    let vec1 = (0..size)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let random_number: f32 = rng.gen_range(min..max);
            random_number
        })
        .collect::<Vec<f32>>();

    let vec2 = (0..size)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let random_number: f32 = rng.gen_range(min..max);
            random_number
        })
        .collect::<Vec<f32>>();

    println!("raw vec A :{:?}", vec1);
    println!("raw vec B :{:?}", vec2);

    let resolution = 1 as u8;
    let quantized_values1: Vec<Vec<u32>> = quantize_to_u32_bits(&vec1.clone(), resolution);
    let quantized_values2: Vec<Vec<u32>> = quantize_to_u32_bits(&vec2.clone(), resolution);

    let vector_list1 = VectorQt {
        quant_vec: quantized_values1,
        resolution: resolution,
    };
    let vector_list2 = VectorQt {
        quant_vec: quantized_values2,
        resolution: resolution,
    };
    println!("quantized vec A :{:?}", vector_list1);
    println!("quantized vec B :{:?}", vector_list2);

    let normal = cosine_similarity(&vec1, &vec2);
    println!("cs <normal>: {}", normal);
    println!("\n\n");

    let scalar_quant_cs = cosine_similarity_new(&vector_list1, &vector_list2);
    println!("scalar_quant_cs : {}", scalar_quant_cs);
}

fn precompute_lookup_table() -> [[u8; 16]; 16] {
    let mut table = [[0; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            table[i][j] = (i * j) as u8;
        }
    }
    table
}

fn multiply_quantized_vectors(a: &[u8], b: &[u8], lookup_table: &[[u8; 16]; 16]) -> Vec<u8> {
    let len = a.len().min(b.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let a_even = (a[i] >> 4) & 0x0F;
        let a_odd = a[i] & 0x0F;
        let b_even = (b[i] >> 4) & 0x0F;
        let b_odd = b[i] & 0x0F;

        let prod_even = lookup_table[a_even as usize][b_even as usize];
        let prod_odd = lookup_table[a_odd as usize][b_odd as usize];

        result.push((prod_even << 4) | prod_odd);
    }

    result
}

fn quaternary_multiply_u8(a0: u8, a1: u8, b0: u8, b1: u8) -> u16 {
    // Calculate intermediate products
    let p0 = a0 & b0; // a0 * b0
    let p1 = (a0 & b1) ^ (a1 & b0); // (a0 * b1) ^ (a1 * b0)
    let p2 = a1 & b1; // a1 * b1

    // Combine intermediate products to form the final result
    let result = (p2 << 2) | (p1 << 1) | p0;
    result
}

fn senary_multiply_u8(a0: u8, a1: u8, a2: u8, b0: u8, b1: u8, b2: u8) -> u16 {
    // Calculate intermediate products
    let p0 = a0 & b0;
    let p1 = (a0 & b1) ^ (a1 & b0);
    let p2 = (a0 & b2) ^ (a1 & b1) ^ (a2 & b0);
    let p3 = (a1 & b2) ^ (a2 & b1);
    let p4 = a2 & b2;

    // Combine intermediate products to form the final result
    let result = (p4 << 4) | (p3 << 3) | (p2 << 2) | (p1 << 1) | p0;
    result
}

pub fn cosine_coalesce(x: &VectorQt, y: &VectorQt, length: usize) -> f32 {
    let parts = 2_usize.pow(x.resolution as u32);
    let mut dot_product: usize = 0;
    let quant_len = length >> 5;
    for index in 0..parts {
        let mut sum = 0;
        for jj in 0..quant_len {
            let x_item = x.quant_vec[index][jj];
            let y_item = y.quant_vec[index][jj];
            let and_result = x_item & y_item;
            println!(
                "x {} {:032b} | y {} {:032b} | xor {:032b}",
                x_item, x_item, y_item, y_item, and_result
            );
            sum += shift_and_accumulate(and_result) as usize;
            println!("sum cumulative: {}", sum);
        }
        dot_product += sum;
    }
    let final_result = dot_product / x.magnitude * y.magnitude;
    final_result
}

fn cosine_similarity_new(x: &VectorQt, y: &VectorQt) -> f32 {
    let and_val = 0.12;
    let or_val = 0.12;
    let xor_val = 1.0;

    let vec1 = &x.quant_vec;
    let vec2 = &y.quant_vec;
    let vec1_len = vec1.len();

    let mut dot_product: f32 = 0.0;
    let mut dot_product_and_count: i32 = 0; // can even have a vec for each level b/w MSB and LSB.
    let mut dot_product_or_count: i32 = 0;
    //let mut dot_product_xor_count: i32 = 0;

    for index in 0..vec1_len {
        let inner_product_len = vec1[0].len();
        for i in 0..inner_product_len {
            dot_product_and_count +=
                ((shift_and_accumulate(vec1[index][i] & vec2[index][i])) << index) as i32 - 16;
            //dot_product_or_count += ((shift_and_accumulate(vec1[index][i] | vec2[index][i])) ) as i32 - 16;
            //dot_product_xor_count += shift_and_accumulate(vec1[index][i] ^ vec2[index][i]) as i32 - 16;
            println!(
                "debug : and {} | or {} | {} {}",
                dot_product_and_count, dot_product_or_count, vec1[index][i], vec2[index][i]
            );
        }
    }
    dot_product = (or_val * dot_product_or_count as f32) + (and_val * dot_product_and_count as f32);
    // dot_product = and_val * dot_product_and_count as f32;

    let mut premag1: f32 = 0.0;
    for (_index, vec) in vec1.iter().enumerate() {
        premag1 += vec
            .iter()
            .enumerate()
            .map(|(_, a)| {
                (or_val * shift_and_accumulate(a | a) as f32)
                    + (and_val * shift_and_accumulate(a & a) as f32)
            })
            .sum::<f32>();

        println!("premag1 : {} {:?}", premag1, vec);
    }

    let mut premag2: f32 = 0.0;
    for (_index, vec) in vec2.iter().enumerate() {
        premag2 += vec
            .iter()
            .enumerate()
            .map(|(_, a)| {
                (or_val * shift_and_accumulate(a | a) as f32)
                    + (and_val * shift_and_accumulate(a & a) as f32)
            })
            .sum::<f32>();
        println!("premag2 : {} {:?}", premag2, vec);
    }

    let magnitude_vec1: f32 = premag1.sqrt();
    let magnitude_vec2: f32 = premag2.sqrt();

    println!("mag new : {} {}", magnitude_vec1, magnitude_vec2);
    println!("dot prod new : {}", dot_product);

    if magnitude_vec1 == 0.0 || magnitude_vec2 == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_vec1 * magnitude_vec2)
}

fn quantize_to_u8(vec: &[f32], u32: length) -> Result<Vec<u8>, QuantizationError> {
    let mut quantized_vec = Vec::with_capacity(length);
    let (out_of_range, has_negative) = v.iter().fold((false, false), |(oor, neg), &x| {
        (oor || x > 1.0 || x < -1.0, neg || x < 0.0)
    });
    if out_of_range {
        return Err(QuantizationError(String::from(
            "values sent in vector for quantization are out of range [-1,+1]",
        )));
    }

    let quantized_vec: Vec<u8> = v
        .iter()
        .map(|&x| {
            let y = if has_negative { x + 1.0 } else { x };
            (y * 255.0).round() as u8
        })
        .collect();

    Ok(quantized_vec)
}

fn quantize_and_combine(vec: &[f32], u32: length) -> Vec<u8> {
    let mut result = Vec::with_capacity(length >> 1);

    for i in (0..len).step_by(2) {
        let even = ((vec[i] * 15.0).round() as u8) << 4;
        let odd = ((vec[i + 1] * 15.0).round() as u8);
        result.push(even | odd);
    }

    result
}

fn to_float_flag(x: f32, bits_per_value: usize, step: f32) -> Vec<bool> {
    let mut num = ((x + 1.0) / step).floor() as usize;
    println!(
        "bits_per_value : {} | step {} | x {} | num {}",
        bits_per_value, step, x, num
    );

    let mut result = vec![];
    for i in (0..bits_per_value).rev() {
        let least_significant_bit = num & 1 == 1;
        result.push(least_significant_bit);
        num >>= 1;
    }
    result.reverse();
    result
}

pub fn quantize_to_u32_bits(fins: &[f32], resolution: u8) -> Vec<Vec<u32>> {
    let bits_per_value = resolution as usize;
    let parts = 2_usize.pow(bits_per_value as u32);
    let step = 2.0 / parts as f32;

    let u32s_per_value = fins.len() / 32;
    let mut quantized: Vec<Vec<u32>> = vec![Vec::with_capacity(u32s_per_value); bits_per_value];

    let mut current_u32s: Vec<u32> = vec![0; bits_per_value];
    let mut bit_index: usize = 0;

    for &f in fins {
        let flags = to_float_flag(f, bits_per_value, step);

        for bit_position in 0..bits_per_value {
            if flags[bit_position] {
                current_u32s[bit_position] |= 1 << bit_index;
            }
        }
        bit_index += 1;

        if bit_index == 32 {
            for bit_position in 0..bits_per_value {
                println!(
                    "{:032b}, {} ",
                    current_u32s[bit_position], current_u32s[bit_position]
                );
                quantized[bit_position].push(current_u32s[bit_position]);
                current_u32s[bit_position] = 0;
            }
            bit_index = 0;
        }
    }

    if bit_index > 0 {
        for bit_position in 0..bits_per_value {
            quantized[bit_position].push(current_u32s[bit_position]);
        }
    }

    quantized
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(vec: &[f32]) -> f32 {
    vec.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dp = dot_product(a, b);
    let maga = magnitude(a);
    let magb = magnitude(b);
    println!("dot product : {} | mag_a {} | mag_b {}", dp, maga, magb);
    dp / (maga * magb)
}
