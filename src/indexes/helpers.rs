// pre-compute powers of 4
const POWERS_OF_4: [u32; 16] = [
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
    268435456, 1073741824,
];
pub(crate) fn generate_power_of_4_list(valx: u32) -> Vec<(u32, u32)> {
    fn largest_power_of_4_exponent(n: u32) -> u32 {
        // binary search to get the exponent
        match POWERS_OF_4.binary_search_by(|&x| x.cmp(&n)) {
            Ok(index) => index as u32,
            Err(index) => {
                if index == 0 {
                    0
                } else {
                    (index - 1) as u32
                }
            }
        }
    }

    let mut result = Vec::new();
    let mut current = 0;

    while current < valx {
        let exponent = largest_power_of_4_exponent(valx - current);
        let delta = 4u32.pow(exponent);
        result.push((current, exponent));
        current += delta;
    }

    if current != valx {
        result.push((current, 0)); // Add the final step to reach the exact dimension index
    }

    result
}
