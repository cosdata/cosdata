pub(crate) fn generate_power_of_4_list(valx: u32) -> Vec<(u32, u32)> {
    fn largest_power_of_4_exponent(n: u32) -> u32 {
        let mut exponent = 0;
        while (4u32.pow(exponent + 1)) <= n {
            exponent += 1;
        }
        exponent
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
