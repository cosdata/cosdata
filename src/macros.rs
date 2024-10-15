// TODO: possible optimization with `std::mem::transmute`
macro_rules! key {
    (v:$version_id:expr) => {{
        let mut key = Vec::with_capacity(5); // prefix = 1 byte, Hash = 4 byte
        key.push(0);
        key.extend_from_slice(&$version_id.to_le_bytes());
        key
    }};
    (e:$embedding_id:expr) => {{
        let string = $embedding_id.to_string();
        let key: &[u8] = string.as_ref();
        let mut prefixed_key = Vec::with_capacity(key.len() + 1); // prefix = 1 bytes
        prefixed_key.push(1);
        prefixed_key.extend_from_slice(key);
        prefixed_key
    }};
    (b:$branch_id:expr) => {{
        let mut key = Vec::with_capacity(9); // prefix = 1 byte, BranchId = 8 bytes
        key.push(2);
        key.extend_from_slice(&$branch_id.to_le_bytes());
        key
    }};
}

pub(crate) use key;
