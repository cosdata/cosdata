// TODO: possible optimization with `std::mem::transmute`
macro_rules! key {
    (v:$version_id:expr) => {{
        let mut key = Vec::with_capacity(5); // prefix = 1 byte, Hash = 4 byte
        key.push(0);
        key.extend_from_slice(&$version_id.to_le_bytes());
        key
    }};
    (e:$embedding_id:expr) => {{
        let mut prefixed_key = Vec::with_capacity(5); // prefix = 1 bytes, id = 4 byte
        prefixed_key.push(1);
        prefixed_key.extend_from_slice(&$embedding_id.to_le_bytes());
        prefixed_key
    }};
    // misc/metadata
    (m:$name:ident) => {{
        let key = stringify!($name).as_bytes();
        let mut prefixed_key = Vec::with_capacity(1 + key.len()); // prefix = 1 byte
        prefixed_key.push(3);
        prefixed_key.extend_from_slice(&key);
        prefixed_key
    }};
}

pub(crate) use key;
