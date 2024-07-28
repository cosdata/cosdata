mod lazy_item;
mod lazy_items;
mod neighbour;
mod node;
mod vector;

#[cfg(test)]
mod tests;

use std::{
    io::{Read, Seek, Write},
    sync::Arc,
};

use super::cache_loader::NodeRegistry;

pub trait CustomSerialize {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32>;
    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
    ) -> std::io::Result<Self>
    where
        Self: Sized;
}
