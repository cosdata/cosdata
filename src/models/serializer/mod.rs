mod lazy_item;
mod lazy_items;
mod neighbour;
mod node;
mod vector;

#[cfg(test)]
mod tests;

use super::cache_loader::NodeRegistry;
use crate::models::types::FileOffset;
use std::collections::HashSet;
use std::{
    io::{Read, Seek, Write},
    sync::Arc,
};

pub trait CustomSerialize {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32>;
    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<Self>
    where
        Self: Sized;
}
