mod lazy_item;
mod lazy_items;
mod neighbour;
mod node;
mod vector;

#[cfg(test)]
mod tests;

use std::io::{Read, Seek, Write};

pub trait CustomSerialize {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32>;
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self>
    where
        Self: Sized;
}
