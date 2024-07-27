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

pub trait CustomSerialize {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32>;
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self>
    where
        Self: Sized;
}

impl<T: CustomSerialize> CustomSerialize for Arc<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        T::serialize(&self, writer)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        T::deserialize(reader, offset).map(Arc::new)
    }
}

impl<T: CustomSerialize> CustomSerialize for Option<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        if let Some(data) = self {
            data.serialize(writer)
        } else {
            Ok(u32::MAX)
        }
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        if offset == u32::MAX {
            return Ok(None);
        }

        T::deserialize(reader, offset).map(Some)
    }
}
