use std::io;

use chrono::{DateTime, TimeZone, Utc};

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    collection_transaction::{ProcessingStats, TransactionStatus},
    types::FileOffset,
    versioning::VersionNumber,
};

use super::SimpleSerialize;

impl SimpleSerialize for TransactionStatus {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let mut buf = Vec::new();

        match self {
            TransactionStatus::NotStarted { last_updated } => {
                buf.push(0);
                buf.extend_from_slice(&last_updated.timestamp().to_le_bytes());
            }
            TransactionStatus::InProgress {
                stats,
                started_at,
                last_updated,
            } => {
                buf.push(1);
                buf.extend_from_slice(&started_at.timestamp().to_le_bytes());
                buf.extend_from_slice(&last_updated.timestamp().to_le_bytes());
                serialize_processing_stats(stats, &mut buf);
            }
            TransactionStatus::Complete {
                stats,
                started_at,
                completed_at,
            } => {
                buf.push(2);
                buf.extend_from_slice(&started_at.timestamp().to_le_bytes());
                buf.extend_from_slice(&completed_at.timestamp().to_le_bytes());
                serialize_processing_stats(stats, &mut buf);
            }
        }

        Ok(bufman.write_to_end_of_file(cursor, &buf)? as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let tag = bufman.read_u8_with_cursor(cursor)?;

        match tag {
            0 => {
                let timestamp = bufman.read_i64_with_cursor(cursor)?;
                let last_updated = Utc.timestamp_opt(timestamp, 0).single().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid timestamp")
                })?;
                Ok(Self::NotStarted { last_updated })
            }
            1 => {
                let started_at_timestamp = bufman.read_i64_with_cursor(cursor)?;
                let last_updated_timestamp = bufman.read_i64_with_cursor(cursor)?;
                let started_at = Utc
                    .timestamp_opt(started_at_timestamp, 0)
                    .single()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid timestamp")
                    })?;
                let last_updated = Utc
                    .timestamp_opt(last_updated_timestamp, 0)
                    .single()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid timestamp")
                    })?;

                let stats = deserialize_processing_stats(bufman, cursor)?;

                Ok(Self::InProgress {
                    stats,
                    started_at,
                    last_updated,
                })
            }
            2 => {
                let started_at_timestamp = bufman.read_i64_with_cursor(cursor)?;
                let started_at = Utc
                    .timestamp_opt(started_at_timestamp, 0)
                    .single()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid timestamp")
                    })?;
                let last_updated_timestamp = bufman.read_i64_with_cursor(cursor)?;
                let last_updated = Utc
                    .timestamp_opt(last_updated_timestamp, 0)
                    .single()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid timestamp")
                    })?;
                let stats = deserialize_processing_stats(bufman, cursor)?;

                Ok(Self::Complete {
                    stats,
                    started_at,
                    completed_at: last_updated,
                })
            }
            tag => Err(BufIoError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid `TransactionStatus` variant `{}`, expected one of `0`, `1`, or `2`",
                    tag,
                ),
            ))),
        }
    }
}

// size: [17, 41]
pub fn serialize_processing_stats(stats: &ProcessingStats, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&stats.records_upserted.to_le_bytes());
    buf.extend_from_slice(&stats.records_deleted.to_le_bytes());
    buf.extend_from_slice(&stats.total_operations.to_le_bytes());
    buf.extend_from_slice(&stats.percentage_complete.to_le_bytes());

    let mut flags_byte = 0;
    let flags_idx = buf.len();
    buf.push(flags_byte);

    if let Some(processing_time_seconds) = stats.processing_time_seconds {
        buf.extend_from_slice(&processing_time_seconds.to_le_bytes());
        flags_byte |= 1;
    }

    if let Some(average_throughput) = stats.average_throughput {
        buf.extend_from_slice(&average_throughput.to_le_bytes());
        flags_byte |= 2;
    }

    if let Some(current_processing_rate) = stats.current_processing_rate {
        buf.extend_from_slice(&current_processing_rate.to_le_bytes());
        flags_byte |= 4;
    }

    if let Some(estimated_completion) = stats.estimated_completion {
        buf.extend_from_slice(&estimated_completion.timestamp().to_le_bytes());
        flags_byte |= 8;
    }

    if let Some(version_created) = stats.version_created {
        buf.extend_from_slice(&version_created.to_le_bytes());
        flags_byte |= 16;
    }

    buf[flags_idx] = flags_byte;
}

pub fn deserialize_processing_stats(
    bufman: &BufferManager,
    cursor: u64,
) -> Result<ProcessingStats, BufIoError> {
    let records_upserted = bufman.read_u32_with_cursor(cursor)?;
    let records_deleted = bufman.read_u32_with_cursor(cursor)?;
    let total_operations = bufman.read_u32_with_cursor(cursor)?;
    let percentage_complete = bufman.read_f32_with_cursor(cursor)?;

    let flags_byte = bufman.read_u8_with_cursor(cursor)?;

    let processing_time_seconds = if flags_byte & 1 != 0 {
        let processing_time_seconds = bufman.read_u32_with_cursor(cursor)?;
        Some(processing_time_seconds)
    } else {
        None
    };

    let average_throughput = if flags_byte & 2 != 0 {
        let average_throughput = bufman.read_f32_with_cursor(cursor)?;
        Some(average_throughput)
    } else {
        None
    };

    let current_processing_rate = if flags_byte & 4 != 0 {
        let current_processing_rate = bufman.read_f32_with_cursor(cursor)?;
        Some(current_processing_rate)
    } else {
        None
    };

    let estimated_completion = if flags_byte & 8 != 0 {
        let estimated_completion_timestamp = bufman.read_i64_with_cursor(cursor)?;
        let estimated_completion = DateTime::from_timestamp(estimated_completion_timestamp, 0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid timestamp"))?;
        Some(estimated_completion)
    } else {
        None
    };

    let version_created = if flags_byte & 16 != 0 {
        let version_created = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
        Some(version_created)
    } else {
        None
    };

    Ok(ProcessingStats {
        records_upserted,
        records_deleted,
        total_operations,
        percentage_complete,
        processing_time_seconds,
        average_throughput,
        current_processing_rate,
        estimated_completion,
        version_created,
    })
}
