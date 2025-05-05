use std::io;

use chrono::{TimeZone, Utc};

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    collection_transaction::{Progress, Summary, TransactionStatus},
    types::FileOffset,
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
                progress,
                last_updated,
            } => {
                buf.push(1);
                buf.extend_from_slice(&progress.percentage_done.to_le_bytes());
                buf.extend_from_slice(&progress.records_indexed.to_le_bytes());
                buf.extend_from_slice(&progress.total_records.to_le_bytes());
                buf.extend_from_slice(&progress.rate_per_second.to_le_bytes());
                buf.extend_from_slice(&progress.estimated_time_remaining_seconds.to_le_bytes());
                buf.extend_from_slice(&last_updated.timestamp().to_le_bytes());
            }
            TransactionStatus::Complete {
                summary,
                last_updated,
            } => {
                buf.push(2);
                buf.extend_from_slice(&summary.total_records_indexed.to_le_bytes());
                buf.extend_from_slice(&summary.duration_seconds.to_le_bytes());
                buf.extend_from_slice(&summary.average_rate_per_second.to_le_bytes());
                buf.extend_from_slice(&last_updated.timestamp().to_le_bytes());
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
                let percentage_done = bufman.read_f32_with_cursor(cursor)?;
                let records_indexed = bufman.read_u32_with_cursor(cursor)?;
                let total_records = bufman.read_u32_with_cursor(cursor)?;
                let rate_per_second = bufman.read_f32_with_cursor(cursor)?;
                let estimated_time_remaining_seconds = bufman.read_u32_with_cursor(cursor)?;
                let timestamp = bufman.read_i64_with_cursor(cursor)?;
                let last_updated = Utc.timestamp_opt(timestamp, 0).single().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid timestamp")
                })?;

                Ok(Self::InProgress {
                    progress: Progress {
                        percentage_done,
                        records_indexed,
                        total_records,
                        rate_per_second,
                        estimated_time_remaining_seconds,
                    },
                    last_updated,
                })
            }
            2 => {
                let total_records_indexed = bufman.read_u32_with_cursor(cursor)?;
                let duration_seconds = bufman.read_u32_with_cursor(cursor)?;
                let average_rate_per_second = bufman.read_f32_with_cursor(cursor)?;
                let timestamp = bufman.read_i64_with_cursor(cursor)?;
                let last_updated = Utc.timestamp_opt(timestamp, 0).single().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid timestamp")
                })?;

                Ok(Self::Complete {
                    summary: Summary {
                        total_records_indexed,
                        duration_seconds,
                        average_rate_per_second,
                    },
                    last_updated,
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
