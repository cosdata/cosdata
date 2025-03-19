use std::fmt;

use serde::de::{self, Visitor};

use super::FieldValue;

pub struct FieldValueVisitor;

impl Visitor<'_> for FieldValueVisitor {
    type Value = FieldValue;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an integer or a string")
    }

    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(FieldValue::Int(value))
    }

    // @TODO: Add i64, u64 etc. as the variants?
    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value >= i32::MIN as i64 && value <= i32::MAX as i64 {
            Ok(FieldValue::Int(value as i32))
        } else {
            Err(E::custom(format!(
                "integer out of range for i32: {}",
                value
            )))
        }
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value <= i32::MAX as u64 {
            Ok(FieldValue::Int(value as i32))
        } else {
            Err(E::custom(format!(
                "integer out of range for i32: {}",
                value
            )))
        }
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(FieldValue::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(FieldValue::String(value))
    }

    // @TODO: May be add boolean as a variant in `FieldValue` enum?
}
