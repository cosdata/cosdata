use crate::grpc::proto;
use crate::metadata::{FieldValue, schema};
use std::collections::HashSet;

// FieldValue conversions
impl From<FieldValue> for proto::FieldValue {
    fn from(value: FieldValue) -> Self {
        let value = match value {
            FieldValue::Int(i) => proto::field_value::Value::IntValue(i),
            FieldValue::String(s) => proto::field_value::Value::StringValue(s),
        };
        proto::FieldValue { value: Some(value) }
    }
}

impl TryFrom<proto::FieldValue> for FieldValue {
    type Error = String;
    fn try_from(value: proto::FieldValue) -> Result<Self, Self::Error> {
        match value.value {
            Some(proto::field_value::Value::IntValue(i)) => Ok(FieldValue::Int(i)),
            Some(proto::field_value::Value::StringValue(s)) => Ok(FieldValue::String(s)),
            None => Err("FieldValue must have a value".to_string()),
        }
    }
}

// MetadataField conversions
impl TryFrom<proto::MetadataField> for schema::MetadataField {
    type Error = crate::metadata::Error;
    fn try_from(field: proto::MetadataField) -> Result<Self, Self::Error> {
        let name = field.name;
        let values: HashSet<FieldValue> = field
            .values
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<HashSet<_>, String>>()
            .map_err(|e| crate::metadata::Error::InvalidFieldValue(e))?;
        schema::MetadataField::new(name, values)
    }
}

impl From<schema::MetadataField> for proto::MetadataField {
    fn from(field: schema::MetadataField) -> Self {
        proto::MetadataField {
            name: field.name,
            values: field.value_index
                .keys()
                .map(|v| proto::FieldValue::from(v.clone()))
                .collect(),
        }
    }
}

// SupportedCondition conversions
impl From<schema::SupportedCondition> for proto::SupportedCondition {
    fn from(condition: schema::SupportedCondition) -> Self {
        match condition {
            schema::SupportedCondition::And(fields) => proto::SupportedCondition {
                op: proto::supported_condition::OperationType::And as i32,
                field_names: fields.into_iter().collect(),
            },
            schema::SupportedCondition::Or(fields) => proto::SupportedCondition {
                op: proto::supported_condition::OperationType::Or as i32,
                field_names: fields.into_iter().collect(),
            },
        }
    }
}

impl TryFrom<proto::SupportedCondition> for schema::SupportedCondition {
    type Error = crate::metadata::Error;
    fn try_from(condition: proto::SupportedCondition) -> Result<Self, Self::Error> {
        let field_names: HashSet<String> = condition.field_names.into_iter().collect();
        match condition.op {
            x if x == proto::supported_condition::OperationType::And as i32 => {
                Ok(schema::SupportedCondition::And(field_names))
            }
            x if x == proto::supported_condition::OperationType::Or as i32 => {
                Ok(schema::SupportedCondition::Or(field_names))
            }
            _ => Err(crate::metadata::Error::InvalidMetadataSchema),
        }
    }
}

// MetadataSchema conversions
impl TryFrom<proto::MetadataSchema> for schema::MetadataSchema {
    type Error = crate::metadata::Error;
    fn try_from(schema: proto::MetadataSchema) -> Result<Self, Self::Error> {
        let fields = schema
            .fields
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<schema::MetadataField>, _>>()?;
        let conditions = schema
            .supported_conditions
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<schema::SupportedCondition>, _>>()?;
        schema::MetadataSchema::new(fields, conditions)
    }
}

impl From<schema::MetadataSchema> for proto::MetadataSchema {
    fn from(schema: schema::MetadataSchema) -> Self {
        proto::MetadataSchema {
            fields: schema.fields
                .into_iter()
                .map(Into::into)
                .collect(),
            supported_conditions: schema.conditions
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_value_conversion() {
        // Test integer conversion
        let int_value = FieldValue::Int(42);
        let proto_value: proto::FieldValue = int_value.clone().into();
        let converted_back: FieldValue = proto_value.try_into().unwrap();
        assert!(matches!(converted_back, FieldValue::Int(42)));

        // Test string conversion
        let string_value = FieldValue::String("test".to_string());
        let proto_value: proto::FieldValue = string_value.clone().into();
        let converted_back: FieldValue = proto_value.try_into().unwrap();
        assert!(matches!(converted_back, FieldValue::String(s) if s == "test"));

        // Test empty value
        let empty_value = proto::FieldValue { value: None };
        assert!(FieldValue::try_from(empty_value).is_err());
    }

    #[test]
    fn test_metadata_field_conversion() {
        let mut values = HashSet::new();
        values.insert(FieldValue::Int(1));
        values.insert(FieldValue::Int(2));

        let field = schema::MetadataField::new(
            "test_field".to_string(),
            values,
        ).unwrap();

        let proto_field: proto::MetadataField = field.clone().into();
        let converted_back: schema::MetadataField = proto_field.try_into().unwrap();

        assert_eq!(converted_back.name, field.name);
        assert_eq!(converted_back.value_index.len(), field.value_index.len());
    }

    #[test]
    fn test_supported_condition_conversion() {
        // Test AND condition
        let mut fields = HashSet::new();
        fields.insert("field1".to_string());
        fields.insert("field2".to_string());

        let condition = schema::SupportedCondition::And(fields.clone());
        let proto_condition: proto::SupportedCondition = condition.into();
        let converted_back: schema::SupportedCondition = proto_condition.try_into().unwrap();

        match converted_back {
            schema::SupportedCondition::And(converted_fields) => {
                assert_eq!(converted_fields, fields);
            },
            _ => panic!("Wrong condition type after conversion"),
        }

        // Test OR condition
        let condition = schema::SupportedCondition::Or(fields.clone());
        let proto_condition: proto::SupportedCondition = condition.into();
        let converted_back: schema::SupportedCondition = proto_condition.try_into().unwrap();

        match converted_back {
            schema::SupportedCondition::Or(converted_fields) => {
                assert_eq!(converted_fields, fields);
            },
            _ => panic!("Wrong condition type after conversion"),
        }
    }

    #[test]
    fn test_metadata_schema_conversion() {
        // Create test fields
        let mut values1 = HashSet::new();
        values1.insert(FieldValue::Int(1));
        values1.insert(FieldValue::Int(2));
        let field1 = schema::MetadataField::new("field1".to_string(), values1).unwrap();

        let mut values2 = HashSet::new();
        values2.insert(FieldValue::String("a".to_string()));
        values2.insert(FieldValue::String("b".to_string()));
        let field2 = schema::MetadataField::new("field2".to_string(), values2).unwrap();

        // Create test conditions
        let mut field_names = HashSet::new();
        field_names.insert("field1".to_string());
        field_names.insert("field2".to_string());
        let condition = schema::SupportedCondition::And(field_names);

        // Create schema
        let schema = schema::MetadataSchema::new(
            vec![field1, field2],
            vec![condition],
        ).unwrap();

        // Convert to proto and back
        let proto_schema: proto::MetadataSchema = schema.clone().into();
        let converted_back: schema::MetadataSchema = proto_schema.try_into().unwrap();

        assert_eq!(converted_back.fields.len(), schema.fields.len());
        assert_eq!(converted_back.conditions.len(), schema.conditions.len());
    }
}
