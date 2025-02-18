use crate::metadata;
use crate::models::collection::{CollectionConfig, DenseVectorOptions, SparseVectorOptions};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub(crate) struct MetadataField {
    pub name: String,
    pub values: Vec<metadata::FieldValue>,
}

impl TryFrom<MetadataField> for metadata::schema::MetadataField {
    type Error = metadata::Error;

    fn try_from(field: MetadataField) -> Result<Self, Self::Error> {
        let name = field.name;
        let values = field.values.into_iter().collect();
        metadata::schema::MetadataField::new(name, values)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ConditionOp {
    And,
    Or,
}

#[derive(Deserialize)]
pub(crate) struct SupportedCondition {
    pub op: ConditionOp,
    pub field_names: Vec<String>,
}

impl TryFrom<SupportedCondition> for metadata::schema::SupportedCondition {
    type Error = metadata::Error;

    fn try_from(cond: SupportedCondition) -> Result<Self, Self::Error> {
        let field_names = cond.field_names.into_iter().collect();
        match cond.op {
            ConditionOp::And => Ok(metadata::schema::SupportedCondition::And(field_names)),
            ConditionOp::Or => Ok(metadata::schema::SupportedCondition::Or(field_names)),
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct MetadataSchemaParam {
    pub fields: Vec<MetadataField>,
    pub supported_conditions: Vec<SupportedCondition>,
}

impl TryFrom<MetadataSchemaParam> for metadata::schema::MetadataSchema {
    type Error = metadata::Error;

    fn try_from(param: MetadataSchemaParam) -> Result<Self, Self::Error> {
        let mut fields = Vec::with_capacity(param.fields.len());
        for f in param.fields {
            fields.push(f.try_into()?);
        }

        let mut conds = Vec::with_capacity(param.supported_conditions.len());
        for c in param.supported_conditions {
            conds.push(c.try_into()?);
        }

        metadata::schema::MetadataSchema::new(fields, conds)
    }
}

#[derive(Deserialize)]
pub(crate) struct CreateCollectionDto {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub metadata_schema: Option<MetadataSchemaParam>, //object (optional)
    pub config: CollectionConfig,
}

#[derive(Serialize)]
pub(crate) struct CreateCollectionDtoResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct GetCollectionsDto {}

#[derive(Serialize)]
pub(crate) struct GetCollectionsResponseDto {
    pub name: String,
    pub description: Option<String>,
}

#[cfg(test)]
mod tests {
    use crate::metadata::FieldValue;

    use super::*;

    #[test]
    fn test_de_metadata_schema_param() {
        let input = "{\"fields\": [{\"name\": \"myfield1\", \"values\": [1, 2, 3]}, {\"name\": \"myfield2\", \"values\": [\"ok\", \"not-ok\"]}], \"supported_conditions\": [{\"op\": \"and\", \"field_names\": [\"myfield1\", \"myfield2\"]}]}";
        let param: MetadataSchemaParam = serde_json::from_str(input).unwrap();

        let field_1 = &param.fields[0];
        assert_eq!("myfield1", field_1.name);
        let expected_vals = vec![1, 2, 3];
        for i in 0..3 {
            let val = &field_1.values[i];
            match val {
                FieldValue::Int(x) => assert_eq!(expected_vals[i], *x),
                FieldValue::String(_) => assert!(false),
            }
        }

        let field_2 = &param.fields[1];
        assert_eq!("myfield2", field_2.name);
        let expected_vals = vec!["ok", "not-ok"];
        for i in 0..2 {
            let val = &field_2.values[i];
            match val {
                FieldValue::Int(_) => assert!(false),
                FieldValue::String(s) => assert_eq!(expected_vals[i], s),
            }
        }

        let cond = &param.supported_conditions[0];
        match cond.op {
            ConditionOp::And => assert!(true),
            ConditionOp::Or => assert!(false),
        }

        assert_eq!(vec!["myfield1", "myfield2"], cond.field_names);
    }
}
