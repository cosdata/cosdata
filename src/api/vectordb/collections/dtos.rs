use crate::metadata;
use crate::models::collection::{
    CollectionConfig, DenseVectorOptions, SparseVectorOptions, TFIDFOptions,
};
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};

// Instead of implementing ToSchema for FieldValue, we'll use value_type in the schema attribute

#[derive(Deserialize, ToSchema)]
pub(crate) struct MetadataField {
    pub name: String,
    #[schema(value_type = Vec<String>, example = "[\"value1\", \"value2\", 123]")]
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

#[derive(Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ConditionOp {
    And,
    Or,
}

#[derive(Deserialize, ToSchema)]
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

#[derive(Deserialize, ToSchema)]
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

#[derive(Deserialize, ToSchema)]
pub(crate) struct CreateCollectionDto {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub tf_idf_options: TFIDFOptions,
    pub metadata_schema: Option<MetadataSchemaParam>, //object (optional)
    pub config: CollectionConfig,
    #[serde(default)]
    pub store_raw_text: bool,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct CreateCollectionDtoResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
}

#[derive(Deserialize, ToSchema, IntoParams)]
pub(crate) struct GetCollectionsDto {}

#[derive(Serialize, ToSchema)]
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
        let expected_vals = [1, 2, 3];
        for (i, expected_val) in expected_vals.into_iter().enumerate() {
            let val = &field_1.values[i];
            match val {
                FieldValue::Int(x) => assert_eq!(expected_val, *x),
                FieldValue::String(_) => panic!(),
            }
        }

        let field_2 = &param.fields[1];
        assert_eq!("myfield2", field_2.name);
        let expected_vals = ["ok", "not-ok"];
        for (i, expected_val) in expected_vals.into_iter().enumerate() {
            let val = &field_2.values[i];
            match val {
                FieldValue::Int(_) => panic!(),
                FieldValue::String(s) => assert_eq!(expected_val, s),
            }
        }

        let cond = &param.supported_conditions[0];
        match cond.op {
            ConditionOp::And => {}
            ConditionOp::Or => panic!(),
        }

        assert_eq!(vec!["myfield1", "myfield2"], cond.field_names);
    }
}

#[derive(Serialize, Debug, ToSchema)]
pub(crate) struct CollectionWithVectorCountsDto {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub tf_idf_options: TFIDFOptions,
    pub config: CollectionConfig,
    pub store_raw_text: bool,
    pub vectors_count: u64,
}
