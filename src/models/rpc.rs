use super::{common::WaCustomError, types::{MetricResult, VectorId}};        
use crate::{metadata, models::user::{AddUserResp, AuthResp, Statistics}};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Authenticate {
    username: String,
    password: String,
    pretty_print: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct AddUser {
    username: String,
    api_expiry_time: Option<String>,
    api_quota: Option<i32>,
    first_name: String,
    last_name: String,
    email: String,
    roles: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorANN {
    pub vector_db_name: String,
    pub vector: Vec<f32>,
    pub filter: Option<Filter>,
    pub nn_count: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct BatchVectorANN {
    pub vector_db_name: String,
    pub vectors: Vec<Vec<f32>>,
    pub filter: Option<Filter>,
    pub nn_count: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct FetchNeighbors {
    pub vector_db_name: String,
    pub vector_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpsertVectors {
    pub vector_db_name: String,
    pub vectors: Vec<DenseVector>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct CreateVectorDb {
    pub vector_db_name: String,
    pub dimensions: i32,
    pub max_val: Option<f32>,
    pub min_val: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum RPCResponseBody {
    AuthenticateResp {
        auth: AuthResp,
    },
    RespAddUser {
        add_user: AddUserResp,
    },
    RespUpsertVectors {
        insert_stats: Option<Statistics>,
    },
    RespVectorKNN {
        knn: Vec<(u64, MetricResult)>,
    },
    RespFetchNeighbors {
        vector: DenseVector,
        neighbors: Vec<(u64, MetricResult)>,
    },
    #[serde(untagged)]
    RespCreateVectorDb {
        id: String,
        name: String,
        dimensions: usize,
        min_val: Option<f32>,
        max_val: Option<f32>,
        // created_at: String, // will be added when vector store has a creation timestamp
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenseVector {
    pub id: VectorId,
    pub values: Vec<f32>,
}

// #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
// pub struct VectorList {
//     pub vectors: Vec<Vector>,
// }

pub type Single = MetadataColumnValue;

// @NOTE: Comparison with multiple fields not support for now
// pub type Multiple = Vec<MetadataColumnValue>;

// Define the generic MetadataColumn type
#[allow(clippy::enum_variant_names)]
#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
pub enum MetadataColumnValue {
    StringValue(String),
    IntValue(i32),
    // @NOTE: Float not supported yet
    // FloatValue(f64),
}

impl MetadataColumnValue {
    fn to_fieldvalue(&self) -> metadata::FieldValue {
        match self {
            Self::StringValue(s) => metadata::FieldValue::String(s.to_owned()),
            Self::IntValue(n) => metadata::FieldValue::Int(*n),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum ComparisonOperator {
    #[serde(rename = "$eq")]
    Eq(Single),

    #[serde(rename = "$ne")]
    Ne(Single),

    // @NOTE: Only eq and neq are supported for now

    // #[serde(rename = "$gt")]
    // Gt(Single),

    // #[serde(rename = "$gte")]
    // Gte(Single),

    // #[serde(rename = "$lt")]
    // Lt(Single),

    // #[serde(rename = "$lte")]
    // Lte(Single),

    // #[serde(rename = "$in")]
    // In(Multiple),

    // #[serde(rename = "$nin")]
    // Nin(Multiple),
}

impl ComparisonOperator {
    fn to_predicate(&self, key: &str) -> metadata::Predicate {
        let (op, v) = match self {
            Self::Eq(v) => (metadata::Operator::Equal, v),
            Self::Ne(v) => (metadata::Operator::NotEqual, v),
        };
        metadata::Predicate {
            field_name: key.to_owned(),
            field_value: v.to_fieldvalue(),
            operator: op,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum LogicalOperator {
    #[serde(rename = "$and")]
    And(Vec<Filter>),

    #[serde(rename = "$or")]
    Or(Vec<Filter>),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
pub enum Filter {
    Comparison {
        #[serde(flatten)]
        column: HashMap<String, ComparisonOperator>,
    },
    Logical(LogicalOperator),
}

impl Filter {
    /// Converts the filter in request body to internal
    /// representation. Perhaps the two types can be unified later
    pub fn to_internal(&self) -> Result<metadata::Filter, WaCustomError> {
        let filter_err = |msg: &str| {
            WaCustomError::MetadataError(metadata::Error::UnsupportedFilter(msg.to_string()))
        };
        match self {
            Self::Comparison { column } => {
                if column.len() == 1 {
                    let (key, cop) = column.iter().next().unwrap();
                    let pred = cop.to_predicate(key);
                    Ok(metadata::Filter::Is(pred))
                } else {
                    let mut preds = vec![];
                    for (key, cop) in column.iter() {
                        preds.push(cop.to_predicate(key));
                    }
                    Ok(metadata::Filter::And(preds))
                }
            },
            Self::Logical(LogicalOperator::And(filters)) => {
                let mut preds = vec![];
                for f in filters {
                    match f {
                        Filter::Comparison { column } => {
                            for (key, cop) in column.iter() {
                                preds.push(cop.to_predicate(key));
                            }
                        },
                        // @NOTE: Nested predicates are not
                        // supported.
                        Filter::Logical(_) => return Err(filter_err("nested predicates not supported")),
                    }
                }
                Ok(metadata::Filter::And(preds))
            },
            Self::Logical(LogicalOperator::Or(filters)) => {
                let mut preds = vec![];
                for f in filters {
                    match f {
                        Filter::Comparison { column } => {
                            if column.len() > 1 {
                                // @NOTE: Mixing And and Or predicates
                                // is not supported. Perhaps change
                                // the error type to add a message
                                return Err(filter_err("mixing and, or predicates not supported"))
                            }
                            for (key, cop) in column.iter() {
                                preds.push(cop.to_predicate(key));
                            }
                        },
                        // @NOTE: Nested predicates are not
                        // supported. Perhaps change the error type to
                        // add a message
                        Filter::Logical(_) => return Err(filter_err("nested predicates not supported")),
                    }
                }
                Ok(metadata::Filter::Or(preds))
            },
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::metadata;

    #[test]
    fn test_filter_serde() {
        let input = "{\"foo\":{\"$eq\":\"hello\"},\"bar\":{\"$ne\":1}}";
        let filter: Filter = serde_json::from_str(input).unwrap();
        match filter {
            Filter::Comparison { column } => {
                let foo = column.get("foo").unwrap();
                match foo {
                    ComparisonOperator::Eq(MetadataColumnValue::StringValue(x)) => assert_eq!("hello", x),
                    _ => assert!(false),
                }

                let bar = column.get("bar").unwrap();
                match bar {
                    ComparisonOperator::Ne(MetadataColumnValue::IntValue(x)) => assert_eq!(1, *x),
                    _ => assert!(false),
                }
            },
            Filter::Logical(_) => assert!(false),
        }

        let input = "{\"$and\":[{\"foo\":{\"$eq\":\"abc\"}},{\"bar\":{\"$ne\":\"def\"}}]}";
        let filter: Filter = serde_json::from_str(input).unwrap();
        match filter {
            Filter::Logical(LogicalOperator::And(vec)) => {
                match &vec[0] {
                    Filter::Comparison { column } => {
                        let foo = column.get("foo").unwrap();
                        match foo {
                            ComparisonOperator::Eq(MetadataColumnValue::StringValue(x)) => assert_eq!("abc", x),
                            _ => assert!(false),
                        }
                    },
                    _ => assert!(false),
                }

                match &vec[1] {
                    Filter::Comparison { column } => {
                        let bar = column.get("bar").unwrap();
                        match bar {
                            ComparisonOperator::Ne(MetadataColumnValue::StringValue(x)) => assert_eq!("def", x),
                            _ => assert!(false),
                        }
                    },
                    _ => assert!(false),
                }
            },
            _ => assert!(false),
        }
    }

    #[test]
    fn test_to_internal() {
        // Filter with a single column
        let mut column = HashMap::new();
        column.insert("foo".to_string(), ComparisonOperator::Eq(MetadataColumnValue::StringValue("hello".to_string())));
        let filter = Filter::Comparison { column };
        let internal = filter.to_internal().unwrap();
        match internal {
            metadata::Filter::Is(pred) => {
                assert_eq!("foo", pred.field_name);
                assert_eq!(metadata::FieldValue::String("hello".to_string()), pred.field_value);
                match pred.operator {
                    metadata::Operator::Equal => assert!(true),
                    _ => assert!(false),
                }
            },
            _ => assert!(false),
        }

        // Filter with multiple columns
        let mut column = HashMap::new();
        column.insert("a".to_string(), ComparisonOperator::Eq(MetadataColumnValue::StringValue("hello".to_string())));
        column.insert("b".to_string(), ComparisonOperator::Ne(MetadataColumnValue::IntValue(2)));
        let filter = Filter::Comparison { column };
        let internal = filter.to_internal().unwrap();
        match internal {
            metadata::Filter::And(preds) => {
                assert_eq!(2, preds.len());
                let p1 = preds.iter().find(|p| p.field_name == "a").unwrap();
                assert_eq!("a", p1.field_name);
                assert_eq!(metadata::FieldValue::String("hello".to_string()), p1.field_value);
                match p1.operator {
                    metadata::Operator::Equal => assert!(true),
                    _ => assert!(false),
                }

                let p2 = preds.iter().find(|p| p.field_name == "b").unwrap();
                assert_eq!("b", p2.field_name);
                assert_eq!(metadata::FieldValue::Int(2), p2.field_value);
                match p2.operator {
                    metadata::Operator::NotEqual => assert!(true),
                    _ => assert!(false),
                }
            },
            _ => assert!(false),
        }

        // Filter with Logical::And + single column filters
        let mut c1 = HashMap::new();
        c1.insert("a".to_string(), ComparisonOperator::Eq(MetadataColumnValue::StringValue("hello".to_string())));
        let f1 = Filter::Comparison { column: c1 };
        let mut c2 = HashMap::new();
        c2.insert("b".to_string(), ComparisonOperator::Ne(MetadataColumnValue::IntValue(2)));
        let f2 = Filter::Comparison { column: c2 };
        let filter = Filter::Logical(LogicalOperator::And(vec![f1, f2]));
        let internal = filter.to_internal().unwrap();
        match internal {
            metadata::Filter::And(preds) => {
                assert_eq!(2, preds.len());
                let p1 = preds.iter().find(|p| p.field_name == "a").unwrap();
                assert_eq!("a", p1.field_name);
                assert_eq!(metadata::FieldValue::String("hello".to_string()), p1.field_value);
                match p1.operator {
                    metadata::Operator::Equal => assert!(true),
                    _ => assert!(false),
                }

                let p2 = preds.iter().find(|p| p.field_name == "b").unwrap();
                assert_eq!("b", p2.field_name);
                assert_eq!(metadata::FieldValue::Int(2), p2.field_value);
                match p2.operator {
                    metadata::Operator::NotEqual => assert!(true),
                    _ => assert!(false),
                }
            },
            _ => assert!(false),
        }

        // Filter with Logical::And + Multiple columns filters
        let mut c1 = HashMap::new();
        c1.insert("a".to_string(), ComparisonOperator::Eq(MetadataColumnValue::StringValue("hello".to_string())));
        c1.insert("b".to_string(), ComparisonOperator::Eq(MetadataColumnValue::StringValue("world".to_string())));
        let f1 = Filter::Comparison { column: c1 };
        let mut c2 = HashMap::new();
        c2.insert("c".to_string(), ComparisonOperator::Ne(MetadataColumnValue::IntValue(2)));
        c2.insert("d".to_string(), ComparisonOperator::Eq(MetadataColumnValue::IntValue(10)));
        let f2 = Filter::Comparison { column: c2 };
        let filter = Filter::Logical(LogicalOperator::And(vec![f1, f2]));
        let internal = filter.to_internal().unwrap();
        match internal {
            metadata::Filter::And(preds) => assert_eq!(4, preds.len()),
            _ => assert!(false),
        }

        // Filter with Logical::Or + single column filters
        let mut c1 = HashMap::new();
        c1.insert("a".to_string(), ComparisonOperator::Eq(MetadataColumnValue::StringValue("hello".to_string())));
        let f1 = Filter::Comparison { column: c1 };
        let mut c2 = HashMap::new();
        c2.insert("b".to_string(), ComparisonOperator::Ne(MetadataColumnValue::IntValue(2)));
        let f2 = Filter::Comparison { column: c2 };
        let filter = Filter::Logical(LogicalOperator::Or(vec![f1, f2]));
        let internal = filter.to_internal().unwrap();
        match internal {
            metadata::Filter::Or(preds) => {
                assert_eq!(2, preds.len());
                let p1 = preds.iter().find(|p| p.field_name == "a").unwrap();
                assert_eq!("a", p1.field_name);
                assert_eq!(metadata::FieldValue::String("hello".to_string()), p1.field_value);
                match p1.operator {
                    metadata::Operator::Equal => assert!(true),
                    _ => assert!(false),
                }

                let p2 = preds.iter().find(|p| p.field_name == "b").unwrap();
                assert_eq!("b", p2.field_name);
                assert_eq!(metadata::FieldValue::Int(2), p2.field_value);
                match p2.operator {
                    metadata::Operator::NotEqual => assert!(true),
                    _ => assert!(false),
                }
            },
            _ => assert!(false),
        }

        // Filter with Logical::Or + multiple columns filters (must fail)
        let mut c1 = HashMap::new();
        c1.insert("a".to_string(), ComparisonOperator::Eq(MetadataColumnValue::StringValue("hello".to_string())));
        c1.insert("b".to_string(), ComparisonOperator::Eq(MetadataColumnValue::StringValue("world".to_string())));
        let f1 = Filter::Comparison { column: c1 };
        let mut c2 = HashMap::new();
        c2.insert("c".to_string(), ComparisonOperator::Ne(MetadataColumnValue::IntValue(2)));
        c2.insert("d".to_string(), ComparisonOperator::Eq(MetadataColumnValue::IntValue(10)));
        let f2 = Filter::Comparison { column: c2 };
        let filter = Filter::Logical(LogicalOperator::Or(vec![f1, f2]));
        match filter.to_internal() {
            Err(e) => {
                match e {
                    WaCustomError::MetadataError(metadata::Error::UnsupportedFilter(msg)) => {
                        assert_eq!("mixing and, or predicates not supported", msg)
                    }
                    _ => assert!(false),
                }
            },
            _ => assert!(false),
        }
    }
}
