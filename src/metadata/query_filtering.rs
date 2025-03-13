use std::collections::HashMap;

use super::{decimal_to_binary_vec, schema::MetadataSchema, Error, FieldName, FieldValue};

pub enum Operator {
    Equal,
    NotEqual,
}

pub struct Predicate {
    pub field_name: FieldName,
    pub field_value: FieldValue,
    pub operator: Operator,
}

// @NOTE: Nested And/Or not supported for now
pub enum Filter {
    Is(Predicate),
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
}

type QueryFilterDimensions = Vec<i8>;

fn query_filter_encoding(value_id: u16, size: usize, operator: &Operator) -> QueryFilterDimensions {
    decimal_to_binary_vec(value_id, size)
        .iter()
        .map(|x| match operator {
            Operator::Equal => {
                if *x == 1 {
                    1
                } else {
                    -1
                }
            }
            Operator::NotEqual => {
                if *x == 1 {
                    -1
                } else {
                    1
                }
            }
        })
        .collect::<Vec<i8>>()
}

/// Return vector of multiple dimensions from a vector of predicates
///
/// It considers the `AND` combination of all predicates and hence can
/// be used for a single predicate too (vector containing a single
/// predicate)
///
/// This is an internal/private function. See
/// `filter_encoded_dimensions`
fn and_predicates_to_dimensions(
    schema: &MetadataSchema,
    preds: Vec<&Predicate>,
) -> Result<QueryFilterDimensions, Error> {
    let pred_index = preds
        .into_iter()
        .map(|p| (p.field_name.as_ref(), p))
        .collect::<HashMap<&str, &Predicate>>();
    let mut result = vec![];
    for field in &schema.fields {
        match pred_index.get(&field.name.as_ref()) {
            Some(pred) => {
                let value_id = field.value_id(&pred.field_value)?;
                let mut dims =
                    query_filter_encoding(value_id, field.num_dims as usize, &pred.operator);
                result.append(&mut dims);
            }
            None => {
                let mut dims = vec![-1; field.num_dims as usize];
                result.append(&mut dims);
            }
        }
    }
    Ok(result)
}

/// Returns vector of dimensions encoding query filter
///
/// @NOTE(vineet): Here we're assuming that the `Filter` is valid, for
/// the schema. Not sure if the check should happen here or in the
/// calling code
pub fn filter_encoded_dimensions(
    schema: &MetadataSchema,
    filter: &Filter,
) -> Result<Vec<QueryFilterDimensions>, Error> {
    match filter {
        Filter::Is(pred) => {
            let dims = and_predicates_to_dimensions(schema, vec![pred])?;
            Ok(vec![dims])
        }
        Filter::And(preds) => {
            let pred_refs = preds.iter().collect();
            let dims = and_predicates_to_dimensions(schema, pred_refs)?;
            Ok(vec![dims])
        }
        Filter::Or(preds) => {
            let mut result: Vec<QueryFilterDimensions> = vec![];
            for pred in preds {
                let dims = and_predicates_to_dimensions(schema, vec![pred])?;
                result.push(dims);
            }
            Ok(result)
        }
    }
}

// Functionality to be implemented
//
// 1. [✓] Given `MetadataSchema` and metadata filter, return Equality and
//    Inequality filter encoding to be appended to the query vector

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use super::super::schema::{MetadataField, SupportedCondition};
    use super::*;

    #[test]
    fn test_query_filter_encoding() {
        // value 7 represented in 5 dimensions
        let e1 = query_filter_encoding(7, 5, &Operator::Equal);
        assert_eq!(vec![-1, -1, 1, 1, 1], e1);

        let e2 = query_filter_encoding(7, 5, &Operator::NotEqual);
        assert_eq!(vec![1, 1, -1, -1, -1], e2);
    }

    #[test]
    fn test_filter_encoded_dimensions() {
        let age_values: HashSet<FieldValue> = (1..=10).map(|x| FieldValue::Int(x)).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();
        let group_values: HashSet<FieldValue> = vec!["a", "b", "c"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let group = MetadataField::new("group".to_owned(), group_values).unwrap();
        let conditions = vec![
            SupportedCondition::And(
                vec!["age", "group"]
                    .into_iter()
                    .map(|s| String::from(s))
                    .collect(),
            ),
            SupportedCondition::Or(
                vec!["age", "group"]
                    .into_iter()
                    .map(|s| String::from(s))
                    .collect(),
            ),
        ];
        let schema = MetadataSchema::new(vec![age, group], conditions).unwrap();

        // Test for `Is` filter
        let filter = Filter::Is(Predicate {
            field_name: "age".to_string(),
            field_value: FieldValue::Int(6),
            operator: Operator::Equal,
        });
        let qfed = filter_encoded_dimensions(&schema, &filter).unwrap();
        assert_eq!(
            vec![vec![
                -1, 1, 1, -1, // 6 (original value: 6)
                -1, -1
            ]],
            qfed
        );

        // Test for `And` filter
        let filter = Filter::And(vec![
            Predicate {
                field_name: "age".to_string(),
                field_value: FieldValue::Int(2),
                operator: Operator::Equal,
            },
            Predicate {
                field_name: "group".to_string(),
                field_value: FieldValue::String("b".to_owned()),
                operator: Operator::NotEqual,
            },
        ]);
        let qfed = filter_encoded_dimensions(&schema, &filter).unwrap();
        assert_eq!(
            vec![vec![
                -1, -1, 1, -1, // 2 (original value: 2)
                -1, 1 // !2 (original value: !b)
            ]],
            qfed
        );

        // Test for `Or` filter
        let filter = Filter::Or(vec![
            Predicate {
                field_name: "age".to_string(),
                field_value: FieldValue::Int(2),
                operator: Operator::Equal,
            },
            Predicate {
                field_name: "group".to_string(),
                field_value: FieldValue::String("b".to_owned()),
                operator: Operator::NotEqual,
            },
        ]);
        let qfed = filter_encoded_dimensions(&schema, &filter).unwrap();
        assert_eq!(
            vec![
                vec![
                    -1, -1, 1, -1, // 2 (original value: 2)
                    -1, -1
                ],
                vec![
                    -1, -1, -1, -1,
                    -1, 1  // !2 (original value: !b)
                ]
            ],
            qfed
        );
    }
}
