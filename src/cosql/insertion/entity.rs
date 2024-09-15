use nom::{
    bytes::complete::tag, character::complete::char, combinator::map, sequence::tuple, IResult,
};

use crate::cosql::common::{parse_identifier, parse_variable, ws};

use super::{parse_attributes1, Attributes};

#[derive(Debug, Clone, PartialEq)]
pub struct EntityInsertion {
    pub variable: String,
    pub entity_type: String,
    pub attributes: Attributes,
}

pub fn parse_entity_insertion(input: &str) -> IResult<&str, EntityInsertion> {
    map(
        tuple((
            ws(parse_variable),
            ws(tag("isa")),
            ws(parse_identifier),
            parse_attributes1,
            ws(char(';')),
        )),
        |(variable, _, entity_type, attributes, _)| EntityInsertion {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes,
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{insertion::Attribute, Date, Value};

    #[test]
    fn test_entity_insertion_parser() {
        let values = [
            (
                r#"$rust_dev isa person (
                    name: "The Rust Dev",
                    age: 54,
                    date_of_birth: 01-01-1970
                );"#,
                EntityInsertion {
                    variable: "rust_dev".to_string(),
                    entity_type: "person".to_string(),
                    attributes: vec![
                        Attribute {
                            name: "name".to_string(),
                            value: Value::String("The Rust Dev".to_string()),
                        },
                        Attribute {
                            name: "age".to_string(),
                            value: Value::Int(54),
                        },
                        Attribute {
                            name: "date_of_birth".to_string(),
                            value: Value::Date(Date(1, 1, 1970)),
                        },
                    ],
                },
            ),
            (
                r#"$rust_project isa project (
                    name: "A Rust Project",
                    start_date: 01-01-2000,
                    end_date: 31-12-2009 
                );"#,
                EntityInsertion {
                    variable: "rust_project".to_string(),
                    entity_type: "project".to_string(),
                    attributes: vec![
                        Attribute {
                            name: "name".to_string(),
                            value: Value::String("A Rust Project".to_string()),
                        },
                        Attribute {
                            name: "start_date".to_string(),
                            value: Value::Date(Date(1, 1, 2000)),
                        },
                        Attribute {
                            name: "end_date".to_string(),
                            value: Value::Date(Date(31, 12, 2009)),
                        },
                    ],
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_entity_insertion(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
