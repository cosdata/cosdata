pub mod entity;
pub mod relationship;

use nom::{
    character::complete::char,
    combinator::map,
    multi::{separated_list0, separated_list1},
    sequence::{delimited, tuple},
    IResult,
};

use super::{
    common::{parse_identifier, ws},
    value::parse_value,
    Value,
};
pub use entity::EntityInsertion;
pub use relationship::RelationshipInsertion;

pub type Attributes = Vec<Attribute>;

#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub value: Value,
}

pub fn parse_attributes0(input: &str) -> IResult<&str, Attributes> {
    delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_attribute),
        ws(char(')')),
    )(input)
}

pub fn parse_attributes1(input: &str) -> IResult<&str, Attributes> {
    delimited(
        ws(char('(')),
        separated_list1(ws(char(',')), parse_attribute),
        ws(char(')')),
    )(input)
}

pub fn parse_attribute(input: &str) -> IResult<&str, Attribute> {
    map(
        tuple((ws(parse_identifier), ws(char(':')), ws(parse_value))),
        |(name, _, value)| Attribute {
            name: name.to_string(),
            value,
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use crate::cosql::Date;

    use super::*;

    #[test]
    fn test_attribute_parser() {
        let values = [
            (
                r#"name: "The Rust Dev""#,
                Attribute {
                    name: "name".to_string(),
                    value: Value::String("The Rust Dev".to_string()),
                },
            ),
            (
                "age: 54",
                Attribute {
                    name: "age".to_string(),
                    value: Value::Int(54),
                },
            ),
            (
                "date_of_birth: 01-01-1970",
                Attribute {
                    name: "date_of_birth".to_string(),
                    value: Value::Date(Date(1, 1, 1970)),
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_attribute(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_attributes_parser() {
        let values = [
            (
                r#"(
                    name: "The Rust Dev",
                    age: 54,
                    date_of_birth: 01-01-1970
                )"#,
                vec![
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
            ),
            (
                r#"(
                    name: "A Rust Project",
                    start_date: 01-01-2000,
                    end_date: 31-12-2009 
                )"#,
                vec![
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
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_attributes1(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
