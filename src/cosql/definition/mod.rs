pub mod entity;
pub mod relationship;

use nom::{
    character::complete::char,
    combinator::map,
    multi::{separated_list0, separated_list1},
    sequence::tuple,
    IResult,
};

use super::{
    common::{parse_identifier, ws},
    data_type::parse_data_type,
    DataType,
};
pub use entity::EntityDefinition;
pub use relationship::RelationshipDefinition;

pub type AttributeDefinitions = Vec<AttributeDefinition>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeDefinition {
    pub name: String,
    pub data_type: DataType,
}

pub fn parse_attribute_definitions0(input: &str) -> IResult<&str, AttributeDefinitions> {
    separated_list0(ws(char(',')), parse_attribute_definition)(input)
}

pub fn parse_attribute_definitions1(input: &str) -> IResult<&str, AttributeDefinitions> {
    separated_list1(ws(char(',')), parse_attribute_definition)(input)
}

pub fn parse_attribute_definition(input: &str) -> IResult<&str, AttributeDefinition> {
    map(
        tuple((ws(parse_identifier), ws(char(':')), ws(parse_data_type))),
        |(name, _, data_type)| AttributeDefinition {
            name: name.to_string(),
            data_type,
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_definition_parser() {
        let values = [
            (
                "name: string",
                AttributeDefinition {
                    name: "name".to_string(),
                    data_type: DataType::String,
                },
            ),
            (
                "age: int",
                AttributeDefinition {
                    name: "age".to_string(),
                    data_type: DataType::Int,
                },
            ),
            (
                "date_of_birth: date",
                AttributeDefinition {
                    name: "date_of_birth".to_string(),
                    data_type: DataType::Date,
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_attribute_definition(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_attribute_definitions_parser() {
        let values = [
            (
                "name: string,
                age: int,
                date_of_birth: date",
                vec![
                    AttributeDefinition {
                        name: "name".to_string(),
                        data_type: DataType::String,
                    },
                    AttributeDefinition {
                        name: "age".to_string(),
                        data_type: DataType::Int,
                    },
                    AttributeDefinition {
                        name: "date_of_birth".to_string(),
                        data_type: DataType::Date,
                    },
                ],
            ),
            (
                "name: string,
                start_date: date,
                end_date: date",
                vec![
                    AttributeDefinition {
                        name: "name".to_string(),
                        data_type: DataType::String,
                    },
                    AttributeDefinition {
                        name: "start_date".to_string(),
                        data_type: DataType::Date,
                    },
                    AttributeDefinition {
                        name: "end_date".to_string(),
                        data_type: DataType::Date,
                    },
                ],
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_attribute_definitions1(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
