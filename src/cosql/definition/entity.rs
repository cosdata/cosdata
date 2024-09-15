use nom::{
    bytes::complete::tag, character::complete::char, combinator::map, sequence::tuple, IResult,
};

use super::{parse_attribute_definitions1, AttributeDefinitions};
use crate::cosql::common::{parse_identifier, ws};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityDefinition {
    pub name: String,
    pub attributes: AttributeDefinitions,
}

pub fn parse_entity_definition(input: &str) -> IResult<&str, EntityDefinition> {
    map(
        tuple((
            ws(parse_identifier),
            ws(tag("as")),
            parse_attribute_definitions1,
            ws(char(';')),
        )),
        |(name, _, attributes, _)| EntityDefinition {
            name: name.to_string(),
            attributes,
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{definition::AttributeDefinition, DataType};

    #[test]
    fn test_entity_definition_parser() {
        // the `parse_entity_definition` function assumes the `define entity` part is already
        // consumed
        let values = [
            (
                "person as
                    name: string,
                    age: int,
                    date_of_birth: date;",
                EntityDefinition {
                    name: "person".to_string(),
                    attributes: vec![
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
                },
            ),
            (
                "project as
                    name: string,
                    start_date: date,
                    end_date: date;",
                EntityDefinition {
                    name: "project".to_string(),
                    attributes: vec![
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
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_entity_definition(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
