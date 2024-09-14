use nom::{
    bytes::complete::tag,
    combinator::{map, opt},
    sequence::tuple,
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, parse_variable, ws},
    insertion::{parse_attributes0, Attributes},
};

#[derive(Debug, Clone, PartialEq)]
pub struct EntityPattern {
    pub variable: String,
    pub entity_type: String,
    pub attributes: Attributes,
}

pub fn parse_entity_pattern(input: &str) -> IResult<&str, EntityPattern> {
    map(
        tuple((
            ws(parse_variable),
            ws(tag("isa")),
            ws(parse_identifier),
            opt(parse_attributes0),
        )),
        |(variable, _, entity_type, attributes)| EntityPattern {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{insertion::Attribute, Value};

    #[test]
    fn test_entity_pattern_parser() {
        let values = [
            (
                r#"$person1 isa person (
                    name: $name,
                    age: 18,
                    gender: "M"
                )"#,
                EntityPattern {
                    variable: "person1".to_string(),
                    entity_type: "person".to_string(),
                    attributes: vec![
                        Attribute {
                            name: "name".to_string(),
                            value: Value::Variable("name".to_string()),
                        },
                        Attribute {
                            name: "age".to_string(),
                            value: Value::Int(18),
                        },
                        Attribute {
                            name: "gender".to_string(),
                            value: Value::String("M".to_string()),
                        },
                    ],
                },
            ),
            (
                r#"$city1 isa city (
                    name: "New York"  
                )"#,
                EntityPattern {
                    variable: "city1".to_string(),
                    entity_type: "city".to_string(),
                    attributes: vec![Attribute {
                        name: "name".to_string(),
                        value: Value::String("New York".to_string()),
                    }],
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_entity_pattern(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
