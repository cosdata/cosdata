use nom::{
    bytes::complete::tag,
    combinator::{map, opt},
    sequence::tuple,
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, ws},
    insertion::{parse_attributes0, Attributes},
    pattern::relationship::{parse_roles1, Roles},
};

#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipInference {
    pub roles: Roles,
    pub relationship_type: String,
    pub attributes: Attributes,
}

pub fn parse_relationship_inference(input: &str) -> IResult<&str, RelationshipInference> {
    map(
        tuple((
            ws(parse_roles1),
            ws(tag("forms")),
            ws(parse_identifier),
            opt(parse_attributes0),
        )),
        |(roles, _, relationship_type, attributes)| RelationshipInference {
            roles,
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{insertion::Attribute, pattern::relationship::Role, value::Date, Value};

    #[test]
    fn test_relationship_inference_parser() {
        let values = [
            (
                "(
                    $person1,
                    $company1
                ) forms works_in (
                    since: 2-10-1999
                )",
                RelationshipInference {
                    roles: vec![
                        Role {
                            role: None,
                            entity: "person1".to_string(),
                        },
                        Role {
                            role: None,
                            entity: "company1".to_string(),
                        },
                    ],
                    relationship_type: "works_in".to_string(),
                    attributes: vec![Attribute {
                        name: "since".to_string(),
                        value: Value::Date(Date(2, 10, 1999)),
                    }],
                },
            ),
            (
                "(
                    from: $city1,
                    to: $city2
                ) forms reachable (
                    distance: $dist
                )",
                RelationshipInference {
                    roles: vec![
                        Role {
                            role: Some("from".to_string()),
                            entity: "city1".to_string(),
                        },
                        Role {
                            role: Some("to".to_string()),
                            entity: "city2".to_string(),
                        },
                    ],
                    relationship_type: "reachable".to_string(),
                    attributes: vec![Attribute {
                        name: "distance".to_string(),
                        value: Value::Variable("dist".to_string()),
                    }],
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_relationship_inference(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
