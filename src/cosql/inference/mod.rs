pub mod entity;
pub mod extend_entity;
pub mod relationship;

use nom::{
    branch::alt, character::complete::char, combinator::map, multi::separated_list1, IResult,
};

use super::common::ws;
use entity::{parse_entity_inference, EntityInference};
use extend_entity::{parse_extend_entity_inference, ExtendEntityInference};
use relationship::{parse_relationship_inference, RelationshipInference};

pub type Inferences = Vec<Inference>;

#[derive(Debug, Clone, PartialEq)]
pub enum Inference {
    EntityInference(EntityInference),
    RelationshipInference(RelationshipInference),
    ExtendEntityInference(ExtendEntityInference),
}

pub fn parse_inferences0(input: &str) -> IResult<&str, Inferences> {
    separated_list1(ws(char(',')), parse_inference)(input)
}

pub fn parse_inferences1(input: &str) -> IResult<&str, Inferences> {
    separated_list1(ws(char(',')), parse_inference)(input)
}

pub fn parse_inference(input: &str) -> IResult<&str, Inference> {
    alt((
        map(parse_entity_inference, Inference::EntityInference),
        map(parse_extend_entity_inference, |i| {
            Inference::ExtendEntityInference(i)
        }),
        map(parse_relationship_inference, |i| {
            Inference::RelationshipInference(i)
        }),
    ))(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{insertion::Attribute, pattern::relationship::Role, value::Date, Value};

    #[test]
    fn test_inference_parser() {
        let values = [
            (
                r#"$person1 isa person (
                    name: $name,
                    age: 18,
                    gender: "M"
                )"#,
                Inference::EntityInference(EntityInference {
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
                }),
            ),
            (
                r#"$city1 isa city (
                    name: "New York"  
                )"#,
                Inference::EntityInference(EntityInference {
                    variable: "city1".to_string(),
                    entity_type: "city".to_string(),
                    attributes: vec![Attribute {
                        name: "name".to_string(),
                        value: Value::String("New York".to_string()),
                    }],
                }),
            ),
            (
                "(
                    $person1,
                    $company1
                ) forms works_in (
                    since: 2-10-1999
                )",
                Inference::RelationshipInference(RelationshipInference {
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
                }),
            ),
            (
                "(
                    from: $city1,
                    to: $city2
                ) forms reachable (
                    distance: $dist
                )",
                Inference::RelationshipInference(RelationshipInference {
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
                }),
            ),
            (
                "extend $person1 (
                    salary: 100000
                )",
                Inference::ExtendEntityInference(ExtendEntityInference {
                    variable: "person1".to_string(),
                    attributes: vec![Attribute {
                        name: "salary".to_string(),
                        value: Value::Int(100000),
                    }],
                }),
            ),
            (
                "extend $city1 (
                    visitors: $visitors
                )",
                Inference::ExtendEntityInference(ExtendEntityInference {
                    variable: "city1".to_string(),
                    attributes: vec![Attribute {
                        name: "visitors".to_string(),
                        value: Value::Variable("visitors".to_string()),
                    }],
                }),
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_inference(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_inferences_parser() {
        let values = [
            (
                r#"$person1 isa person (
                    name: $name,
                    age: 18,
                    gender: "M"
                ),
                
                $city1 isa city (
                    name: "New York"  
                ),
                
                (
                    $person1,
                    $company1
                ) forms works_in (
                    since: 2-10-1999
                )"#,
                vec![
                    Inference::EntityInference(EntityInference {
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
                    }),
                    Inference::EntityInference(EntityInference {
                        variable: "city1".to_string(),
                        entity_type: "city".to_string(),
                        attributes: vec![Attribute {
                            name: "name".to_string(),
                            value: Value::String("New York".to_string()),
                        }],
                    }),
                    Inference::RelationshipInference(RelationshipInference {
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
                    }),
                ],
            ),
            (
                "(
                    from: $city1,
                    to: $city2
                ) forms reachable (
                    distance: $dist
                ),
                
                extend $person1 (
                    salary: 100000
                ),
                
                extend $city1 (
                    visitors: $visitors
                )",
                vec![
                    Inference::RelationshipInference(RelationshipInference {
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
                    }),
                    Inference::ExtendEntityInference(ExtendEntityInference {
                        variable: "person1".to_string(),
                        attributes: vec![Attribute {
                            name: "salary".to_string(),
                            value: Value::Int(100000),
                        }],
                    }),
                    Inference::ExtendEntityInference(ExtendEntityInference {
                        variable: "city1".to_string(),
                        attributes: vec![Attribute {
                            name: "visitors".to_string(),
                            value: Value::Variable("visitors".to_string()),
                        }],
                    }),
                ],
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_inferences1(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
