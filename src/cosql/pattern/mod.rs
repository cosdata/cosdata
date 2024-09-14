pub mod entity;
pub mod relationship;

use nom::{
    branch::alt, character::complete::char, combinator::map, multi::separated_list0, IResult,
};

use entity::{parse_entity_pattern, EntityPattern};
use relationship::{parse_relationship_pattern, RelationshipPattern};

use super::{
    common::ws,
    condition::{parse_condition, Condition},
};

pub type Patterns = Vec<Pattern>;

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    EntityPattern(EntityPattern),
    RelationshipPattern(RelationshipPattern),
    Condition(Condition),
}

pub fn parse_patterns0(input: &str) -> IResult<&str, Patterns> {
    separated_list0(ws(char(',')), parse_pattern)(input)
}

pub fn parse_patterns1(input: &str) -> IResult<&str, Patterns> {
    separated_list0(ws(char(',')), parse_pattern)(input)
}

pub fn parse_pattern(input: &str) -> IResult<&str, Pattern> {
    alt((
        map(parse_entity_pattern, |ep| Pattern::EntityPattern(ep)),
        map(parse_relationship_pattern, |rp| {
            Pattern::RelationshipPattern(rp)
        }),
        map(parse_condition, |c| Pattern::Condition(c)),
    ))(input)
}

#[cfg(test)]
mod tests {
    use super::{relationship::Role, *};
    use crate::cosql::{
        condition::{BinaryCondition, BinaryConditionOperator, LogicalCondition, LogicalOperator},
        insertion::Attribute,
        value::Date,
        Value,
    };

    #[test]
    fn test_pattern_parser() {
        let values = [
            (
                r#"$person1 isa person (
                    name: $name,
                    age: 18,
                    gender: "M"
                )"#,
                Pattern::EntityPattern(EntityPattern {
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
                Pattern::EntityPattern(EntityPattern {
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
                Pattern::RelationshipPattern(RelationshipPattern {
                    variable: None,
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
                "$relation1 (
                    from: $city1,
                    to: $city2
                ) forms reachable (
                    distance: $dist
                )",
                Pattern::RelationshipPattern(RelationshipPattern {
                    variable: Some("relation1".to_string()),
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
                "$age < 18",
                Pattern::Condition(Condition::Binary(BinaryCondition {
                    left: "age".to_string(),
                    operator: BinaryConditionOperator::LessThan,
                    right: Value::Int(18),
                })),
            ),
            (
                r#"(((($programming_language == "Rust"))))"#,
                Pattern::Condition(Condition::Binary(BinaryCondition {
                    left: "programming_language".to_string(),
                    operator: BinaryConditionOperator::Equality,
                    right: Value::String("Rust".to_string()),
                })),
            ),
            (
                r#"$programming_language == "Rust" and $salary >= 1000000"#,
                Pattern::Condition(Condition::Logical(Box::new(LogicalCondition {
                    left: Condition::Binary(BinaryCondition {
                        left: "programming_language".to_string(),
                        operator: BinaryConditionOperator::Equality,
                        right: Value::String("Rust".to_string()),
                    }),
                    operator: LogicalOperator::And,
                    right: Condition::Binary(BinaryCondition {
                        left: "salary".to_string(),
                        operator: BinaryConditionOperator::GreaterEqualThan,
                        right: Value::Int(1000000),
                    }),
                }))),
            ),
            (
                r#"($programming_language == "Rust" and $salary >= 1000000) or $age < 18"#,
                Pattern::Condition(Condition::Logical(Box::new(LogicalCondition {
                    left: Condition::Logical(Box::new(LogicalCondition {
                        left: Condition::Binary(BinaryCondition {
                            left: "programming_language".to_string(),
                            operator: BinaryConditionOperator::Equality,
                            right: Value::String("Rust".to_string()),
                        }),
                        operator: LogicalOperator::And,
                        right: Condition::Binary(BinaryCondition {
                            left: "salary".to_string(),
                            operator: BinaryConditionOperator::GreaterEqualThan,
                            right: Value::Int(1000000),
                        }),
                    })),
                    operator: LogicalOperator::Or,
                    right: Condition::Binary(BinaryCondition {
                        left: "age".to_string(),
                        operator: BinaryConditionOperator::LessThan,
                        right: Value::Int(18),
                    }),
                }))),
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_pattern(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
