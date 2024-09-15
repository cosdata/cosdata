use nom::{branch::alt, bytes::complete::tag, character::complete::char, combinator::map, IResult};

use super::{
    common::{parse_identifier, ws},
    inference::parse_inferences1,
    pattern::parse_patterns0,
    ComputeClauses, Inferences, Patterns,
};

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceType {
    Derive,
    Materialize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub name: String,
    pub patterns: Patterns,
    pub compute_clauses: Option<ComputeClauses>,
    pub inference_type: InferenceType,
    pub inferences: Inferences,
}

pub fn parse_inference_type(input: &str) -> IResult<&str, InferenceType> {
    alt((
        map(tag("derive"), |_| InferenceType::Derive),
        map(tag("materialize"), |_| InferenceType::Materialize),
    ))(input)
}

pub fn parse_rule(input: &str) -> IResult<&str, Rule> {
    let (input, name) = ws(parse_identifier)(input)?;
    let (input, _) = ws(tag("as"))(input)?;
    let (input, _) = ws(tag("match"))(input)?;

    let (input, patterns) = parse_patterns0(input)?;

    let (input, _) = ws(tag("infer"))(input)?;
    let (input, inference_type) = parse_inference_type(input)?;
    let (input, inferences) = parse_inferences1(input)?;
    let (input, _) = ws(char(';'))(input)?;

    let rule = Rule {
        name: name.to_string(),
        patterns,
        compute_clauses: None,
        inference_type,
        inferences,
    };

    Ok((input, rule))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{
        condition::{BinaryCondition, BinaryConditionOperator, Condition},
        inference::relationship::RelationshipInference,
        pattern::relationship::{RelationshipPattern, Role},
        Inference, Pattern, Value,
    };

    #[test]
    fn test_inference_type_parser() {
        let values = [
            ("derive", InferenceType::Derive),
            ("materialize", InferenceType::Materialize),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_inference_type(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_rule_parser() {
        // the `parse_rule` function assumes the `define rule` part is already consumed
        let values = [
            (
                "reachable_direct as
                    match
                        (from: $city1, to: $city2) forms direct_flight
                    infer
                        materialize (from: $city1, to: $city2) forms reachable;",
                Rule {
                    name: "reachable_direct".to_string(),
                    patterns: vec![Pattern::RelationshipPattern(RelationshipPattern {
                        variable: None,
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
                        relationship_type: "direct_flight".to_string(),
                        attributes: vec![],
                    })],
                    compute_clauses: None,
                    inference_type: InferenceType::Materialize,
                    inferences: vec![Inference::RelationshipInference(RelationshipInference {
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
                        attributes: vec![],
                    })],
                },
            ),
            (
                "reachable_indirect as
                    match
                        (from: $city1, to: $intermediate) forms reachable,
                        (from: $intermediate, to: $city2) forms reachable,
                        $city1 != $city2
                    infer
                        materialize (from: $city1, to: $city2) forms reachable;",
                Rule {
                    name: "reachable_indirect".to_string(),
                    patterns: vec![
                        Pattern::RelationshipPattern(RelationshipPattern {
                            variable: None,
                            roles: vec![
                                Role {
                                    role: Some("from".to_string()),
                                    entity: "city1".to_string(),
                                },
                                Role {
                                    role: Some("to".to_string()),
                                    entity: "intermediate".to_string(),
                                },
                            ],
                            relationship_type: "reachable".to_string(),
                            attributes: vec![],
                        }),
                        Pattern::RelationshipPattern(RelationshipPattern {
                            variable: None,
                            roles: vec![
                                Role {
                                    role: Some("from".to_string()),
                                    entity: "intermediate".to_string(),
                                },
                                Role {
                                    role: Some("to".to_string()),
                                    entity: "city2".to_string(),
                                },
                            ],
                            relationship_type: "reachable".to_string(),
                            attributes: vec![],
                        }),
                        Pattern::Condition(Condition::Binary(BinaryCondition {
                            left: "city1".to_string(),
                            operator: BinaryConditionOperator::Inequality,
                            right: Value::Variable("city2".to_string()),
                        })),
                    ],

                    compute_clauses: None,
                    inference_type: InferenceType::Materialize,
                    inferences: vec![Inference::RelationshipInference(RelationshipInference {
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
                        attributes: vec![],
                    })],
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_rule(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
