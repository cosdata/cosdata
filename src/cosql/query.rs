use nom::{
    bytes::complete::tag, character::complete::char, combinator::map, multi::separated_list1,
    sequence::tuple, IResult,
};

use super::{
    common::{parse_variable, ws},
    pattern::parse_patterns0,
    Patterns,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub patterns: Patterns,
    pub get_variables: Vec<String>,
}

pub fn parse_query(input: &str) -> IResult<&str, Query> {
    map(
        tuple((
            parse_patterns0,
            ws(tag("get")),
            separated_list1(ws(char(',')), parse_variable),
            ws(char(';')),
        )),
        |(patterns, _, get_variables, _)| Query {
            patterns,
            get_variables: get_variables.into_iter().map(ToString::to_string).collect(),
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{
        condition::{BinaryCondition, BinaryConditionOperator, Condition},
        insertion::Attribute,
        pattern::{
            entity::EntityPattern,
            relationship::{RelationshipPattern, Role},
        },
        Pattern, Value,
    };

    #[test]
    fn test_query_parser() {
        // the `parse_query` function assumes the `match` part is already consumed
        let values = [
            (
                "
                    $employee1 isa person (
                        name: $name1
                    ),
                    $employee2 isa person (
                        name: $name2
                    ),
                    $project isa project (
                        name: $project_name
                    ),
                    ($employee1, $project) forms assigned_to,
                    ($employee2, $project) forms assigned_to,
                    $employee1 != $employee2
                get $name1, $name2, $project_name;",
                Query {
                    patterns: vec![
                        Pattern::EntityPattern(EntityPattern {
                            variable: "employee1".to_string(),
                            entity_type: "person".to_string(),
                            attributes: vec![Attribute {
                                name: "name".to_string(),
                                value: Value::Variable("name1".to_string()),
                            }],
                        }),
                        Pattern::EntityPattern(EntityPattern {
                            variable: "employee2".to_string(),
                            entity_type: "person".to_string(),
                            attributes: vec![Attribute {
                                name: "name".to_string(),
                                value: Value::Variable("name2".to_string()),
                            }],
                        }),
                        Pattern::EntityPattern(EntityPattern {
                            variable: "project".to_string(),
                            entity_type: "project".to_string(),
                            attributes: vec![Attribute {
                                name: "name".to_string(),
                                value: Value::Variable("project_name".to_string()),
                            }],
                        }),
                        Pattern::RelationshipPattern(RelationshipPattern {
                            variable: None,
                            roles: vec![
                                Role {
                                    role: None,
                                    entity: "employee1".to_string(),
                                },
                                Role {
                                    role: None,
                                    entity: "project".to_string(),
                                },
                            ],
                            relationship_type: "assigned_to".to_string(),
                            attributes: vec![],
                        }),
                        Pattern::RelationshipPattern(RelationshipPattern {
                            variable: None,
                            roles: vec![
                                Role {
                                    role: None,
                                    entity: "employee2".to_string(),
                                },
                                Role {
                                    role: None,
                                    entity: "project".to_string(),
                                },
                            ],
                            relationship_type: "assigned_to".to_string(),
                            attributes: vec![],
                        }),
                        Pattern::Condition(Condition::Binary(BinaryCondition {
                            left: "employee1".to_string(),
                            operator: BinaryConditionOperator::Inequality,
                            right: Value::Variable("employee2".to_string()),
                        })),
                    ],
                    get_variables: vec![
                        "name1".to_string(),
                        "name2".to_string(),
                        "project_name".to_string(),
                    ],
                },
            ),
            (
                r#"
                    $employee isa person (
                        name: $name
                    ),
                    $project isa project (
                        name: "AI Initiative"
                    ),
                    $assignment (
                        employee: $employee,
                        project: $project,
                        department: $dept
                    ) forms project_assignment (
                        start_date: $start_date
                    ),
                    $dept isa department (
                        name: "Tech Department"
                    )
                get $name, $start_date;"#,
                Query {
                    patterns: vec![
                        Pattern::EntityPattern(EntityPattern {
                            variable: "employee".to_string(),
                            entity_type: "person".to_string(),
                            attributes: vec![Attribute {
                                name: "name".to_string(),
                                value: Value::Variable("name".to_string()),
                            }],
                        }),
                        Pattern::EntityPattern(EntityPattern {
                            variable: "project".to_string(),
                            entity_type: "project".to_string(),
                            attributes: vec![Attribute {
                                name: "name".to_string(),
                                value: Value::String("AI Initiative".to_string()),
                            }],
                        }),
                        Pattern::RelationshipPattern(RelationshipPattern {
                            variable: Some("assignment".to_string()),
                            roles: vec![
                                Role {
                                    role: Some("employee".to_string()),
                                    entity: "employee".to_string(),
                                },
                                Role {
                                    role: Some("project".to_string()),
                                    entity: "project".to_string(),
                                },
                                Role {
                                    role: Some("department".to_string()),
                                    entity: "dept".to_string(),
                                },
                            ],
                            relationship_type: "project_assignment".to_string(),
                            attributes: vec![Attribute {
                                name: "start_date".to_string(),
                                value: Value::Variable("start_date".to_string()),
                            }],
                        }),
                        Pattern::EntityPattern(EntityPattern {
                            variable: "dept".to_string(),
                            entity_type: "department".to_string(),
                            attributes: vec![Attribute {
                                name: "name".to_string(),
                                value: Value::String("Tech Department".to_string()),
                            }],
                        }),
                    ],
                    get_variables: vec!["name".to_string(), "start_date".to_string()],
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_query(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
