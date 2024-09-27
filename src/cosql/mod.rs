pub mod common;
pub mod compute_clause;
pub mod condition;
pub mod data_type;
pub mod definition;
pub mod expression;
pub mod inference;
pub mod insertion;
pub mod pattern;
mod precedence;
pub mod query;
pub mod rule;
pub mod value;

use common::ws_tag;
use nom::{branch::alt, combinator::map, multi::many0, sequence::preceded, IResult};

use definition::{
    entity::parse_entity_definition, relationship::parse_relationship_definition, EntityDefinition,
    RelationshipDefinition,
};
use insertion::{
    entity::parse_entity_insertion, relationship::parse_relationship_insertion, EntityInsertion,
    RelationshipInsertion,
};
use query::{parse_query, Query};
use rule::{parse_rule, Rule};

pub use compute_clause::{ComputeClause, ComputeClauses};
pub use data_type::DataType;
pub use expression::Expression;
pub use inference::{Inference, Inferences};
pub use pattern::{Pattern, Patterns};
pub use precedence::Precedence;
pub use value::{Date, Value};

pub type CosQLStatements = Vec<CosQLStatement>;

#[derive(Debug, Clone, PartialEq)]
pub enum CosQLStatement {
    EntityDefinition(EntityDefinition),
    RelationshipDefinition(RelationshipDefinition),
    EntityInsertion(EntityInsertion),
    RelationshipInsertion(RelationshipInsertion),
    Query(Query),
    Rule(Rule),
}

pub fn parse_cosql_statements(input: &str) -> IResult<&str, CosQLStatements> {
    many0(parse_cosql_statement)(input)
}

pub fn parse_cosql_statement(input: &str) -> IResult<&str, CosQLStatement> {
    alt((
        preceded(
            ws_tag("define"),
            alt((
                preceded(
                    ws_tag("entity"),
                    map(parse_entity_definition, |ed| {
                        CosQLStatement::EntityDefinition(ed)
                    }),
                ),
                preceded(
                    ws_tag("relationship"),
                    map(parse_relationship_definition, |rd| {
                        CosQLStatement::RelationshipDefinition(rd)
                    }),
                ),
                preceded(ws_tag("rule"), map(parse_rule, |r| CosQLStatement::Rule(r))),
            )),
        ),
        preceded(
            ws_tag("insert"),
            alt((
                map(parse_entity_insertion, |ei| {
                    CosQLStatement::EntityInsertion(ei)
                }),
                map(parse_relationship_insertion, |ri| {
                    CosQLStatement::RelationshipInsertion(ri)
                }),
            )),
        ),
        preceded(
            ws_tag("match"),
            map(parse_query, |q| CosQLStatement::Query(q)),
        ),
    ))(input)
}

#[cfg(test)]
mod tests {
    use super::{
        condition::{BinaryCondition, BinaryConditionOperator, Condition},
        definition::{relationship::RoleDefinition, AttributeDefinition},
        insertion::Attribute,
        pattern::{
            entity::EntityPattern,
            relationship::{RelationshipPattern, Role},
        },
        *,
    };

    #[test]
    fn test_cosql_statement_parser() {
        let values = [
            (
                "define entity person as
                    name: string,
                    age: int,
                    date_of_birth: date;",
                CosQLStatement::EntityDefinition(EntityDefinition {
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
                }),
            ),
            (
                "define entity project as
                    name: string,
                    start_date: date,
                    end_date: date;",
                CosQLStatement::EntityDefinition(EntityDefinition {
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
                }),
            ),
            (
                "define relationship assigned_to as (
                    project: project,
                    assignee: person
                );",
                CosQLStatement::RelationshipDefinition(RelationshipDefinition {
                    name: "assigned_to".to_string(),
                    roles: vec![
                        RoleDefinition {
                            name: "project".to_string(),
                            entity_type: "project".to_string(),
                        },
                        RoleDefinition {
                            name: "assignee".to_string(),
                            entity_type: "person".to_string(),
                        },
                    ],
                    attributes: vec![],
                }),
            ),
            (
                "define relationship works_in as (
                    employee: person,
                    department: department
                ), salary: int;",
                CosQLStatement::RelationshipDefinition(RelationshipDefinition {
                    name: "works_in".to_string(),
                    roles: vec![
                        RoleDefinition {
                            name: "employee".to_string(),
                            entity_type: "person".to_string(),
                        },
                        RoleDefinition {
                            name: "department".to_string(),
                            entity_type: "department".to_string(),
                        },
                    ],
                    attributes: vec![AttributeDefinition {
                        name: "salary".to_string(),
                        data_type: DataType::Int,
                    }],
                }),
            ),
            (
                r#"insert $rust_dev isa person (
                    name: "The Rust Dev",
                    age: 54,
                    date_of_birth: 01-01-1970
                );"#,
                CosQLStatement::EntityInsertion(EntityInsertion {
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
                }),
            ),
            (
                r#"insert $rust_project isa project (
                    name: "A Rust Project",
                    start_date: 01-01-2000,
                    end_date: 31-12-2009 
                );"#,
                CosQLStatement::EntityInsertion(EntityInsertion {
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
                }),
            ),
            (
                "insert $relation1 (
                    project: $rust_project,
                    assignee: $rust_dev
                ) forms assigned_to;",
                CosQLStatement::RelationshipInsertion(RelationshipInsertion {
                    variable: "relation1".to_string(),
                    roles: vec![
                        Role {
                            role: Some("project".to_string()),
                            entity: "rust_project".to_string(),
                        },
                        Role {
                            role: Some("assignee".to_string()),
                            entity: "rust_dev".to_string(),
                        },
                    ],
                    relationship_type: "assigned_to".to_string(),
                    attributes: vec![],
                }),
            ),
            (
                "insert $relation2 (
                    employee: $rust_dev,
                    department: $department1
                ) forms works_in (
                    salary: 100000
                );",
                CosQLStatement::RelationshipInsertion(RelationshipInsertion {
                    variable: "relation2".to_string(),
                    roles: vec![
                        Role {
                            role: Some("employee".to_string()),
                            entity: "rust_dev".to_string(),
                        },
                        Role {
                            role: Some("department".to_string()),
                            entity: "department1".to_string(),
                        },
                    ],
                    relationship_type: "works_in".to_string(),
                    attributes: vec![Attribute {
                        name: "salary".to_string(),
                        value: Value::Int(100000),
                    }],
                }),
            ),
            (
                "match
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
                CosQLStatement::Query(Query {
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
                }),
            ),
            (
                r#"match
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
                CosQLStatement::Query(Query {
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
                }),
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_cosql_statement(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
