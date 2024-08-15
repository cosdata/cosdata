use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    sequence::tuple,
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, parse_variable, ws},
    pattern::relationship::{parse_role_assignments, RoleAssignments},
};

use super::{parse_attributes0, Attributes};

#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipInsertion {
    pub variable: String,
    pub roles: RoleAssignments,
    pub relationship_type: String,
    pub attributes: Attributes,
}

pub fn parse_relationship_insertion(input: &str) -> IResult<&str, RelationshipInsertion> {
    map(
        tuple((
            ws(parse_variable),
            parse_role_assignments,
            ws(tag("forms")),
            ws(parse_identifier),
            opt(parse_attributes0),
            ws(char(';')),
        )),
        |(variable, roles, _, relationship_type, attributes, _)| RelationshipInsertion {
            variable: variable.to_string(),
            roles,
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{insertion::Attribute, pattern::relationship::RoleAssignment, Value};

    #[test]
    fn test_relationship_insertion_parser() {
        let values = [
            (
                "$relation1 (
                    project: $rust_project,
                    assignee: $rust_dev
                ) forms assigned_to;",
                RelationshipInsertion {
                    variable: "relation1".to_string(),
                    roles: vec![
                        RoleAssignment {
                            role: "project".to_string(),
                            entity: "rust_project".to_string(),
                        },
                        RoleAssignment {
                            role: "assignee".to_string(),
                            entity: "rust_dev".to_string(),
                        },
                    ],
                    relationship_type: "assigned_to".to_string(),
                    attributes: vec![],
                },
            ),
            (
                "$relation2 (
                    employee: $rust_dev,
                    department: $department1
                ) forms works_in (
                    salary: 100000
                );",
                RelationshipInsertion {
                    variable: "relation2".to_string(),
                    roles: vec![
                        RoleAssignment {
                            role: "employee".to_string(),
                            entity: "rust_dev".to_string(),
                        },
                        RoleAssignment {
                            role: "department".to_string(),
                            entity: "department1".to_string(),
                        },
                    ],
                    relationship_type: "works_in".to_string(),
                    attributes: vec![Attribute {
                        name: "salary".to_string(),
                        value: Value::Int(100000),
                    }],
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_relationship_insertion(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
