use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    multi::separated_list1,
    sequence::{delimited, tuple},
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, parse_variable, ws},
    insertion::{parse_attributes0, Attributes},
};

pub type RoleAssignments = Vec<RoleAssignment>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleAssignment {
    pub role: String,
    pub entity: String,
}

#[derive(Debug, Clone)]
pub struct RelationshipPattern {
    pub variable: Option<String>,
    pub roles: RoleAssignments,
    pub relationship_type: String,
    pub attributes: Attributes,
}

pub fn parse_role_assignment(input: &str) -> IResult<&str, RoleAssignment> {
    map(
        tuple((ws(parse_identifier), ws(char(':')), ws(parse_variable))),
        |(role, _, entity)| RoleAssignment {
            role: role.to_string(),
            entity: entity.to_string(),
        },
    )(input)
}

pub fn parse_role_assignments(input: &str) -> IResult<&str, RoleAssignments> {
    delimited(
        ws(char('(')),
        separated_list1(ws(char(',')), parse_role_assignment),
        ws(char(')')),
    )(input)
}

pub fn parse_relationship_pattern(input: &str) -> IResult<&str, RelationshipPattern> {
    map(
        tuple((
            opt(tuple((ws(char('$')), ws(parse_identifier)))),
            parse_role_assignments,
            ws(tag("forms")),
            ws(parse_identifier),
            opt(parse_attributes0),
        )),
        |(variable, roles, _, relationship_type, attributes)| RelationshipPattern {
            variable: variable.map(|(_, v)| v.to_string()),
            roles,
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_assignment_parser() {
        let values = [
            (
                "project: $rust_project",
                RoleAssignment {
                    role: "project".to_string(),
                    entity: "rust_project".to_string(),
                },
            ),
            (
                "assignee: $rust_dev",
                RoleAssignment {
                    role: "assignee".to_string(),
                    entity: "rust_dev".to_string(),
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_role_assignment(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_role_assignments_parser() {
        let values = [
            (
                "(
                    project: $rust_project,
                    assignee: $rust_dev
                )",
                vec![
                    RoleAssignment {
                        role: "project".to_string(),
                        entity: "rust_project".to_string(),
                    },
                    RoleAssignment {
                        role: "assignee".to_string(),
                        entity: "rust_dev".to_string(),
                    },
                ],
            ),
            (
                "(
                    employee: $rust_dev,
                    department: $department1
                )",
                vec![
                    RoleAssignment {
                        role: "employee".to_string(),
                        entity: "rust_dev".to_string(),
                    },
                    RoleAssignment {
                        role: "department".to_string(),
                        entity: "department1".to_string(),
                    },
                ],
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_role_assignments(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
