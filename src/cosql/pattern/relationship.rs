use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, terminated, tuple},
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, parse_variable, ws},
    insertion::{parse_attributes0, Attributes},
};

pub type Roles = Vec<Role>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    pub role: Option<String>,
    pub entity: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipPattern {
    pub variable: Option<String>,
    pub roles: Roles,
    pub relationship_type: String,
    pub attributes: Attributes,
}

pub fn parse_role(input: &str) -> IResult<&str, Role> {
    map(
        tuple((
            opt(terminated(ws(parse_identifier), ws(char(':')))),
            ws(parse_variable),
        )),
        |(role, entity)| Role {
            role: role.map(ToString::to_string),
            entity: entity.to_string(),
        },
    )(input)
}

pub fn parse_roles1(input: &str) -> IResult<&str, Roles> {
    delimited(
        ws(char('(')),
        separated_list1(ws(char(',')), parse_role),
        ws(char(')')),
    )(input)
}

pub fn parse_roles0(input: &str) -> IResult<&str, Roles> {
    delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_role),
        ws(char(')')),
    )(input)
}

pub fn parse_relationship_pattern(input: &str) -> IResult<&str, RelationshipPattern> {
    map(
        tuple((
            opt(ws(parse_variable)),
            parse_roles1,
            ws(tag("forms")),
            ws(parse_identifier),
            opt(parse_attributes0),
        )),
        |(variable, roles, _, relationship_type, attributes)| RelationshipPattern {
            variable: variable.map(|v| v.to_string()),
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
                Role {
                    role: Some("project".to_string()),
                    entity: "rust_project".to_string(),
                },
            ),
            (
                "assignee: $rust_dev",
                Role {
                    role: Some("assignee".to_string()),
                    entity: "rust_dev".to_string(),
                },
            ),
            (
                "$person1",
                Role {
                    role: None,
                    entity: "person1".to_string(),
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_role(source).unwrap();

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
                    Role {
                        role: Some("project".to_string()),
                        entity: "rust_project".to_string(),
                    },
                    Role {
                        role: Some("assignee".to_string()),
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
                    Role {
                        role: Some("employee".to_string()),
                        entity: "rust_dev".to_string(),
                    },
                    Role {
                        role: Some("department".to_string()),
                        entity: "department1".to_string(),
                    },
                ],
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_roles1(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
