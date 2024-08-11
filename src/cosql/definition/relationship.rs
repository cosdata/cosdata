use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, preceded, tuple},
    IResult,
};

use crate::cosql::common::{parse_identifier, ws};

use super::{parse_attribute_definition, AttributeDefinitions};

pub type Roles = Vec<Role>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    pub entity_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationshipDefinition {
    pub name: String,
    pub roles: Roles,
    pub attributes: AttributeDefinitions,
}

pub fn parse_roles0(input: &str) -> IResult<&str, Roles> {
    delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_role),
        ws(char(')')),
    )(input)
}

pub fn parse_roles1(input: &str) -> IResult<&str, Roles> {
    delimited(
        ws(char('(')),
        separated_list1(ws(char(',')), parse_role),
        ws(char(')')),
    )(input)
}

pub fn parse_role(input: &str) -> IResult<&str, Role> {
    map(
        tuple((ws(parse_identifier), ws(char(':')), ws(parse_identifier))),
        |(name, _, entity_type)| Role {
            name: name.to_string(),
            entity_type: entity_type.to_string(),
        },
    )(input)
}

pub fn parse_relationship_definition(input: &str) -> IResult<&str, RelationshipDefinition> {
    map(
        tuple((
            ws(parse_identifier),
            ws(tag("as")),
            parse_roles0,
            opt(preceded(
                ws(char(',')),
                separated_list0(ws(char(',')), parse_attribute_definition),
            )),
            ws(char(';')),
        )),
        |(name, _, roles, attributes, _)| RelationshipDefinition {
            name: name.to_string(),
            roles,
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{definition::AttributeDefinition, DataType};

    #[test]
    fn test_role_parser() {
        let values = [
            (
                "project: project",
                Role {
                    name: "project".to_string(),
                    entity_type: "project".to_string(),
                },
            ),
            (
                "assignee: person",
                Role {
                    name: "assignee".to_string(),
                    entity_type: "person".to_string(),
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_role(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_roles_parser() {
        let values = [
            (
                "(project: project, assignee: person)",
                vec![
                    Role {
                        name: "project".to_string(),
                        entity_type: "project".to_string(),
                    },
                    Role {
                        name: "assignee".to_string(),
                        entity_type: "person".to_string(),
                    },
                ],
            ),
            (
                "(employee: person, department: department)",
                vec![
                    Role {
                        name: "employee".to_string(),
                        entity_type: "person".to_string(),
                    },
                    Role {
                        name: "department".to_string(),
                        entity_type: "department".to_string(),
                    },
                ],
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_roles1(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_relationship_definition_parser() {
        // the `parse_relationship_definition` function assumes the `define relationship`
        let values = [
            (
                "assigned_to as (project: project, assignee: person);",
                RelationshipDefinition {
                    name: "assigned_to".to_string(),
                    roles: vec![
                        Role {
                            name: "project".to_string(),
                            entity_type: "project".to_string(),
                        },
                        Role {
                            name: "assignee".to_string(),
                            entity_type: "person".to_string(),
                        },
                    ],
                    attributes: vec![],
                },
            ),
            (
                "works_in as (employee: person, department: department), salary: int;",
                RelationshipDefinition {
                    name: "works_in".to_string(),
                    roles: vec![
                        Role {
                            name: "employee".to_string(),
                            entity_type: "person".to_string(),
                        },
                        Role {
                            name: "department".to_string(),
                            entity_type: "department".to_string(),
                        },
                    ],
                    attributes: vec![AttributeDefinition {
                        name: "salary".to_string(),
                        data_type: DataType::Int,
                    }],
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_relationship_definition(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
