use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    sequence::tuple,
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, ws},
    pattern::relationship::{parse_role_assignments, RoleAssignments},
};

use super::{parse_attributes0, Attributes};

#[derive(Debug, Clone)]
pub struct RelationshipInsertion {
    pub variable: String,
    pub roles: RoleAssignments,
    pub relationship_type: String,
    pub attributes: Attributes,
}

pub fn parse_relationship_insertion(input: &str) -> IResult<&str, RelationshipInsertion> {
    map(
        tuple((
            ws(char('$')),
            ws(parse_identifier),
            parse_role_assignments,
            ws(tag("forms")),
            ws(parse_identifier),
            opt(parse_attributes0),
            ws(char(';')),
        )),
        |(_, variable, roles, _, relationship_type, attributes, _)| RelationshipInsertion {
            variable: variable.to_string(),
            roles,
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}
