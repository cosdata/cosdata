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

#[derive(Debug, Clone)]
pub struct Role {
    pub name: String,
    pub entity_type: String,
}

#[derive(Debug, Clone)]
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
