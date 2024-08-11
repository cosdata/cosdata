use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    multi::separated_list1,
    sequence::{delimited, preceded, tuple},
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, ws},
    insertion::{parse_attributes0, Attributes},
    pattern::relationship::{RoleAssignment, RoleAssignments},
};

#[derive(Debug, Clone)]
pub struct RelationshipInference {
    pub roles: RoleAssignments,
    pub relationship_type: String,
    pub attributes: Attributes,
}

pub fn parse_relationship_inference(input: &str) -> IResult<&str, RelationshipInference> {
    map(
        tuple((
            delimited(
                ws(char('(')),
                separated_list1(ws(char(',')), preceded(char('$'), parse_identifier)),
                ws(char(')')),
            ),
            ws(tag("forms")),
            ws(parse_identifier),
            opt(parse_attributes0),
        )),
        |(roles, _, relationship_type, attributes)| RelationshipInference {
            roles: roles
                .into_iter()
                .map(|r| RoleAssignment {
                    role: r.to_string(),
                    entity: r.to_string(),
                })
                .collect(),
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}
