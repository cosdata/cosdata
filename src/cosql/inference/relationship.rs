use nom::{
    bytes::complete::tag,
    combinator::{map, opt},
    sequence::tuple,
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, ws},
    insertion::{parse_attributes0, Attributes},
    pattern::relationship::{parse_roles1, Roles},
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
            ws(parse_roles1),
            ws(tag("forms")),
            ws(parse_identifier),
            opt(parse_attributes0),
        )),
        |(roles, _, relationship_type, attributes)| RelationshipInference {
            roles,
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}
