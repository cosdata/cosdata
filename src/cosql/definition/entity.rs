use nom::{
    bytes::complete::tag, character::complete::char, combinator::map, sequence::tuple, IResult,
};

use super::{parse_attribute_definitions1, AttributeDefinitions};
use crate::cosql::common::{parse_identifier, ws};

#[derive(Debug, Clone)]
pub struct EntityDefinition {
    pub name: String,
    pub attributes: AttributeDefinitions,
}

pub fn parse_entity_definition(input: &str) -> IResult<&str, EntityDefinition> {
    map(
        tuple((
            ws(parse_identifier),
            ws(tag("as")),
            parse_attribute_definitions1,
            ws(char(';')),
        )),
        |(name, _, attributes, _)| EntityDefinition {
            name: name.to_string(),
            attributes,
        },
    )(input)
}
