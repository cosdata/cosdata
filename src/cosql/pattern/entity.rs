use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    sequence::tuple,
    IResult,
};

use crate::cosql::{
    common::{parse_identifier, ws},
    insertion::{parse_attributes0, Attributes},
};

#[derive(Debug, Clone)]
pub struct EntityPattern {
    pub variable: String,
    pub entity_type: String,
    pub attributes: Attributes,
}

pub fn parse_entity_pattern(input: &str) -> IResult<&str, EntityPattern> {
    map(
        tuple((
            ws(char('$')),
            ws(parse_identifier),
            ws(tag("isa")),
            ws(parse_identifier),
            opt(parse_attributes0),
        )),
        |(_, variable, _, entity_type, attributes)| EntityPattern {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}
