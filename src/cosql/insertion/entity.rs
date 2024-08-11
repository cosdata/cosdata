use nom::{
    bytes::complete::tag, character::complete::char, combinator::map, sequence::tuple, IResult,
};

use crate::cosql::common::{parse_identifier, ws};

use super::{parse_attributes1, Attributes};

#[derive(Debug, Clone)]
pub struct EntityInsertion {
    pub variable: String,
    pub entity_type: String,
    pub attributes: Attributes,
}

pub fn parse_entity_insertion(input: &str) -> IResult<&str, EntityInsertion> {
    map(
        tuple((
            ws(char('$')),
            ws(parse_identifier),
            ws(tag("isa")),
            ws(parse_identifier),
            parse_attributes1,
            ws(char(';')),
        )),
        |(_, variable, _, entity_type, attributes, _)| EntityInsertion {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes,
        },
    )(input)
}
