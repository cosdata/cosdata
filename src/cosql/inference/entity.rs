use nom::{
    bytes::complete::tag, character::complete::char, combinator::map, sequence::tuple, IResult,
};

use crate::cosql::{
    common::{parse_identifier, ws},
    insertion::{parse_attributes0, Attributes},
};

#[derive(Debug, Clone)]
pub struct EntityInference {
    pub variable: String,
    pub entity_type: String,
    pub attributes: Attributes,
}

#[derive(Debug, Clone)]
pub struct ExtendEntityInference {
    pub variable: String,
    pub attributes: Attributes,
}

pub fn parse_entity_inference(input: &str) -> IResult<&str, EntityInference> {
    map(
        tuple((
            ws(char('$')),
            ws(parse_identifier),
            ws(tag("isa")),
            ws(parse_identifier),
            ws(parse_attributes0),
        )),
        |(_, variable, _, entity_type, attributes)| EntityInference {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes,
        },
    )(input)
}

pub fn parse_extend_entity_inference(input: &str) -> IResult<&str, ExtendEntityInference> {
    map(
        tuple((
            ws(tag("extend")),
            ws(char('$')),
            ws(parse_identifier),
            ws(parse_attributes0),
        )),
        |(_, _, variable, attributes)| ExtendEntityInference {
            variable: variable.to_string(),
            attributes,
        },
    )(input)
}
