use nom::{bytes::complete::tag, combinator::map, sequence::tuple, IResult};

use crate::cosql::{
    common::{parse_identifier, parse_variable, ws},
    insertion::{parse_attributes0, Attributes},
};

#[derive(Debug, Clone)]
pub struct EntityInference {
    pub variable: String,
    pub entity_type: String,
    pub attributes: Attributes,
}

pub fn parse_entity_inference(input: &str) -> IResult<&str, EntityInference> {
    map(
        tuple((
            ws(parse_variable),
            ws(tag("isa")),
            ws(parse_identifier),
            ws(parse_attributes0),
        )),
        |(variable, _, entity_type, attributes)| EntityInference {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes,
        },
    )(input)
}
