use nom::{
    bytes::complete::tag, character::complete::char, combinator::map, sequence::tuple, IResult,
};

use crate::cosql::{
    common::{parse_identifier, ws},
    insertion::{parse_attributes0, Attributes},
};

#[derive(Debug, Clone)]
pub struct ExtendEntityInference {
    pub variable: String,
    pub attributes: Attributes,
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
