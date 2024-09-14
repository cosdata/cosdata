use nom::{bytes::complete::tag, combinator::map, sequence::tuple, IResult};

use crate::cosql::{
    common::{parse_variable, ws},
    insertion::{parse_attributes0, Attributes},
};

#[derive(Debug, Clone)]
pub struct ExtendEntityInference {
    pub variable: String,
    pub attributes: Attributes,
}

pub fn parse_extend_entity_inference(input: &str) -> IResult<&str, ExtendEntityInference> {
    map(
        tuple((ws(tag("extend")), ws(parse_variable), ws(parse_attributes0))),
        |(_, variable, attributes)| ExtendEntityInference {
            variable: variable.to_string(),
            attributes,
        },
    )(input)
}
