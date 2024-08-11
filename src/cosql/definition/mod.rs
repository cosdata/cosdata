pub mod entity;
pub mod relationship;

use nom::{
    character::complete::char,
    combinator::map,
    multi::{separated_list0, separated_list1},
    sequence::tuple,
    IResult,
};

use super::{
    common::{parse_identifier, ws},
    data_type::parse_data_type,
    DataType,
};
pub use entity::EntityDefinition;
pub use relationship::RelationshipDefinition;

pub type AttributeDefinitions = Vec<AttributeDefinition>;

#[derive(Debug, Clone)]
pub struct AttributeDefinition {
    pub name: String,
    pub data_type: DataType,
}

pub fn parse_attribute_definitions0(input: &str) -> IResult<&str, AttributeDefinitions> {
    separated_list0(ws(char(',')), parse_attribute_definition)(input)
}

pub fn parse_attribute_definitions1(input: &str) -> IResult<&str, AttributeDefinitions> {
    separated_list1(ws(char(',')), parse_attribute_definition)(input)
}

pub fn parse_attribute_definition(input: &str) -> IResult<&str, AttributeDefinition> {
    map(
        tuple((ws(parse_identifier), ws(char(':')), ws(parse_data_type))),
        |(name, _, data_type)| AttributeDefinition {
            name: name.to_string(),
            data_type,
        },
    )(input)
}
