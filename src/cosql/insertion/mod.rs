pub mod entity;
pub mod relationship;

use nom::{
    character::complete::char,
    combinator::map,
    multi::{separated_list0, separated_list1},
    sequence::{delimited, tuple},
    IResult,
};

use super::{
    common::{parse_identifier, ws},
    value::parse_value,
    Value,
};
pub use entity::EntityInsertion;
pub use relationship::RelationshipInsertion;

pub type Attributes = Vec<Attribute>;

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub value: Value,
}

pub fn parse_attributes0(input: &str) -> IResult<&str, Attributes> {
    delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_attribute),
        ws(char(')')),
    )(input)
}

pub fn parse_attributes1(input: &str) -> IResult<&str, Attributes> {
    delimited(
        ws(char('(')),
        separated_list1(ws(char(',')), parse_attribute),
        ws(char(')')),
    )(input)
}

pub fn parse_attribute(input: &str) -> IResult<&str, Attribute> {
    map(
        tuple((ws(parse_identifier), ws(char(':')), ws(parse_value))),
        |(name, _, value)| Attribute {
            name: name.to_string(),
            value,
        },
    )(input)
}
