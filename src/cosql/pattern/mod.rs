pub mod entity;
pub mod relationship;

use nom::{
    branch::alt, character::complete::char, combinator::map, multi::separated_list0, IResult,
};

use entity::{parse_entity_pattern, EntityPattern};
use relationship::{parse_relationship_pattern, RelationshipPattern};

use super::common::ws;

pub type Patterns = Vec<Pattern>;

#[derive(Debug, Clone)]
pub enum Pattern {
    EntityPattern(EntityPattern),
    RelationshipPattern(RelationshipPattern),
    Condition(String),
}

pub fn parse_patterns0(input: &str) -> IResult<&str, Patterns> {
    separated_list0(ws(char(',')), parse_pattern)(input)
}

pub fn parse_patterns1(input: &str) -> IResult<&str, Patterns> {
    separated_list0(ws(char(',')), parse_pattern)(input)
}

pub fn parse_pattern(input: &str) -> IResult<&str, Pattern> {
    alt((
        map(parse_entity_pattern, |p| Pattern::EntityPattern(p)),
        map(parse_relationship_pattern, |p| {
            Pattern::RelationshipPattern(p)
        }),
    ))(input)
}
