pub mod entity;
pub mod relationship;

use nom::{
    branch::alt, character::complete::char, combinator::map, multi::separated_list0, IResult,
};

use entity::{parse_entity_pattern, EntityPattern};
use relationship::{parse_relationship_pattern, RelationshipPattern};

use super::{
    common::ws,
    condition::{parse_condition, Condition},
};

pub type Patterns = Vec<Pattern>;

#[derive(Debug, Clone)]
pub enum Pattern {
    EntityPattern(EntityPattern),
    RelationshipPattern(RelationshipPattern),
    Condition(Condition),
}

pub fn parse_patterns0(input: &str) -> IResult<&str, Patterns> {
    separated_list0(ws(char(',')), parse_pattern)(input)
}

pub fn parse_patterns1(input: &str) -> IResult<&str, Patterns> {
    separated_list0(ws(char(',')), parse_pattern)(input)
}

pub fn parse_pattern(input: &str) -> IResult<&str, Pattern> {
    alt((
        map(parse_entity_pattern, |ep| Pattern::EntityPattern(ep)),
        map(parse_relationship_pattern, |rp| {
            Pattern::RelationshipPattern(rp)
        }),
        map(parse_condition, |c| Pattern::Condition(c)),
    ))(input)
}
