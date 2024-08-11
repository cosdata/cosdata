pub mod entity;
pub mod relationship;

use nom::{
    branch::alt, character::complete::char, combinator::map, multi::separated_list1, IResult,
};

use super::common::ws;
use entity::{
    parse_entity_inference, parse_extend_entity_inference, EntityInference, ExtendEntityInference,
};
use relationship::{parse_relationship_inference, RelationshipInference};

pub type Inferences = Vec<Inference>;

#[derive(Debug, Clone)]
pub enum Inference {
    EntityInference(EntityInference),
    RelationshipInference(RelationshipInference),
    ExtendEntityInference(ExtendEntityInference),
}

pub fn parse_inferences0(input: &str) -> IResult<&str, Inferences> {
    separated_list1(ws(char(',')), parse_inference)(input)
}

pub fn parse_inferences1(input: &str) -> IResult<&str, Inferences> {
    separated_list1(ws(char(',')), parse_inference)(input)
}

pub fn parse_inference(input: &str) -> IResult<&str, Inference> {
    alt((
        map(parse_entity_inference, |i| Inference::EntityInference(i)),
        map(parse_extend_entity_inference, |i| {
            Inference::ExtendEntityInference(i)
        }),
        map(parse_relationship_inference, |i| {
            Inference::RelationshipInference(i)
        }),
    ))(input)
}
