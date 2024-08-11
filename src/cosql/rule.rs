use nom::{branch::alt, bytes::complete::tag, character::complete::char, combinator::map, IResult};

use super::{
    common::{parse_identifier, ws},
    inference::parse_inferences1,
    pattern::parse_patterns0,
    ComputeClauses, Inferences, Patterns,
};

#[derive(Debug, Clone)]
pub enum InferenceType {
    Derive,
    Materialize,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub patterns: Patterns,
    pub compute_clauses: Option<ComputeClauses>,
    pub inference_type: InferenceType,
    pub inferences: Inferences,
}

pub fn parse_inference_type(input: &str) -> IResult<&str, InferenceType> {
    alt((
        map(tag("derive"), |_| InferenceType::Derive),
        map(tag("materialize"), |_| InferenceType::Materialize),
    ))(input)
}

pub fn parse_rule(input: &str) -> IResult<&str, Rule> {
    let (input, name) = ws(parse_identifier)(input)?;
    let (input, _) = ws(tag("as"))(input)?;
    let (input, _) = ws(tag("match"))(input)?;

    let (input, patterns) = parse_patterns0(input)?;

    let (input, _) = ws(tag("infer"))(input)?;
    let (input, inference_type) = parse_inference_type(input)?;
    let (input, inferences) = parse_inferences1(input)?;
    let (input, _) = ws(char(';'))(input)?;

    let rule = Rule {
        name: name.to_string(),
        patterns,
        compute_clauses: None,
        inference_type,
        inferences,
    };

    Ok((input, rule))
}
