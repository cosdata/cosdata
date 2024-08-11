use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1},
    combinator::{map, opt, recognize},
    sequence::{pair, preceded, tuple},
    IResult,
};

use super::common::{parse_identifier, parse_string_literal};

#[derive(Debug, Clone)]
pub enum Value {
    String(String),
    Int(i64),
    Double(f64),
    Date(String),
    Boolean(bool),
    Variable(String),
}

pub fn parse_value(input: &str) -> IResult<&str, Value> {
    alt((
        map(parse_string_literal, |s| Value::String(s.to_string())),
        map(recognize(pair(opt(char('-')), digit1)), |s: &str| {
            Value::Int(s.parse().unwrap())
        }),
        map(
            recognize(tuple((opt(char('-')), digit1, char('.'), digit1))),
            |s: &str| Value::Double(s.parse().unwrap()),
        ),
        map(
            recognize(tuple((
                digit1::<&str, _>,
                char('-'),
                digit1::<&str, _>,
                char('-'),
                digit1::<&str, _>,
            ))),
            |s| Value::Date(s.to_string()),
        ),
        map(tag("true"), |_| Value::Boolean(true)),
        map(tag("false"), |_| Value::Boolean(false)),
        map(preceded(char('$'), parse_identifier), |s| {
            Value::Variable(s.to_string())
        }),
    ))(input)
}
