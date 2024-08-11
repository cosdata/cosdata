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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_parser() {
        let values = [
            (r#""Hello, Rust""#, Value::String("Hello, Rust".to_string())),
            ("23", Value::Int(23)),
            ("-756", Value::Int(-756)),
            ("2345.12", Value::Double(2345.12)),
            ("-765.2", Value::Double(-765.2)),
            ("11-8-2024", Value::Date("11-8-2024".to_string())),
            ("true", Value::Boolean(true)),
            ("false", Value::Boolean(false)),
            ("$var_name", Value::Variable("var_name".to_string())),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_value(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
