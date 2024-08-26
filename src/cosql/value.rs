use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1},
    combinator::{map, map_res, opt, recognize},
    sequence::{pair, tuple},
    IResult,
};

use super::common::{parse_string_literal, parse_variable};

use std::num::{ParseFloatError, ParseIntError};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Int(i64),
    Double(f64),
    Date(Date),
    Boolean(bool),
    Variable(String),
}

// MM/DD/YYYY
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Date(pub u8, pub u8, pub u16);

pub fn parse_date(input: &str) -> IResult<&str, Date> {
    let (input, (d, _, m, _, y)) = tuple((
        map_res(digit1::<&str, _>, str::parse),
        char('-'),
        map_res(digit1::<&str, _>, str::parse),
        char('-'),
        map_res(digit1::<&str, _>, str::parse),
    ))(input)?;

    Ok((input, Date(d, m, y)))
}

pub fn parse_value(input: &str) -> IResult<&str, Value> {
    alt((
        map(parse_string_literal, |s| Value::String(s.to_string())),
        map(parse_date, |date| Value::Date(date)),
        map_res(
            recognize(tuple((opt(char('-')), digit1, char('.'), digit1))),
            |s: &str| Ok::<_, ParseFloatError>(Value::Double(s.parse()?)),
        ),
        map_res(recognize(pair(opt(char('-')), digit1)), |s: &str| {
            Ok::<_, ParseIntError>(Value::Int(s.parse()?))
        }),
        map(tag("true"), |_| Value::Boolean(true)),
        map(tag("false"), |_| Value::Boolean(false)),
        map(parse_variable, |s| Value::Variable(s.to_string())),
    ))(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_parser() {
        let values = [
            ("01-01-1970", Date(1, 1, 1970)),
            ("01-01-2000", Date(1, 1, 2000)),
            ("31-12-2009", Date(31, 12, 2009)),
            ("01-01-1970", Date(1, 1, 1970)),
            ("01-01-1970", Date(1, 1, 1970)),
            ("01-01-1970", Date(1, 1, 1970)),
            ("01-01-1970", Date(1, 1, 1970)),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_date(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_value_parser() {
        let values = [
            (r#""Hello, Rust""#, Value::String("Hello, Rust".to_string())),
            ("23", Value::Int(23)),
            ("-756", Value::Int(-756)),
            ("2345.12", Value::Double(2345.12)),
            ("-765.2", Value::Double(-765.2)),
            ("11-8-2024", Value::Date(Date(11, 8, 2024))),
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
