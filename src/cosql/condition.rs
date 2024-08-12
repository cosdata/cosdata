use nom::{
    branch::alt, bytes::complete::tag, character::complete::char, combinator::map, sequence::tuple,
    IResult,
};

use super::{
    common::{parse_variable, ws},
    value::parse_value,
    Value,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    Binary(BinaryCondition),
    Logical(Box<LogicalCondition>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOperator {
    Equality,
    Inequality,
    LessThan,
    LessEqualThan,
    GreaterThan,
    GreaterEqualThan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryCondition {
    pub left: String,
    pub operator: BinaryOperator,
    pub right: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicalOperator {
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalCondition {
    pub left: Condition,
    pub operator: LogicalOperator,
    pub right: Condition,
}

pub fn parse_binary_operator(input: &str) -> IResult<&str, BinaryOperator> {
    alt((
        map(tag("=="), |_| BinaryOperator::Equality),
        map(tag("!="), |_| BinaryOperator::Inequality),
        map(tag("<="), |_| BinaryOperator::LessEqualThan),
        map(tag("<"), |_| BinaryOperator::LessThan),
        map(tag(">="), |_| BinaryOperator::GreaterEqualThan),
        map(tag(">"), |_| BinaryOperator::GreaterThan),
    ))(input)
}

pub fn parse_binary_condition(input: &str) -> IResult<&str, BinaryCondition> {
    map(
        tuple((parse_variable, ws(parse_binary_operator), parse_value)),
        |(left, operator, right)| BinaryCondition {
            left: left.to_string(),
            operator,
            right,
        },
    )(input)
}

pub fn parse_logical_operator(input: &str) -> IResult<&str, LogicalOperator> {
    alt((
        map(tag("and"), |_| LogicalOperator::And),
        map(tag("or"), |_| LogicalOperator::Or),
    ))(input)
}

pub fn continue_parsing_condition(input: &str, left: Condition) -> IResult<&str, Condition> {
    if let Ok((input, (operator, right))) =
        tuple((ws(parse_logical_operator), parse_condition))(input)
    {
        Ok((
            input,
            Condition::Logical(Box::new(LogicalCondition {
                left,
                operator,
                right,
            })),
        ))
    } else {
        Ok((input, left))
    }
}

pub fn parse_condition(input: &str) -> IResult<&str, Condition> {
    let (input, first) = alt((
        map(parse_binary_condition, |bc| Condition::Binary(bc)),
        map(
            tuple((char('('), ws(parse_condition), char(')'))),
            |(_, condition, _)| condition,
        ),
    ))(input)?;

    continue_parsing_condition(input, first)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_operator_parser() {
        let values = [
            ("==", BinaryOperator::Equality),
            ("!=", BinaryOperator::Inequality),
            ("<=", BinaryOperator::LessEqualThan),
            ("<", BinaryOperator::LessThan),
            (">=", BinaryOperator::GreaterEqualThan),
            (">", BinaryOperator::GreaterThan),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_binary_operator(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_binary_condition_parser() {
        let values = [
            (
                r#"$programming_language == "Rust""#,
                BinaryCondition {
                    left: "programming_language".to_string(),
                    operator: BinaryOperator::Equality,
                    right: Value::String("Rust".to_string()),
                },
            ),
            (
                "$salary >= 1000000",
                BinaryCondition {
                    left: "salary".to_string(),
                    operator: BinaryOperator::GreaterEqualThan,
                    right: Value::Int(1000000),
                },
            ),
            (
                "$age < 18",
                BinaryCondition {
                    left: "age".to_string(),
                    operator: BinaryOperator::LessThan,
                    right: Value::Int(18),
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_binary_condition(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_logical_operator_parser() {
        let values = [("and", LogicalOperator::And), ("or", LogicalOperator::Or)];

        for (source, expected) in values {
            let (_, parsed) = parse_logical_operator(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_condition_parser() {
        let values = [
            (
                "$age < 18",
                Condition::Binary(BinaryCondition {
                    left: "age".to_string(),
                    operator: BinaryOperator::LessThan,
                    right: Value::Int(18),
                }),
            ),
            (
                r#"(((($programming_language == "Rust"))))"#,
                Condition::Binary(BinaryCondition {
                    left: "programming_language".to_string(),
                    operator: BinaryOperator::Equality,
                    right: Value::String("Rust".to_string()),
                }),
            ),
            (
                r#"$programming_language == "Rust" and $salary >= 1000000"#,
                Condition::Logical(Box::new(LogicalCondition {
                    left: Condition::Binary(BinaryCondition {
                        left: "programming_language".to_string(),
                        operator: BinaryOperator::Equality,
                        right: Value::String("Rust".to_string()),
                    }),
                    operator: LogicalOperator::And,
                    right: Condition::Binary(BinaryCondition {
                        left: "salary".to_string(),
                        operator: BinaryOperator::GreaterEqualThan,
                        right: Value::Int(1000000),
                    }),
                })),
            ),
            (
                r#"($programming_language == "Rust" and $salary >= 1000000) or $age < 18"#,
                Condition::Logical(Box::new(LogicalCondition {
                    left: Condition::Logical(Box::new(LogicalCondition {
                        left: Condition::Binary(BinaryCondition {
                            left: "programming_language".to_string(),
                            operator: BinaryOperator::Equality,
                            right: Value::String("Rust".to_string()),
                        }),
                        operator: LogicalOperator::And,
                        right: Condition::Binary(BinaryCondition {
                            left: "salary".to_string(),
                            operator: BinaryOperator::GreaterEqualThan,
                            right: Value::Int(1000000),
                        }),
                    })),
                    operator: LogicalOperator::Or,
                    right: Condition::Binary(BinaryCondition {
                        left: "age".to_string(),
                        operator: BinaryOperator::LessThan,
                        right: Value::Int(18),
                    }),
                })),
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_condition(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
