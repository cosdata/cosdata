use nom::{
    branch::alt, bytes::complete::tag, character::complete::char, combinator::map, sequence::tuple,
    IResult,
};

use super::{
    common::ws,
    condition::{parse_logical_operator, LogicalOperator},
    value::parse_value,
    Precedence, Value,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Value(Value),
    BinaryExpression(Box<BinaryExpression>),
    UnaryExpression(Box<UnaryExpression>),
    LogicalExpression(Box<LogicalExpression>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryExpressionOperator {
    // ==
    Equality,
    // !=
    Inequality,
    // <
    LessThan,
    // <=
    LessEqualThan,
    // >
    GreaterThan,
    // >=
    GreaterEqualThan,
    // +
    Addition,
    // -
    Subtraction,
    // *
    Multiplication,
    // /
    Division,
    // %
    Remainder,
    // **
    Exponential,
}

impl BinaryExpressionOperator {
    pub fn precedence(&self) -> Precedence {
        match self {
            Self::Equality | Self::Inequality => Precedence::Equals,
            Self::LessThan | Self::LessEqualThan | Self::GreaterThan | Self::GreaterEqualThan => {
                Precedence::Compare
            }
            Self::Addition | Self::Subtraction => Precedence::Add,
            Self::Multiplication | Self::Division | Self::Remainder => Precedence::Multiply,
            Self::Exponential => Precedence::Exponentiation,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryExpression {
    pub left: Expression,
    pub operator: BinaryExpressionOperator,
    pub right: Expression,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOperator {
    // -
    Negation,
    // !
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryExpression {
    pub operator: UnaryOperator,
    pub argument: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalExpression {
    pub left: Expression,
    pub operator: LogicalOperator,
    pub right: Expression,
}

pub fn parse_unary_operator(input: &str) -> IResult<&str, UnaryOperator> {
    alt((
        map(char('-'), |_| UnaryOperator::Negation),
        map(char('!'), |_| UnaryOperator::Not),
    ))(input)
}

pub fn parse_binary_expression_operator(input: &str) -> IResult<&str, BinaryExpressionOperator> {
    alt((
        map(tag("=="), |_| BinaryExpressionOperator::Equality),
        map(tag("!="), |_| BinaryExpressionOperator::Inequality),
        map(tag("<="), |_| BinaryExpressionOperator::LessEqualThan),
        map(char('<'), |_| BinaryExpressionOperator::LessThan),
        map(tag(">="), |_| BinaryExpressionOperator::GreaterEqualThan),
        map(char('>'), |_| BinaryExpressionOperator::GreaterThan),
        map(tag("**"), |_| BinaryExpressionOperator::Exponential),
        map(char('+'), |_| BinaryExpressionOperator::Addition),
        map(char('-'), |_| BinaryExpressionOperator::Subtraction),
        map(char('*'), |_| BinaryExpressionOperator::Multiplication),
        map(char('/'), |_| BinaryExpressionOperator::Division),
        map(char('%'), |_| BinaryExpressionOperator::Remainder),
    ))(input)
}

pub fn parse_unary_expression_argument(input: &str) -> IResult<&str, Expression> {
    alt((map(parse_value, Expression::Value), parse_paren_expression))(input)
}

pub fn parse_unary_expression_or_higher(input: &str) -> IResult<&str, Expression> {
    if let Ok((input, expression)) = parse_unary_expression_argument(input) {
        return Ok((input, expression));
    }

    let (input, operator) = parse_unary_operator(input)?;
    let (input, argument) = parse_unary_expression_argument(input)?;

    Ok((
        input,
        Expression::UnaryExpression(Box::new(UnaryExpression { operator, argument })),
    ))
}

pub fn parse_binary_expression_or_highier(
    input: &str,
    lhs_precedence: Precedence,
) -> IResult<&str, Expression> {
    let (input, lhs) = parse_unary_expression_or_higher(input)?;

    parse_binary_expression_rest(input, lhs, lhs_precedence)
}

pub fn parse_binary_expression_rest(
    mut input: &str,
    lhs: Expression,
    min_precedence: Precedence,
) -> IResult<&str, Expression> {
    // Pratt Parsing Algorithm
    // <https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html>
    let mut lhs = lhs;
    loop {
        if let Ok((new_input, op)) = ws(parse_binary_expression_operator)(input) {
            let left_precedence = op.precedence();

            let stop = if left_precedence.is_right_associative() {
                left_precedence < min_precedence
            } else {
                left_precedence <= min_precedence
            };

            if stop {
                break;
            }

            input = new_input;

            let (new_input, rhs) = parse_binary_expression_or_highier(input, left_precedence)?;

            lhs = Expression::BinaryExpression(Box::new(BinaryExpression {
                left: lhs,
                operator: op,
                right: rhs,
            }));

            input = new_input;
        } else if let Ok((new_input, op)) = ws(parse_logical_operator)(input) {
            let left_precedence = op.precedence();

            let stop = if left_precedence.is_right_associative() {
                left_precedence < min_precedence
            } else {
                left_precedence <= min_precedence
            };

            if stop {
                break;
            }

            input = new_input;

            let (new_input, rhs) = parse_binary_expression_or_highier(input, left_precedence)?;

            lhs = Expression::LogicalExpression(Box::new(LogicalExpression {
                left: lhs,
                operator: op,
                right: rhs,
            }));

            input = new_input;
        } else {
            break;
        }
    }

    Ok((input, lhs))
}

pub fn parse_paren_expression(input: &str) -> IResult<&str, Expression> {
    map(
        tuple((char('('), ws(parse_expression), char(')'))),
        |(_, expression, _)| expression,
    )(input)
}

pub fn parse_expression(input: &str) -> IResult<&str, Expression> {
    parse_binary_expression_or_highier(input, Precedence::Lowest)
}

#[cfg(test)]
mod tests {
    use crate::cosql::Date;

    use super::*;

    #[test]
    fn test_unary_operator_parser() {
        let values = [("-", UnaryOperator::Negation), ("!", UnaryOperator::Not)];

        for (source, expected) in values {
            let (_, parsed) = parse_unary_operator(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_binary_expression_operator_parser() {
        let values = [
            ("==", BinaryExpressionOperator::Equality),
            ("!=", BinaryExpressionOperator::Inequality),
            ("<=", BinaryExpressionOperator::LessEqualThan),
            ("<", BinaryExpressionOperator::LessThan),
            (">=", BinaryExpressionOperator::GreaterEqualThan),
            (">", BinaryExpressionOperator::GreaterThan),
            ("+", BinaryExpressionOperator::Addition),
            ("-", BinaryExpressionOperator::Subtraction),
            ("*", BinaryExpressionOperator::Multiplication),
            ("/", BinaryExpressionOperator::Division),
            ("%", BinaryExpressionOperator::Remainder),
            ("**", BinaryExpressionOperator::Exponential),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_binary_expression_operator(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_unary_expression_parser() {
        let values = [
            (
                "!$variable",
                Expression::UnaryExpression(Box::new(UnaryExpression {
                    operator: UnaryOperator::Not,
                    argument: Expression::Value(Value::Variable("variable".to_string())),
                })),
            ),
            (
                "-$age",
                Expression::UnaryExpression(Box::new(UnaryExpression {
                    operator: UnaryOperator::Negation,
                    argument: Expression::Value(Value::Variable("age".to_string())),
                })),
            ),
            ("-2.0", Expression::Value(Value::Double(-2.0))),
            (
                "-(2 + 3 * 4)",
                Expression::UnaryExpression(Box::new(UnaryExpression {
                    operator: UnaryOperator::Negation,
                    argument: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::Value(Value::Int(2)),
                        operator: BinaryExpressionOperator::Addition,
                        right: Expression::BinaryExpression(Box::new(BinaryExpression {
                            left: Expression::Value(Value::Int(3)),
                            operator: BinaryExpressionOperator::Multiplication,
                            right: Expression::Value(Value::Int(4)),
                        })),
                    })),
                })),
            ),
            (
                "-(2 * 3 + 4)",
                Expression::UnaryExpression(Box::new(UnaryExpression {
                    operator: UnaryOperator::Negation,
                    argument: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::BinaryExpression(Box::new(BinaryExpression {
                            left: Expression::Value(Value::Int(2)),
                            operator: BinaryExpressionOperator::Multiplication,
                            right: Expression::Value(Value::Int(3)),
                        })),
                        operator: BinaryExpressionOperator::Addition,
                        right: Expression::Value(Value::Int(4)),
                    })),
                })),
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_unary_expression_or_higher(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_expression_parser() {
        let values = [
            ("-2.0", Expression::Value(Value::Double(-2.0))),
            (
                "$num % 2",
                Expression::BinaryExpression(Box::new(BinaryExpression {
                    left: Expression::Value(Value::Variable("num".to_string())),
                    operator: BinaryExpressionOperator::Remainder,
                    right: Expression::Value(Value::Int(2)),
                })),
            ),
            (
                "2 + 3 * 4",
                Expression::BinaryExpression(Box::new(BinaryExpression {
                    left: Expression::Value(Value::Int(2)),
                    operator: BinaryExpressionOperator::Addition,
                    right: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::Value(Value::Int(3)),
                        operator: BinaryExpressionOperator::Multiplication,
                        right: Expression::Value(Value::Int(4)),
                    })),
                })),
            ),
            (
                "2 * 3 + 4",
                Expression::BinaryExpression(Box::new(BinaryExpression {
                    left: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::Value(Value::Int(2)),
                        operator: BinaryExpressionOperator::Multiplication,
                        right: Expression::Value(Value::Int(3)),
                    })),
                    operator: BinaryExpressionOperator::Addition,
                    right: Expression::Value(Value::Int(4)),
                })),
            ),
            (
                "((((((((((26-08-2024))))))))))",
                Expression::Value(Value::Date(Date(26, 8, 2024))),
            ),
            (
                "$selling_price - $cost_price",
                Expression::BinaryExpression(Box::new(BinaryExpression {
                    left: Expression::Value(Value::Variable("selling_price".to_string())),
                    operator: BinaryExpressionOperator::Subtraction,
                    right: Expression::Value(Value::Variable("cost_price".to_string())),
                })),
            ),
            (
                "($profit / $cost_price) * 100",
                Expression::BinaryExpression(Box::new(BinaryExpression {
                    left: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::Value(Value::Variable("profit".to_string())),
                        operator: BinaryExpressionOperator::Division,
                        right: Expression::Value(Value::Variable("cost_price".to_string())),
                    })),
                    operator: BinaryExpressionOperator::Multiplication,
                    right: Expression::Value(Value::Int(100)),
                })),
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_expression(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
