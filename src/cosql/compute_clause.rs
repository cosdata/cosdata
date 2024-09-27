use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::map,
    multi::separated_list1,
    sequence::{preceded, tuple},
    IResult,
};

use super::{
    common::{parse_variable, ws},
    expression::parse_expression,
    Expression,
};

pub type ComputeClauses = Vec<ComputeClause>;

#[derive(Debug, Clone, PartialEq)]
pub struct ComputeClause {
    pub variable: String,
    pub expression: Expression,
}

pub fn parse_compute_clauses(input: &str) -> IResult<&str, ComputeClauses> {
    preceded(
        ws(tag("compute")),
        separated_list1(ws(char(',')), parse_compute_clause),
    )(input)
}

pub fn parse_compute_clause(input: &str) -> IResult<&str, ComputeClause> {
    map(
        tuple((ws(parse_variable), ws(char('=')), parse_expression)),
        |(variable, _, expression)| ComputeClause {
            variable: variable.to_string(),
            expression,
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use crate::cosql::{
        expression::{BinaryExpression, BinaryExpressionOperator},
        Value,
    };

    use super::*;

    #[test]
    fn test_compute_clause_parser() {
        let values = [
            (
                "$profit = $selling_price - $cost_price",
                ComputeClause {
                    variable: "profit".to_string(),
                    expression: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::Value(Value::Variable("selling_price".to_string())),
                        operator: BinaryExpressionOperator::Subtraction,
                        right: Expression::Value(Value::Variable("cost_price".to_string())),
                    })),
                },
            ),
            (
                "$profit_percentage = ($profit / $cost_price) * 100",
                ComputeClause {
                    variable: "profit_percentage".to_string(),
                    expression: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::BinaryExpression(Box::new(BinaryExpression {
                            left: Expression::Value(Value::Variable("profit".to_string())),
                            operator: BinaryExpressionOperator::Division,
                            right: Expression::Value(Value::Variable("cost_price".to_string())),
                        })),
                        operator: BinaryExpressionOperator::Multiplication,
                        right: Expression::Value(Value::Int(100)),
                    })),
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_compute_clause(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_compute_clauses_parser() {
        let values = [(
            "compute
                $profit = $selling_price - $cost_price,
                $profit_percentage = ($profit / $cost_price) * 100",
            vec![
                ComputeClause {
                    variable: "profit".to_string(),
                    expression: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::Value(Value::Variable("selling_price".to_string())),
                        operator: BinaryExpressionOperator::Subtraction,
                        right: Expression::Value(Value::Variable("cost_price".to_string())),
                    })),
                },
                ComputeClause {
                    variable: "profit_percentage".to_string(),
                    expression: Expression::BinaryExpression(Box::new(BinaryExpression {
                        left: Expression::BinaryExpression(Box::new(BinaryExpression {
                            left: Expression::Value(Value::Variable("profit".to_string())),
                            operator: BinaryExpressionOperator::Division,
                            right: Expression::Value(Value::Variable("cost_price".to_string())),
                        })),
                        operator: BinaryExpressionOperator::Multiplication,
                        right: Expression::Value(Value::Int(100)),
                    })),
                },
            ],
        )];

        for (source, expected) in values {
            let (_, parsed) = parse_compute_clauses(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
