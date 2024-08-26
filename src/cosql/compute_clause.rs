use nom::{
    bytes::complete::tag,
    character::complete::char,
    combinator::map,
    multi::separated_list1,
    sequence::{preceded, tuple},
    IResult,
};

use super::{
    common::{parse_identifier, ws},
    expression::parse_expression,
    Expression,
};

pub type ComputeClauses = Vec<ComputeClause>;

#[derive(Debug, Clone)]
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
        tuple((
            preceded(char('$'), ws(parse_identifier)),
            ws(char('=')),
            parse_expression,
        )),
        |(variable, _, expression)| ComputeClause {
            variable: variable.to_string(),
            expression,
        },
    )(input)
}
