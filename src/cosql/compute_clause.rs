use nom::{
    bytes::complete::{tag, take_while1},
    character::complete::char,
    combinator::map,
    multi::separated_list1,
    sequence::{preceded, tuple},
    IResult,
};

use super::common::{parse_identifier, ws};

pub type ComputeClauses = Vec<ComputeClause>;

#[derive(Debug, Clone)]
pub struct ComputeClause {
    pub variable: String,
    pub expression: String,
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
            take_while1(|c| c != ',' && c != '\n'),
        )),
        |(variable, _, expression)| ComputeClause {
            variable: variable.to_string(),
            expression: expression.trim().to_string(),
        },
    )(input)
}
