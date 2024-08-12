use nom::{bytes::complete::tag, character::complete::char, multi::separated_list1, IResult};

use super::{
    common::{parse_variable, ws},
    pattern::parse_patterns0,
    Patterns,
};

#[derive(Debug, Clone)]
pub struct Query {
    pub patterns: Patterns,
    pub get_variables: Vec<String>,
}

pub fn parse_query(input: &str) -> IResult<&str, Query> {
    let (input, patterns) = parse_patterns0(input)?;

    let (input, _) = ws(tag("get"))(input)?;
    let (input, get_variables) = separated_list1(ws(char(',')), parse_variable)(input)?;
    let (input, _) = ws(char(';'))(input)?;

    let query = Query {
        patterns,
        get_variables: get_variables.into_iter().map(|s| s.to_string()).collect(),
    };

    Ok((input, query))
}
