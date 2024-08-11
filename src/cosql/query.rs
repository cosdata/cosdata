use nom::{bytes::complete::tag, character::complete::char, multi::separated_list1, IResult};

use super::{
    common::{parse_variable, ws},
    pattern::parse_patterns0,
    Pattern, Patterns,
};

#[derive(Debug, Clone)]
pub struct Query {
    pub patterns: Patterns,
    pub conditions: Vec<String>,
    pub get_variables: Vec<String>,
}

pub fn parse_query(input: &str) -> IResult<&str, Query> {
    let (input, patterns_and_conditions) = parse_patterns0(input)?;

    let mut patterns = Vec::new();
    let mut conditions = Vec::new();

    for pattern_or_condition in patterns_and_conditions {
        if let Pattern::Condition(condition) = pattern_or_condition {
            conditions.push(condition);
        } else {
            patterns.push(pattern_or_condition);
        }
    }

    let (input, _) = ws(tag("get"))(input)?;
    let (input, get_variables) = separated_list1(ws(char(',')), parse_variable)(input)?;
    let (input, _) = ws(char(';'))(input)?;

    let query = Query {
        patterns,
        conditions,
        get_variables: get_variables.into_iter().map(|s| s.to_string()).collect(),
    };

    Ok((input, query))
}
