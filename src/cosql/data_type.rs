use nom::{branch::alt, bytes::complete::tag, combinator::map, IResult};

#[derive(Debug, Clone)]
pub enum DataType {
    String,
    Int,
    Double,
    Date,
    Boolean,
}

pub fn parse_data_type(input: &str) -> IResult<&str, DataType> {
    alt((
        map(tag("string"), |_| DataType::String),
        map(tag("int"), |_| DataType::Int),
        map(tag("double"), |_| DataType::Double),
        map(tag("date"), |_| DataType::Date),
        map(tag("boolean"), |_| DataType::Boolean),
    ))(input)
}
