use nom::{branch::alt, bytes::complete::tag, combinator::map, IResult};

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_parser() {
        let values = [
            ("string", DataType::String),
            ("int", DataType::Int),
            ("double", DataType::Double),
            ("date", DataType::Date),
            ("boolean", DataType::Boolean),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_data_type(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
