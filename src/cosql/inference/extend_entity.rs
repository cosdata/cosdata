use nom::{bytes::complete::tag, combinator::map, sequence::tuple, IResult};

use crate::cosql::{
    common::{parse_variable, ws},
    insertion::{parse_attributes0, Attributes},
};

#[derive(Debug, Clone, PartialEq)]
pub struct ExtendEntityInference {
    pub variable: String,
    pub attributes: Attributes,
}

pub fn parse_extend_entity_inference(input: &str) -> IResult<&str, ExtendEntityInference> {
    map(
        tuple((ws(tag("extend")), ws(parse_variable), ws(parse_attributes0))),
        |(_, variable, attributes)| ExtendEntityInference {
            variable: variable.to_string(),
            attributes,
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosql::{insertion::Attribute, Value};

    #[test]
    fn test_extend_entity_inference_parser() {
        let values = [
            (
                "extend $person1 (
                    salary: 100000
                )",
                ExtendEntityInference {
                    variable: "person1".to_string(),
                    attributes: vec![Attribute {
                        name: "salary".to_string(),
                        value: Value::Int(100000),
                    }],
                },
            ),
            (
                "extend $city1 (
                    visitors: $visitors
                )",
                ExtendEntityInference {
                    variable: "city1".to_string(),
                    attributes: vec![Attribute {
                        name: "visitors".to_string(),
                        value: Value::Variable("visitors".to_string()),
                    }],
                },
            ),
        ];

        for (source, expected) in values {
            let (_, parsed) = parse_extend_entity_inference(source).unwrap();

            assert_eq!(parsed, expected);
        }
    }
}
