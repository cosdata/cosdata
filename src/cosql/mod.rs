pub mod common;
pub mod compute_clause;
pub mod condition;
pub mod data_type;
pub mod definition;
pub mod inference;
pub mod insertion;
pub mod pattern;
pub mod query;
pub mod rule;
pub mod value;

use common::ws_tag;
use nom::{branch::alt, combinator::map, multi::many0, sequence::preceded, IResult};

use definition::{
    entity::parse_entity_definition, relationship::parse_relationship_definition, EntityDefinition,
    RelationshipDefinition,
};
use insertion::{
    entity::parse_entity_insertion, relationship::parse_relationship_insertion, EntityInsertion,
    RelationshipInsertion,
};
use query::{parse_query, Query};
use rule::{parse_rule, Rule};

pub use compute_clause::{ComputeClause, ComputeClauses};
pub use data_type::DataType;
pub use inference::{Inference, Inferences};
pub use pattern::{Pattern, Patterns};
pub use value::Value;

pub type CosQLStatements = Vec<CosQLStatement>;

#[derive(Debug, Clone)]
pub enum CosQLStatement {
    EntityDefinition(EntityDefinition),
    RelationshipDefinition(RelationshipDefinition),
    EntityInsertion(EntityInsertion),
    RelationshipInsertion(RelationshipInsertion),
    Query(Query),
    Rule(Rule),
}

pub fn parse_cosql_statements(input: &str) -> IResult<&str, CosQLStatements> {
    many0(parse_cosql_statement)(input)
}

pub fn parse_cosql_statement(input: &str) -> IResult<&str, CosQLStatement> {
    alt((
        preceded(
            ws_tag("define"),
            alt((
                preceded(
                    ws_tag("entity"),
                    map(parse_entity_definition, |ed| {
                        CosQLStatement::EntityDefinition(ed)
                    }),
                ),
                preceded(
                    ws_tag("relationship"),
                    map(parse_relationship_definition, |rd| {
                        CosQLStatement::RelationshipDefinition(rd)
                    }),
                ),
                preceded(ws_tag("rule"), map(parse_rule, |r| CosQLStatement::Rule(r))),
            )),
        ),
        preceded(
            ws_tag("insert"),
            alt((
                map(parse_entity_insertion, |ei| {
                    CosQLStatement::EntityInsertion(ei)
                }),
                map(parse_relationship_insertion, |ri| {
                    CosQLStatement::RelationshipInsertion(ri)
                }),
            )),
        ),
        preceded(
            ws_tag("match"),
            map(parse_query, |q| CosQLStatement::Query(q)),
        ),
    ))(input)
}
