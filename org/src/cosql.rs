use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0, multispace1},
    combinator::{map, opt, recognize},
    multi::{many0, many1, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};

use nom::{
    error::{context, ErrorKind, ParseError},
    Err,
};
// Common parsers (same as before)

fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn string_literal(input: &str) -> IResult<&str, &str> {
    delimited(char('"'), take_while1(|c| c != '"'), char('"'))(input)
}

fn ws<'a, F: 'a, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: Fn(&'a str) -> IResult<&'a str, O>,
{
    delimited(multispace0, inner, multispace0)
}

// Data Types and Values

#[derive(Debug, Clone)]
enum DataType {
    String,
    Int,
    Double,
    Date,
    Boolean,
}

fn data_type(input: &str) -> IResult<&str, DataType> {
    alt((
        map(tag("string"), |_| DataType::String),
        map(tag("int"), |_| DataType::Int),
        map(tag("double"), |_| DataType::Double),
        map(tag("date"), |_| DataType::Date),
        map(tag("boolean"), |_| DataType::Boolean),
    ))(input)
}

#[derive(Debug, Clone)]
enum Value {
    String(String),
    Int(i64),
    Double(f64),
    Date(String),
    Boolean(bool),
    Variable(String),
}

fn value(input: &str) -> IResult<&str, Value> {
    alt((
        map(string_literal, |s| Value::String(s.to_string())),
        map(recognize(pair(opt(char('-')), digit1)), |s: &str| {
            Value::Int(s.parse().unwrap())
        }),
        map(
            recognize(tuple((opt(char('-')), digit1, char('.'), digit1))),
            |s: &str| Value::Double(s.parse().unwrap()),
        ),
        map(
            recognize(tuple((
                digit1::<&str, _>,
                char('-'),
                digit1::<&str, _>,
                char('-'),
                digit1::<&str, _>,
            ))),
            |s| Value::Date(s.to_string()),
        ),
        map(tag("true"), |_| Value::Boolean(true)),
        map(tag("false"), |_| Value::Boolean(false)),
        map(preceded(char('$'), identifier), |s| {
            Value::Variable(s.to_string())
        }),
    ))(input)
}

// DDL Parsers

#[derive(Debug, Clone)]
struct Attribute {
    name: String,
    data_type: DataType,
}

fn attribute(input: &str) -> IResult<&str, Attribute> {
    map(
        tuple((ws(identifier), ws(char(':')), ws(data_type))),
        |(name, _, data_type)| Attribute {
            name: name.to_string(),
            data_type,
        },
    )(input)
}

#[derive(Debug)]
struct EntityDefinition {
    name: String,
    attributes: Vec<Attribute>,
}

fn entity_definition(input: &str) -> IResult<&str, EntityDefinition> {
    map(
        tuple((
            ws(identifier),
            ws(tag("as")),
            separated_list1(ws(char(',')), attribute),
            ws(char(';')),
        )),
        |(name, _, attributes, _)| EntityDefinition {
            name: name.to_string(),
            attributes,
        },
    )(input)
}

#[derive(Debug, Clone)]
struct Role {
    name: String,
    entity_type: String,
}

#[derive(Debug)]
struct RelationshipDefinition {
    name: String,
    roles: Vec<Role>,
    attributes: Vec<Attribute>,
}

fn role(input: &str) -> IResult<&str, Role> {
    map(
        tuple((ws(identifier), ws(char(':')), ws(identifier))),
        |(name, _, entity_type)| Role {
            name: name.to_string(),
            entity_type: entity_type.to_string(),
        },
    )(input)
}

fn relationship_definition(input: &str) -> IResult<&str, RelationshipDefinition> {
    map(
        tuple((
            ws(identifier),
            ws(tag("as")),
            delimited(
                ws(char('(')),
                separated_list1(ws(char(',')), role),
                ws(char(')')),
            ),
            opt(preceded(
                ws(char(',')),
                separated_list0(ws(char(',')), attribute),
            )),
            ws(char(';')),
        )),
        |(name, _, roles, attributes, _)| RelationshipDefinition {
            name: name.to_string(),
            roles,
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

// DML Parsers

#[derive(Debug, Clone)]
struct AttributeValue {
    name: String,
    value: Value,
}

fn attribute_value(input: &str) -> IResult<&str, AttributeValue> {
    map(
        tuple((ws(identifier), ws(char(':')), ws(value))),
        |(name, _, value)| AttributeValue {
            name: name.to_string(),
            value,
        },
    )(input)
}

#[derive(Debug)]
struct EntityInsertion {
    variable: String,
    entity_type: String,
    attributes: Vec<AttributeValue>,
}

fn entity_insertion(input: &str) -> IResult<&str, EntityInsertion> {
    map(
        tuple((
            ws(char('$')),
            ws(identifier),
            ws(tag("isa")),
            ws(identifier),
            delimited(
                ws(char('(')),
                separated_list1(ws(char(',')), attribute_value),
                ws(char(')')),
            ),
            ws(char(';')),
        )),
        |(_, variable, _, entity_type, attributes, _)| EntityInsertion {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes,
        },
    )(input)
}

#[derive(Debug, Clone)]
struct RoleAssignment {
    role: String,
    entity: String,
}

#[derive(Debug)]
struct RelationshipInsertion {
    variable: String,
    roles: Vec<RoleAssignment>,
    relationship_type: String,
    attributes: Vec<AttributeValue>,
}

fn role_assignment(input: &str) -> IResult<&str, RoleAssignment> {
    map(
        tuple((ws(identifier), ws(char(':')), ws(char('$')), ws(identifier))),
        |(role, _, _, entity)| RoleAssignment {
            role: role.to_string(),
            entity: entity.to_string(),
        },
    )(input)
}

fn relationship_insertion(input: &str) -> IResult<&str, RelationshipInsertion> {
    map(
        tuple((
            ws(char('$')),
            ws(identifier),
            delimited(
                ws(char('(')),
                separated_list1(ws(char(',')), role_assignment),
                ws(char(')')),
            ),
            ws(tag("forms")),
            ws(identifier),
            opt(delimited(
                ws(char('(')),
                separated_list0(ws(char(',')), attribute_value),
                ws(char(')')),
            )),
            ws(char(';')),
        )),
        |(_, variable, roles, _, relationship_type, attributes, _)| RelationshipInsertion {
            variable: variable.to_string(),
            roles,
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

// Query Parsers

#[derive(Debug, Clone)]
enum Pattern {
    EntityPattern {
        variable: String,
        entity_type: String,
        attributes: Vec<AttributeValue>,
    },
    RelationshipPattern {
        variable: Option<String>,
        roles: Vec<RoleAssignment>,
        relationship_type: String,
        attributes: Vec<AttributeValue>,
    },
    Condition(String), // Add condition as a variant
}

fn entity_pattern(input: &str) -> IResult<&str, Pattern> {
    map(
        tuple((
            ws(char('$')),
            ws(identifier),
            ws(tag("isa")),
            ws(identifier),
            opt(delimited(
                ws(char('(')),
                separated_list0(ws(char(',')), attribute_value),
                ws(char(')')),
            )),
        )),
        |(_, variable, _, entity_type, attributes)| Pattern::EntityPattern {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

fn relationship_pattern(input: &str) -> IResult<&str, Pattern> {
    map(
        tuple((
            opt(tuple((ws(char('$')), ws(identifier)))),
            delimited(
                ws(char('(')),
                separated_list1(ws(char(',')), role_assignment),
                ws(char(')')),
            ),
            ws(tag("forms")),
            ws(identifier),
            opt(delimited(
                ws(char('(')),
                separated_list0(ws(char(',')), attribute_value),
                ws(char(')')),
            )),
        )),
        |(variable, roles, _, relationship_type, attributes)| Pattern::RelationshipPattern {
            variable: variable.map(|(_, v)| v.to_string()),
            roles,
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

#[derive(Debug, Clone)]
struct Query {
    patterns: Vec<Pattern>,
    conditions: Vec<String>,
    get_variables: Vec<String>,
}
fn variable(input: &str) -> IResult<&str, &str> {
    preceded(char('$'), identifier)(input)
}

fn condition(input: &str) -> IResult<&str, String> {
    map(
        tuple((ws(variable), ws(tag(">")), ws(digit1))),
        |(var, _, value)| format!("{} > {}", var, value),
    )(input)
}

use either::Either;

fn query(input: &str) -> IResult<&str, Query> {
    println!("Attempting to parse query");
    println!("Input: {:?}", input);

    // Define a parser for either patterns or conditions
    let pattern_or_condition = alt((
        map(entity_pattern, |p| (p, None)),
        map(relationship_pattern, |p| (p, None)),
        map(condition, |c| (Pattern::Condition(c), None)),
    ));

    // Parse a list of patterns and/or conditions
    let (input, patterns_and_conditions) =
        separated_list0(ws(char(',')), pattern_or_condition)(input)?;

    // Separate patterns and conditions from the combined list
    let mut patterns = Vec::new();
    let mut conditions = Vec::new();

    for (pattern, condition) in patterns_and_conditions {
        if let Some(cond) = condition {
            conditions.push(cond);
        } else {
            patterns.push(pattern);
        }
    }
    println!("Parsed patterns: {:?}", patterns);
    println!("Remaining input: {:?}", input);

    // Parse "get" keyword
    let get_result = ws(tag("get"))(input);
    println!("Parsing 'get': {:?}", get_result);
    let (input, _) = get_result?;
    println!("Matched 'get' keyword");
    println!("Remaining input: {:?}", input);

    // Parse get variables
    let get_variables_result = separated_list1(ws(char(',')), variable)(input);
    println!("Parsing get variables: {:?}", get_variables_result);
    let (input, get_variables) = get_variables_result?;
    println!("Parsed get variables: {:?}", get_variables);
    println!("Remaining input: {:?}", input);

    // Parse semicolon
    let semicolon_result = ws(char(';'))(input);
    println!("Parsing semicolon: {:?}", semicolon_result);
    let (input, _) = semicolon_result?;
    println!("Matched semicolon");
    println!("Remaining input: {:?}", input);

    // Construct Query struct
    let query = Query {
        patterns,
        conditions,
        get_variables: get_variables.into_iter().map(|s| s.to_string()).collect(),
    };
    println!("Constructed Query: {:?}", query);

    Ok((input, query))
}

#[derive(Debug)]
struct ComputeClause {
    variable: String,
    expression: String,
}

// fn compute_clause(input: &str) -> IResult<&str, Vec<ComputeClause>> {
//     preceded(
//         ws(tag("compute")),
//         separated_list1(
//             ws(char(',')),
//             map(
//                 tuple((
//                     preceded(char('$'), ws(identifier)),
//                     ws(char('=')),
//                     take_while1(|c| c != ',' && c != '\n'),
//                 )),
//                 |(variable, _, expression)| ComputeClause {
//                     variable: variable.to_string(),
//                     expression: expression.trim().to_string(),
//                 },
//             ),
//         ),
//     )(input)
// }

#[derive(Debug)]
enum Inference {
    EntityInference {
        variable: String,
        entity_type: String,
        attributes: Vec<AttributeValue>,
    },
    RelationshipInference {
        roles: Vec<RoleAssignment>,
        relationship_type: String,
        attributes: Vec<AttributeValue>,
    },
    ExtendEntity {
        variable: String,
        attributes: Vec<AttributeValue>,
    },
}

fn entity_inference(input: &str) -> IResult<&str, Inference> {
    map(
        tuple((
            ws(char('$')),
            ws(identifier),
            ws(tag("isa")),
            ws(identifier),
            delimited(
                ws(char('(')),
                separated_list0(ws(char(',')), attribute_value),
                ws(char(')')),
            ),
        )),
        |(_, variable, _, entity_type, attributes)| Inference::EntityInference {
            variable: variable.to_string(),
            entity_type: entity_type.to_string(),
            attributes,
        },
    )(input)
}

fn relationship_inference(input: &str) -> IResult<&str, Inference> {
    map(
        tuple((
            delimited(
                ws(char('(')),
                separated_list1(ws(char(',')), preceded(char('$'), identifier)),
                ws(char(')')),
            ),
            ws(tag("forms")),
            ws(identifier),
            opt(delimited(
                ws(char('(')),
                separated_list0(ws(char(',')), attribute_value),
                ws(char(')')),
            )),
        )),
        |(roles, _, relationship_type, attributes)| Inference::RelationshipInference {
            roles: roles
                .into_iter()
                .map(|r| RoleAssignment {
                    role: r.to_string(),
                    entity: r.to_string(),
                })
                .collect(),
            relationship_type: relationship_type.to_string(),
            attributes: attributes.unwrap_or_default(),
        },
    )(input)
}

fn extend_entity_inference(input: &str) -> IResult<&str, Inference> {
    map(
        tuple((
            ws(tag("extend")),
            ws(char('$')),
            ws(identifier),
            delimited(
                ws(char('(')),
                separated_list1(ws(char(',')), attribute_value),
                ws(char(')')),
            ),
        )),
        |(_, _, variable, attributes)| Inference::ExtendEntity {
            variable: variable.to_string(),
            attributes,
        },
    )(input)
}

fn rule(input: &str) -> IResult<&str, Rule> {
    println!("Starting to parse rule {}", input);

    let (input, name) = ws(identifier)(input)?;
    println!("Parsed rule name: {}", name);
    let (input, _) = ws(tag("as"))(input)?;
    println!("Parsed 'as'");
    let (input, _) = ws(tag("match"))(input)?;
    println!("Parsed 'match'");

    // Define a parser for either patterns or conditions
    let pattern_or_condition = alt((
        map(entity_pattern, |p| (p, None::<String>)),
        map(relationship_pattern, |p| (p, None::<String>)),
        map(condition, |c| (Pattern::Condition(c.clone()), Some(c))),
    ));

    // Parse a list of patterns and/or conditions
    let (input, patterns_and_conditions) =
        separated_list0(ws(char(',')), pattern_or_condition)(input)?;

    // Separate patterns and conditions from the combined list
    let mut patterns = Vec::new();
    let mut conditions = Vec::new();

    for (pattern, condition) in patterns_and_conditions {
        if let Some(cond) = condition {
            conditions.push(cond);
        } else {
            patterns.push(pattern);
        }
    }
    println!("Parsed patterns: {:?}", patterns);

    let (input, _) = ws(tag("infer"))(input)?;
    println!("Parsed 'infer'");
    let (input, inference_type) = inference_type(input)?;
    println!("Parsed inference type: {:?}", inference_type);
    let (input, inferences) = separated_list1(
        ws(char(',')),
        alt((
            entity_inference,
            relationship_inference,
            extend_entity_inference,
        )),
    )(input)?;
    println!("Parsed inferences: {:?}", inferences);
    let (input, _) = ws(char(';'))(input)?;
    println!("Parsed semicolon");

    println!("Constructing Rule struct");
    let rule = Rule {
        name: name.to_string(),
        patterns,
        compute_clauses: None,
        inference_type,
        inferences,
    };
    println!("Rule struct constructed: {:?}", rule);

    Ok((input, rule))
}

// Update the InferenceType enum and parser
#[derive(Debug)]
enum InferenceType {
    Derive,
    Materialize,
}

fn inference_type(input: &str) -> IResult<&str, InferenceType> {
    alt((
        map(tag("derive"), |_| InferenceType::Derive),
        map(tag("materialize"), |_| InferenceType::Materialize),
    ))(input)
}

// Update the Rule struct to match the parser
#[derive(Debug)]
struct Rule {
    name: String,
    patterns: Vec<Pattern>,
    compute_clauses: Option<Vec<ComputeClause>>,
    inference_type: InferenceType,
    inferences: Vec<Inference>,
}

// Main parser function
#[derive(Debug)]
pub enum CosGraphQuery {
    EntityDefinition(EntityDefinition),
    RelationshipDefinition(RelationshipDefinition),
    EntityInsertion(EntityInsertion),
    RelationshipInsertion(RelationshipInsertion),
    Query(Query),
    Rule(Rule),
}

// Helper function to parse keywords with flexible whitespace, including newlines
fn ws_tag<'a>(t: &'a str) -> impl Fn(&'a str) -> IResult<&'a str, &'a str> {
    move |input| {
        println!("Attempting to match tag '{}' in: {:?}", t, input);
        let result = delimited(multispace0, tag(t), multispace0)(input);
        println!("Result of matching '{}': {:?}", t, result);
        result
    }
}

pub fn parse_cos_graph_query(input: &str) -> IResult<&str, Vec<CosGraphQuery>> {
    println!("Starting to parse input:\n{}", input);
    let result = many0(alt((
        preceded(
            ws_tag("define"),
            alt((
                preceded(
                    ws_tag("entity"),
                    map(entity_definition, |ed| {
                        println!("Parsed entity definition: {:?}", ed);
                        CosGraphQuery::EntityDefinition(ed)
                    }),
                ),
                preceded(
                    ws_tag("relationship"),
                    map(relationship_definition, |rd| {
                        println!("Parsed relationship definition: {:?}", rd);
                        CosGraphQuery::RelationshipDefinition(rd)
                    }),
                ),
                preceded(
                    ws_tag("rule"),
                    map(rule, |r| {
                        println!("Parsed rule: {:?}", r);
                        CosGraphQuery::Rule(r)
                    }),
                ),
            )),
        ),
        preceded(
            ws_tag("insert"),
            alt((
                map(entity_insertion, |ei| {
                    println!("Parsed entity insertion: {:?}", ei);
                    CosGraphQuery::EntityInsertion(ei)
                }),
                map(relationship_insertion, |ri| {
                    println!("Parsed relationship insertion: {:?}", ri);
                    CosGraphQuery::RelationshipInsertion(ri)
                }),
            )),
        ),
        preceded(
            ws_tag("match"),
            map(query, |q| {
                println!("Parsed query: {:?}", q);
                CosGraphQuery::Query(q)
            }),
        ),
    )))(input);

    match &result {
        Ok((remaining, queries)) => {
            println!("Parsing completed. {} queries parsed.", queries.len());
            if !remaining.is_empty() {
                println!("Remaining unparsed input:\n{}", remaining);
            }
        }
        Err(e) => println!("Error during parsing: {:?}", e),
    }

    result
}

// Test the parser
fn main() {
    let input = r#"
        define entity person as
            name: string,
            age: int,
            email: string;

        define relationship works_in as
            (employee: person, department: department);

        insert $john isa person (
            name: "John Doe",
            age: 30,
            email: "john@example.com"
        );

        insert $job1 (
            employee: $john,
            department: $tech_dept
        ) forms works_in;

        match
            $person isa person (name: $name, age: $age),
            $age > 25
        get $name;

        define rule infer_senior_employee as
        match
            $employee isa person (
                name: $name,
                hire_date: $hire_date
            ),
            ($employee, $company) forms employment (
                role: $role
            )

        infer derive
            extend $employee (
                seniority: "Senior"
            ),
            ($employee, $company) forms senior_role (
                role: $role
            );
    "#;

    match parse_cos_graph_query(input) {
        Ok((_, queries)) => {
            for query in queries {
                println!("{:#?}", query);
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }
}
