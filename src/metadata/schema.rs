use serde::{Deserialize, Serialize};

use super::{decimal_to_binary_vec, nearest_power_of_two, Error, FieldName, FieldValue};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportedCondition {
    And(HashSet<String>),
    Or(HashSet<String>),
}

impl SupportedCondition {
    fn field_names(&self) -> HashSet<&str> {
        match self {
            Self::And(s) => s.iter().map(|x| x.as_ref()).collect(),
            Self::Or(s) => s.iter().map(|x| x.as_ref()).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataField {
    pub name: String,
    /// Values are associated with numeric identifiers so that they
    /// can be consistently represented in binary (Vec<u8>)
    pub value_index: HashMap<FieldValue, u16>,
    pub num_dims: u8,
}

fn set_to_value_index(value_set: HashSet<FieldValue>) -> HashMap<FieldValue, u16> {
    let mut values = value_set.into_iter().collect::<Vec<FieldValue>>();
    values.sort_by(|a, b| {
        match (a, b) {
            (FieldValue::Int(a), FieldValue::Int(b)) => a.cmp(b),
            (FieldValue::String(a), FieldValue::String(b)) => a.cmp(b),
            // The following cases need not be handled as the set is
            // validated to be homogeneous by this time.
            (FieldValue::Int(_), FieldValue::String(_)) => std::cmp::Ordering::Less,
            (FieldValue::String(_), FieldValue::Int(_)) => std::cmp::Ordering::Greater,
        }
    });
    let mut value_index = HashMap::with_capacity(values.len());
    for (i, v) in values.into_iter().enumerate() {
        value_index.insert(v, i as u16);
    }
    value_index
}

impl MetadataField {
    /// Constructor for MetadataField
    ///
    /// Also checks that all FieldValue's are of the same variant
    pub fn new(name: String, values: HashSet<FieldValue>) -> Result<Self, Error> {
        // validate input
        let unique_types = values
            .iter()
            .map(|value| value.type_as_str())
            .collect::<HashSet<&str>>();
        if unique_types.len() > 1 {
            return Err(Error::InvalidFieldValues(
                "Field values must be homogeneous in type".to_owned(),
            ));
        }
        let value_index = set_to_value_index(values);
        let num_dims = nearest_power_of_two(value_index.len() as u16)
            .ok_or(Error::InvalidFieldCardinality(format!("Field = {name}")))?;
        Ok(Self {
            name,
            value_index,
            num_dims,
        })
    }

    /// Returns a numeric identifier for the value from the `value_index`
    pub fn value_id(&self, value: &FieldValue) -> Result<u16, Error> {
        self.value_index
            .get(value)
            .copied()
            .ok_or(Error::InvalidFieldValue(format!(
                "Invalid value {:?} for field {}",
                value, self.name
            )))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSchema {
    pub fields: Vec<MetadataField>,
    pub conditions: Vec<SupportedCondition>,
}

impl MetadataSchema {
    /// Constructor for MetadataSchema
    ///
    /// Also checks that the MetadataSchema is valid i.e. the field
    /// names in `conditions` are subset of field names defined in the
    /// schema
    pub fn new(
        fields: Vec<MetadataField>,
        conditions: Vec<SupportedCondition>,
    ) -> Result<Self, Error> {
        let field_names = fields
            .iter()
            .map(|f| f.name.as_ref())
            .collect::<HashSet<&str>>();
        let conditions_are_valid = conditions
            .iter()
            .all(|c| c.field_names().is_subset(&field_names));
        if conditions_are_valid {
            Ok(Self { fields, conditions })
        } else {
            Err(Error::InvalidMetadataSchema)
        }
    }

    /// Return base dimensions for the MetadataSchema instance
    ///
    /// Base dimensions are to be used when there are no metadata
    /// fields associated with the vector.
    pub fn base_dimensions(&self) -> MetadataDimensions {
        // Assuming the no. of metadata fields will be small enough
        // that the sum of dimensions will fit in 8 bits.
        //
        // @TODO(vineet): Perhaps we should put a limit on max no. of
        // metadata fields supported
        let sum: u8 = self.fields.iter().map(|field| field.num_dims).sum();
        vec![0; sum as usize]
    }

    /// Return weighted dimensions based on given metadata fields
    ///
    /// @NOTE(vineet): We're not checking that the fields are valid
    /// for the schema. Not sure if that check should happen here or
    /// in the calling function
    pub fn weighted_dimensions(
        &self,
        fields: &HashMap<FieldName, FieldValue>,
        weight: i32,
    ) -> Result<MetadataDimensions, Error> {
        let mut result = vec![];
        for field in &self.fields {
            match fields.get(&field.name) {
                Some(value) => {
                    let value_id = field.value_id(value)?;
                    let mut field_dims = decimal_to_binary_vec(value_id, field.num_dims as usize)
                        .iter()
                        .map(|x| (*x as i32) * weight)
                        .collect::<Vec<i32>>();
                    result.append(&mut field_dims);
                }
                None => {
                    let mut field_dims = vec![0; field.num_dims as usize];
                    result.append(&mut field_dims);
                }
            }
        }
        Ok(result)
    }

    pub fn get_field(&self, name: &str) -> Result<&MetadataField, Error> {
        self.fields
            .iter()
            .find(|field| field.name == name)
            .ok_or(Error::InvalidField(name.to_string()))
    }
}

type MetadataDimensions = Vec<i32>;

// Functionality to be implemented
//
// 1. Json representation of MetadataSchema
//
// 2. Deserializing from Json
//
// 3. Binary representation of MetadataScheme to store in lmdb (bincode?)
//
// 4. Serializing to the above binary representation
//
// 5. [✓] Validation of MetadataSchema to ensure that `conditions` vector
//    contains valid fields
//
// 6. [✓] Converting MetadataSchema to `MetadataDimensions` base
//    vector (without high weight values)
//
// 7. [✓] Given `MetadataSchema` and metadata names + values, create
//    `MetadataDimensions` with high weight values

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_to_value_index() {
        let mut s = HashSet::new();
        let b = FieldValue::String("b".to_owned());
        let a = FieldValue::String("a".to_owned());
        let n = FieldValue::String("n".to_owned());
        let d = FieldValue::String("d".to_owned());
        s.insert(b.clone());
        s.insert(a.clone());
        s.insert(n.clone());
        s.insert(d.clone());

        let m = set_to_value_index(s);
        assert_eq!(&0, m.get(&a).unwrap());
        assert_eq!(&1, m.get(&b).unwrap());
        assert_eq!(&2, m.get(&d).unwrap());
        assert_eq!(&3, m.get(&n).unwrap());
    }

    #[test]
    fn test_metadata_field_new_valid() {
        let mut values = HashSet::new();
        let b = FieldValue::String("b".to_owned());
        let a = FieldValue::String("a".to_owned());
        let n = FieldValue::String("n".to_owned());
        let d = FieldValue::String("d".to_owned());
        values.insert(b.clone());
        values.insert(a.clone());
        values.insert(n.clone());
        values.insert(d.clone());

        let name = "myfield".to_owned();
        let result = MetadataField::new(name.clone(), values);
        assert!(result.is_ok());
        let m = result.unwrap();
        assert_eq!(name, m.name);
        assert_eq!(2, m.num_dims);
        let vi = m.value_index;
        assert_eq!(&0, vi.get(&a).unwrap());
        assert_eq!(&1, vi.get(&b).unwrap());
        assert_eq!(&2, vi.get(&d).unwrap());
        assert_eq!(&3, vi.get(&n).unwrap());
    }

    #[test]
    fn test_metadata_field_new_invalid() {
        let mut values = HashSet::new();
        let v1 = FieldValue::String("a".to_owned());
        let v2 = FieldValue::Int(1);
        values.insert(v1.clone());
        values.insert(v2.clone());
        match MetadataField::new("myfield".to_owned(), values) {
            Err(Error::InvalidFieldValues(msg)) => {
                assert_eq!("Field values must be homogeneous in type", msg)
            }
            _ => panic!(),
        }

        let values: HashSet<FieldValue> = (1..2000).map(FieldValue::Int).collect();
        match MetadataField::new("myfield".to_owned(), values) {
            Err(Error::InvalidFieldCardinality(msg)) => assert_eq!("Field = myfield", msg),
            _ => panic!(),
        }
    }

    #[test]
    fn test_metadata_schema_new_valid() {
        let age_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();
        let group_values: HashSet<FieldValue> = vec!["a", "b", "c"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let group = MetadataField::new("group".to_owned(), group_values).unwrap();
        let conditions = vec![
            SupportedCondition::And(vec!["age", "group"].into_iter().map(String::from).collect()),
            SupportedCondition::Or(vec!["age", "group"].into_iter().map(String::from).collect()),
        ];
        let schema = MetadataSchema::new(vec![age, group], conditions);
        assert!(schema.is_ok());
    }

    #[test]
    fn test_metadata_schema_new_invalid() {
        let age_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();
        let conditions = vec![SupportedCondition::And(
            vec!["age", "group"].into_iter().map(String::from).collect(),
        )];
        match MetadataSchema::new(vec![age], conditions) {
            Err(Error::InvalidMetadataSchema) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn test_weighted_dimensions() {
        let age_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();
        let group_values: HashSet<FieldValue> = vec!["a", "b", "c"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let group = MetadataField::new("group".to_owned(), group_values).unwrap();
        let conditions = vec![SupportedCondition::And(
            vec!["age", "group"].into_iter().map(String::from).collect(),
        )];
        let schema = MetadataSchema::new(vec![age, group], conditions).unwrap();
        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(5));
        fields.insert("group".to_owned(), FieldValue::String("a".to_owned()));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();
        assert_eq!(
            vec![
                0, 1024, 0, 0, // 4 (original value: 5)
                0, 0 // 0 (original value: a)
            ],
            wd
        );

        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(3));
        fields.insert("group".to_owned(), FieldValue::String("c".to_owned()));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();
        assert_eq!(
            vec![
                0, 0, 1024, 0, // 2 (original value: 4)
                1024, 0 // 2 (original value: c)
            ],
            wd
        );
    }

    #[test]
    fn test_weighted_dimensions_invalid() {
        let age_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();
        let group_values: HashSet<FieldValue> = vec!["a", "b", "c"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let group = MetadataField::new("group".to_owned(), group_values).unwrap();
        let conditions = vec![SupportedCondition::And(
            vec!["age", "group"].into_iter().map(String::from).collect(),
        )];
        let schema = MetadataSchema::new(vec![age, group], conditions).unwrap();
        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(100));
        fields.insert("group".to_owned(), FieldValue::String("c".to_owned()));
        match schema.weighted_dimensions(&fields, 1024) {
            Err(Error::InvalidFieldValue(_)) => {}
            _ => panic!(),
        }
    }
}
