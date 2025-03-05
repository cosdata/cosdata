use serde::{Deserialize, Serialize};

use super::{decimal_to_binary_vec, nearest_power_of_two, Error, FieldName, FieldValue, MetadataFields};
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
            .map(|v| *v)
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

    /// Given the schema and metadata fields (name and value) associated
    /// with an input vector, find all relevant field combinations for
    /// which high weight dimensions need to be computed. The resulting
    /// combinations are supposed to be used for creating replica nodes in
    /// dense/HNSW index.
    ///
    /// The combinations are generated based on the vector of
    /// SupportedConditions defined in schema.
    fn input_field_combinations<'a>(
        &self,
        input_fields: &'a MetadataFields
    ) -> Vec<HashMap<&'a str, &'a FieldValue>> {
        let input_fields_set = input_fields
            .keys()
            .map(|n| n.as_ref())
            .collect::<HashSet<&str>>();
        let mut combinations = HashSet::new();
        for condition in &self.conditions {
            // @NOTE: We only consider `And` conditions, `Or` conditions
            // will be covered by the "all fields" combination, which will
            // be added to the combinations set later.
            if let SupportedCondition::And(cond_fields) = condition {
                let cond_fields_set = cond_fields
                    .iter()
                    .map(|s| s.as_ref())
                    .collect::<HashSet<&str>>();
                // The following handles the case where not all
                // metadata fields in the schema are specified for the
                // input vector. Only those conditions will be
                // considered which are relevant to the input.
                if cond_fields_set.is_subset(&input_fields_set) {
                    // Convert hashset to vector to be able to push it
                    // to a hashset of combinations (because HashSet
                    // doesn't implement the Hash trait)
                    let mut combination = cond_fields_set.into_iter().collect::<Vec<&str>>();
                    combination.sort();
                    combinations.insert(combination);
                }
            }
        }
        // Add a combination for all fields in the schema. This is
        // equivalent to a condition And(<all fields>). To get this,
        // we compute the intersection of all fields supported by the
        // schema with actual fields associated with the vector.
        //
        // Note that it's possible that this combination has already
        // been considered before, but it will get deduped when
        // inserting into the HashSet of combinations.
        let mut all_fields_combination = self.fields
            .iter()
            .map(|f| f.name.as_ref())
            .collect::<HashSet<&str>>()
            .intersection(&input_fields_set)
            .map(|s| *s)
            .collect::<Vec<&str>>();
        all_fields_combination.sort();
        combinations.insert(all_fields_combination);

        let mut result = vec![];
        for combination in combinations {
            let mut m: HashMap<&str, &FieldValue> = HashMap::new();
            for field_name in combination {
                // @SAFETY: It's safe to use unwrap below because the
                // fields in all combinations are guaranteed to be subset
                // of all_fields. Refer to `cond_fields_set.is_subset` and
                // `.intersection(&input_fields_set)` calls in the above
                // code.
                let key = input_fields_set.get(field_name).unwrap();
                let value = input_fields.get(*key).unwrap();
                m.insert(*key, value);
            }
            result.push(m);
        }
        result
    }

    /// Return vector of weighted dimensions based on given input
    /// metadata fields
    ///
    /// The resulting dimensions are supposed to be used for creating
    /// replica nodes in the dense/HNSW index.
    ///
    /// @NOTE(vineet): We're not checking that the fields are valid
    /// for the schema. Not sure if that check should happen here or
    /// in the calling function
    pub fn weighted_dimensions(
        &self,
        fields: &HashMap<FieldName, FieldValue>,
        weight: i32,
    ) -> Result<Vec<MetadataDimensions>, Error> {
        let field_combinations = self.input_field_combinations(&fields);
        let mut cache: HashMap<&str, Vec<i32>> = HashMap::new();
        let mut result = Vec::with_capacity(field_combinations.len());
        for field_combination in field_combinations {
            let mut dims = vec![];
            for field in &self.fields {
                if let Some(v) = cache.get(&field.name.as_ref()) {
                    let mut field_dims = v.clone();
                    dims.append(&mut field_dims);
                } else {
                    let field_dims = match field_combination.get(&field.name.as_ref()) {
                        Some(value) => {
                            let value_id = field.value_id(&value)?;
                            decimal_to_binary_vec(value_id, field.num_dims as usize)
                                .iter()
                                .map(|x| (*x as i32) * weight)
                                .collect::<Vec<i32>>()
                        }
                        None => {
                            vec![0; field.num_dims as usize]
                        }
                    };
                    let mut field_dims_clone = field_dims.clone();
                    cache.insert(&field.name, field_dims);
                    dims.append(&mut field_dims_clone);
                }
            }
            result.push(dims);
        }
        Ok(result)
    }

    pub fn get_field(&self, name: &str) -> Result<&MetadataField, Error> {
        self.fields
            .iter()
            .find(|field| field.name == name)
            .ok_or(Error::InvalidField(format!("{name}")))
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

    fn hashset(xs: Vec<&str>) -> HashSet<String> {
        xs.into_iter()
            .map(|s| String::from(s))
                .collect::<HashSet<String>>()
    }

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
            _ => assert!(false),
        }

        let values: HashSet<FieldValue> = (1..2000).map(|x| FieldValue::Int(x)).collect();
        match MetadataField::new("myfield".to_owned(), values) {
            Err(Error::InvalidFieldCardinality(msg)) => assert_eq!("Field = myfield", msg),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_metadata_schema_new_valid() {
        let age_values: HashSet<FieldValue> = (1..=10).map(|x| FieldValue::Int(x)).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();
        let group_values: HashSet<FieldValue> = vec!["a", "b", "c"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let group = MetadataField::new("group".to_owned(), group_values).unwrap();
        let conditions = vec![
            SupportedCondition::And(
                vec!["age", "group"]
                    .into_iter()
                    .map(|s| String::from(s))
                    .collect(),
            ),
            SupportedCondition::Or(
                vec!["age", "group"]
                    .into_iter()
                    .map(|s| String::from(s))
                    .collect(),
            ),
        ];
        let schema = MetadataSchema::new(vec![age, group], conditions);
        assert!(schema.is_ok());
    }

    #[test]
    fn test_metadata_schema_new_invalid() {
        let age_values: HashSet<FieldValue> = (1..=10).map(|x| FieldValue::Int(x)).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();
        let conditions = vec![SupportedCondition::And(
            vec!["age", "group"]
                .into_iter()
                .map(|s| String::from(s))
                .collect(),
        )];
        match MetadataSchema::new(vec![age], conditions) {
            Err(Error::InvalidMetadataSchema) => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_input_field_combinations() {
        let a_values: HashSet<FieldValue> = (1..=10).map(|x| FieldValue::Int(x)).collect();
        let a = MetadataField::new("a".to_owned(), a_values).unwrap();

        let b_values: HashSet<FieldValue> = (1..=10).map(|x| FieldValue::Int(x)).collect();
        let b = MetadataField::new("b".to_owned(), b_values).unwrap();

        let c_values: HashSet<FieldValue> = (1..=10).map(|x| FieldValue::Int(x)).collect();
        let c = MetadataField::new("c".to_owned(), c_values).unwrap();

        let conditions = vec![
            SupportedCondition::And(hashset(vec!["a", "b"])),
            SupportedCondition::And(hashset(vec!["a", "c"])),
            SupportedCondition::Or(hashset(vec!["a", "b"])),
        ];

        // Helper fn to create the MetadataFields hashmap from
        // key-value tuples
        let input_fields = |kvs: Vec<(&str, i32)>| {
            kvs.into_iter()
                .map(|(k, v)| (k.to_owned(), FieldValue::Int(v)))
                .collect::<HashMap<String, FieldValue>>()
        };

        let schema = MetadataSchema::new(vec![a, b, c], conditions).unwrap();

        // As the input contains only a single field, only the
        // following supported conditions are relevant to this vector:
        //
        //  1. matches "a"
        //
        // Hence the result is 1 combination i.e. 1 replica
        let fs = input_fields(vec![("a", 1)]);
        let cs = schema.input_field_combinations(&fs);
        assert_eq!(1, cs.len());
        let mut ec1 = HashMap::with_capacity(1);
        ec1.insert("a", &FieldValue::Int(1));
        assert_eq!(ec1, cs[0]);

        // As the input contains only "a" and "b", only the following
        // supported `AND` conditions are relevant to this vector
        //
        //   1. matches "a and b"
        //
        // Hence the result is 1 combination i.e. 1 replica. It is
        // sufficient and it's also equivalent to the "all fields"
        // replica to support individual field matches and OR
        // conditions.
        let fs = input_fields(vec![("a", 1), ("b", 2)]);
        let cs = schema.input_field_combinations(&fs);
        assert_eq!(1, cs.len());
        let mut ec1 = HashMap::with_capacity(1);
        ec1.insert("a", &FieldValue::Int(1));
        ec1.insert("b", &FieldValue::Int(2));
        assert_eq!(ec1, cs[0]);

        // The input contains fields "b" and "c" only. There is no
        // supported `AND` condition that includes these 2 fields. But
        // we need the "all fields" replica to support individual
        // field matches and OR conditions.
        //
        // Hence the result is 1 combination i.e. 1 replica
        let fs = input_fields(vec![("b", 3), ("c", 2)]);
        let cs = schema.input_field_combinations(&fs);
        assert_eq!(1, cs.len());
        let mut ec1 = HashMap::with_capacity(1);
        ec1.insert("b", &FieldValue::Int(3));
        ec1.insert("c", &FieldValue::Int(2));
        assert_eq!(ec1, cs[0]);

        // The input contains all 3 fields. So considering the
        // supported conditions, 3 combinations or replicas are required
        //
        //   1. to match "a == x and b == y"
        //   2. to match "a == x and c == z"
        //   3. all fields replica for individual fields and OR queries
        let fs = input_fields(vec![("a", 4), ("b", 5), ("c", 6)]);
        let mut cs = schema.input_field_combinations(&fs);
        // To make the order of returned combinations deterministic,
        // sort them by sum of the field values.
        cs.sort_by_key(|c| {
            c.values()
                .map(|v| {
                    match v {
                        FieldValue::Int(i) => *i,
                        FieldValue::String(_) => 0,
                    }})
                .sum::<i32>()});
        assert_eq!(3, cs.len());

        let mut ec1 = HashMap::new();
        ec1.insert("a", &FieldValue::Int(4));
        ec1.insert("b", &FieldValue::Int(5));

        let mut ec2 = HashMap::new();
        ec2.insert("a", &FieldValue::Int(4));
        ec2.insert("c", &FieldValue::Int(6));

        let mut ec3 = HashMap::new();
        ec3.insert("a", &FieldValue::Int(4));
        ec3.insert("b", &FieldValue::Int(5));
        ec3.insert("c", &FieldValue::Int(6));

        assert_eq!(ec1, cs[0]);
        assert_eq!(ec2, cs[1]);
        assert_eq!(ec3, cs[2]);
    }

    #[test]
    fn test_weighted_dimensions() {
        let age_values: HashSet<FieldValue> = (1..=10).map(|x| FieldValue::Int(x)).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();

        let group_values: HashSet<FieldValue> = vec!["a", "b", "c"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let group = MetadataField::new("group".to_owned(), group_values).unwrap();

        let level_values: HashSet<FieldValue> = vec!["first", "second", "third"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let level = MetadataField::new("level".to_owned(), level_values).unwrap();

        let conditions = vec![
            SupportedCondition::And(hashset(vec!["age", "group"])),
            SupportedCondition::And(hashset(vec!["age", "level"])),
        ];

        let schema = MetadataSchema::new(vec![age, group, level], conditions).unwrap();

        // Case 1: When only "age" field is specified in input vector

        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(5));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();
        let exp_dim1 = vec![
            0, 1024, 0, 0, // 4 (original value: 5)
            0, 0, // (not specified)
            0, 0, // (not specified)
        ];
        assert_eq!(vec![exp_dim1], wd);

        // Case 2: When only "age" and "group" fields are specified in
        // input vector

        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(5));
        fields.insert("group".to_owned(), FieldValue::String("a".to_owned()));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();
        let exp_dim1 = vec![
            0, 1024, 0, 0, // 4 (original value: 5)
            0, 0, // 0 (original value: a) @TODO: Fix: 0 is indistinguishable from not specified
            0, 0, // (not specified)
        ];
        assert_eq!(vec![exp_dim1], wd);

        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(3));
        fields.insert("group".to_owned(), FieldValue::String("c".to_owned()));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();
        let exp_dim1 = vec![
            0, 0, 1024, 0, // 2 (original value: 4)
            1024, 0, // 2 (original value: c)
            0, 0, // (not specified) @TODO: Fix
        ];
        assert_eq!(vec![exp_dim1], wd);

        // Case 3: When all fields are specified in input vector

        let mut fields = HashMap::with_capacity(3);
        fields.insert("age".to_owned(), FieldValue::Int(5));
        fields.insert("group".to_owned(), FieldValue::String("a".to_owned()));
        fields.insert("level".to_owned(), FieldValue::String("third".to_owned()));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();

        // dimensions to support AND(age, group)
        let exp_dim1 = vec![
            0, 1024, 0, 0, // 4 (original value: 5)
            0, 0, // 0 (original value: a)
            0, 0,
        ];

        // dimensions to support AND(age, level)
        let exp_dim2 = vec![
            0, 1024, 0, 0, // 4 (original value: 5)
            0, 0,
            1024, 0, // 2 (original value: third)
        ];

        // all fields dimensions to support individual field and OR queries
        let exp_dim3 = vec![
            0, 1024, 0, 0, // 4 (original value: 5)
            0, 0, // 0 (original value: a) @TODO: Fix
            1024, 0, // 2 (original value: third)
        ];
        for i in 0..wd.len() {
            assert!(wd[i] == exp_dim1 || wd[i] == exp_dim2 || wd[i] == exp_dim3);
        }
    }

    #[test]
    fn test_weighted_dimensions_invalid() {
        let age_values: HashSet<FieldValue> = (1..=10).map(|x| FieldValue::Int(x)).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();
        let group_values: HashSet<FieldValue> = vec!["a", "b", "c"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let group = MetadataField::new("group".to_owned(), group_values).unwrap();
        let conditions = vec![
            SupportedCondition::And(hashset(vec!["age", "group"])),
        ];
        let schema = MetadataSchema::new(vec![age, group], conditions).unwrap();
        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(100));
        fields.insert("group".to_owned(), FieldValue::String("c".to_owned()));
        match schema.weighted_dimensions(&fields, 1024) {
            Err(Error::InvalidFieldValue(_)) => assert!(true),
            _ => assert!(false),
        }
    }
}
