use serde::{Deserialize, Serialize};


use super::{
    nearest_power_of_two, Error, FieldName, FieldValue, MetadataFields,
    decimal_to_binary_vec_memoized, gen_combinations_optimized,
};
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

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

    /// Returns a tuple that can be used to compare instances of
    /// this struct.
    ///
    /// This is done because this struct uses HashSet, so it can't
    /// derive from the Hash trait. The use case for comparing
    /// `SupportedCondition` instances is rare, so `BTreeSet` seems to
    /// be an overkill.
    fn comparable_encoding(&self) -> (&str, Vec<&str>) {
        let mut field_names = self.field_names().into_iter().collect::<Vec<&str>>();
        field_names.sort();
        let kind = match self {
            Self::And(_) => "and",
            Self::Or(_) => "or",
        };
        (kind, field_names)
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

// Converts a set of `FieldValue`s into a HashMap in which
// `FieldValue`s are the keys and integers (monotonically increasing
// id) as values. Note that the ids start from 1 and not 0 because 0
// is considered as the base dimension. The input hash set is first
// sorted to make sure the result is deterministic.
//
// @NOTE: This function assumes that all FieldValue instances in the
// HashSet are of the same variant.
fn set_to_value_index(value_set: HashSet<FieldValue>) -> HashMap<FieldValue, u16> {
    let mut values = value_set.into_iter().collect::<Vec<FieldValue>>();
    values.sort_by(|a, b| {
        match (a, b) {
            (FieldValue::Int(a), FieldValue::Int(b)) => a.cmp(b),
            (FieldValue::String(a), FieldValue::String(b)) => a.cmp(b),
            // @NOTE: We are assuming that the input hash set is
            // homogeneous i.e. contains FieldValue instances of the
            // same variant.
            _ => std::cmp::Ordering::Less,
        }
    });
    let mut value_index = HashMap::with_capacity(values.len());
    for (i, v) in values.into_iter().enumerate() {
        // @NOTE: the values are monotonically increasing identifiers
        // that are one-indexed and not zero-indexed
        value_index.insert(v, (i + 1) as u16);
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
        let cardinality = value_index.len();
        // @NOTE: No. of dimensions are calculated to support 1 value
        // more than the cardinality because the index in value_index
        // starts with 1 and not 0. In other words, 0 is not used to
        // represent any value.
        let num_dims = nearest_power_of_two((cardinality + 1) as u16)
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

    pub fn max_cardinality(&self) -> u8 {
        2u8.pow(self.num_dims as u32) - 1
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
    ///
    /// In case there duplicate conditions are specified, it takes
    /// care of deduplicating them.
    pub fn new(
        fields: Vec<MetadataField>,
        conditions: Vec<SupportedCondition>,
    ) -> Result<Self, Error> {
        let field_names = fields
            .iter()
            .map(|f| f.name.as_ref())
            .collect::<HashSet<&str>>();

        let mut encoded_conds: HashSet<(&str, Vec<&str>)> = HashSet::new();
        let mut deduped_conditions = vec![];
        for condition in &conditions {
            if !condition.field_names().is_subset(&field_names) {
                return Err(Error::InvalidMetadataSchema);
            }
            let cond_enc = condition.comparable_encoding();
            if !encoded_conds.contains(&cond_enc) {
                deduped_conditions.push(condition.clone());
            }
            encoded_conds.insert(cond_enc);
        }

        Ok(Self {
            fields,
            conditions: deduped_conditions,
        })
    }

    pub fn num_total_dims(&self) -> u8 {
        self.fields.iter().map(|field| field.num_dims).sum()
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
        let sum: u8 = self.num_total_dims();
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
        input_fields: &'a MetadataFields,
    ) -> Vec<HashMap<&'a str, &'a FieldValue>> {
        // If no input fields are specified, return empty vector
        if input_fields.is_empty() {
            return vec![];
        }

        let mut input_fields_set: HashSet<&str> = HashSet::new();
        let mut combinations: HashSet<Vec<&str>> = HashSet::new();

        // Create individual field combinations. This is an oxymoron
        // but for the ease of deduplicating all combinations in a
        // single place, we consider it as a combination that contains
        // a single field.
        for key in input_fields.keys() {
            input_fields_set.insert(key.as_ref());
            combinations.insert(vec![key.as_ref()]);
        }

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
        // Create a cache key for this specific call
        let cache_key = format!("{:?}_{}", fields, weight);
        let dimensions_cache = super::get_dimensions_cache();
        
        // Check if result is already cached
        if let Some(cached_result) = dimensions_cache.get(&(cache_key.clone(), None)) {
            return Ok(cached_result.clone());
        }
        
        let field_combinations = self.input_field_combinations(fields);
        // We cache the results of calling fn `decimal_to_binary_vec_memoized`
        // as there's a chance it would get called multiple times for
        // the same input
        let mut cache: HashMap<(u16, usize), Vec<i32>> = HashMap::new();
        let mut result = Vec::with_capacity(field_combinations.len());
        for field_combination in field_combinations {
            let mut dims = vec![];
            for field in &self.fields {
                let mut field_dims = match field_combination.get(&field.name.as_ref()) {
                    Some(value) => {
                        let value_id = field.value_id(value)?;
                        let size = field.num_dims as usize;
                        match cache.get(&(value_id, size)) {
                            Some(r) => r.clone(),
                            None => {
                                let dims = decimal_to_binary_vec_memoized(value_id, size)
                                    .iter()
                                    .map(|x| (*x as i32) * weight)
                                    .collect::<Vec<i32>>();
                                cache.insert((value_id, size), dims.clone());
                                dims
                            }
                        }
                    }
                    None => vec![0; field.num_dims as usize],
                };
                dims.append(&mut field_dims);
            }
            result.push(dims);
        }
        
        // Cache the result
        dimensions_cache.insert((cache_key, None), result.clone());
        
        Ok(result)
    }

    /// Computes dimensions for the pseudo root node
    pub fn pseudo_root_dimensions(&self, weight: i32) -> MetadataDimensions {
        let total_dims = self.num_total_dims();
        vec![weight; total_dims as usize]
    }

    // Computes dimensions for all non-root pseudo nodes
    //
    // In other words, the result will never include the dimensions
    // for pseudo node (see `pseudo_root_dimensions` fn). There are
    // two separate functions because the pseudo root node and other
    // pseudo nodes are created in separate code paths.
    pub fn pseudo_nonroot_dimensions(&self, weight: i32) -> Vec<MetadataDimensions> {
        // Create a cache key for this specific call
        let cache_key = format!("pseudo_nonroot_{:?}_{}", self.fields, weight);
        let dimensions_cache = super::get_dimensions_cache();
        
        // Check if result is already cached
        if let Some(cached_result) = dimensions_cache.get(&(cache_key.clone(), Some("pseudo_nonroot".to_string()))) {
            return cached_result.clone();
        }
        
        // A hashmap of field names to the value_ids in asc
        // order. Will be constructed when iterting through the fields
        // for the first time.
        let mut field_value_ids: HashMap<String, Vec<u16>> =
            HashMap::with_capacity(self.fields.len());

        // We first find all combinations of field values ids i.e. if
        // there are 2 fields `foo`: [1, 2] and `bar`: [3, 4], then
        // combinations will be a vector of items such as [1, 3], [1,
        // 4], [2, 3], [2, 4]. Note that the order of value_ids in the
        // inner vectors matter (same as the order of `self.fields`).
        let mut combinations: Vec<Vec<u16>> = vec![];

        let num_fields = self.fields.len();

        // Compute combinations for individual fields i.e. for every
        // field, only that field has non-zero value and the remaining
        // are all zero.
        for x in &self.fields {
            let c = x.max_cardinality();
            let vids = (1..=c).map(|i| i as u16).collect::<Vec<u16>>();
            field_value_ids.insert(x.name.clone(), vids.clone());

            let mut vs = Vec::with_capacity(num_fields);
            for y in &self.fields {
                if x.name == y.name {
                    vs.push(vids.clone());
                } else {
                    vs.push(vec![0]);
                }
            }
            let mut cs = gen_combinations_optimized(&vs);
            combinations.append(&mut cs);
        }

        let mut is_pseudo_root_included = self.fields.len() == 1 && self.conditions.is_empty();

        // Compute combinations for supported `And` conditions
        // i.e. only the fields that are part of a supported `And`
        // condition will have non-zero values.
        for condition in &self.conditions {
            if let SupportedCondition::And(cond_fields) = condition {
                let mut vs = Vec::with_capacity(num_fields);
                for field in &self.fields {
                    if cond_fields.contains(&field.name) {
                        let vids = field_value_ids.get(&field.name).unwrap();
                        vs.push(vids.clone());
                    } else {
                        vs.push(vec![0]);
                    }
                }
                let mut cs = gen_combinations_optimized(&vs);
                combinations.append(&mut cs);

                // Check if the schema supports an `And` condition
                // between all fields.
                if cond_fields.len() == self.fields.len() {
                    is_pseudo_root_included = true;
                }
            }
        }

        // Sort all combinations by the sum of the values. This is so
        // that if the combination that has max value ids for every
        // field exists, it ends up at the beginning of the vector
        combinations.sort_by_cached_key(|c| Reverse(c.iter().copied().sum::<u16>()));

        // Convert combinations of valid_ids into pseudo_dimensions
        // (binary_vec with high weight values)
        let mut pseudo_dimensions = Vec::with_capacity(combinations.len());
        // We cache the results of calling fn `decimal_to_binary_vec_memoized`
        // as there's a chance it would get called multiple times for
        // the same input
        let mut cache: HashMap<(u16, usize), Vec<i32>> = HashMap::new();

        for combination in combinations {
            let mut dims = Vec::with_capacity(num_fields);
            for (i, value_id) in combination.iter().enumerate() {
                let size = self.fields[i].num_dims as usize;
                let mut field_dims = match cache.get(&(*value_id, size)) {
                    Some(r) => r.clone(),
                    None => {
                        let r = decimal_to_binary_vec_memoized(*value_id, size)
                            .iter()
                            .map(|x| (*x as i32) * weight)
                            .collect::<Vec<i32>>();
                        cache.insert((*value_id, size), r.clone());
                        r
                    }
                };
                dims.append(&mut field_dims);
            }
            pseudo_dimensions.push(dims);
        }

        // There's a chance that pseudo root dimensions would also
        // have been considered. Exclude it from the result
        if is_pseudo_root_included {
            let root_dims = self.pseudo_root_dimensions(weight);
            pseudo_dimensions.retain(|dims| *dims != root_dims);
        }
        
        // Cache the result
        dimensions_cache.insert((cache_key, Some("pseudo_nonroot".to_string())), pseudo_dimensions.clone());

        pseudo_dimensions
    }

    // Computes all pseudo dimensions
    pub fn pseudo_weighted_dimensions(&self, weight: i32) -> Vec<MetadataDimensions> {
        let root_dims = self.pseudo_root_dimensions(weight);
        let mut non_root_dims = self.pseudo_nonroot_dimensions(weight);
        let mut result = vec![root_dims];
        result.append(&mut non_root_dims);
        result
    }

    pub fn get_field(&self, name: &str) -> Result<&MetadataField, Error> {
        self.fields
            .iter()
            .find(|field| field.name == name)
            .ok_or(Error::InvalidField(name.to_string()))
    }

    /// Returns the max no. of replica nodes that will be created per
    /// vector inserted into the index.
    pub fn max_num_replicas(&self) -> u8 {
        // Base replica + 1 replica for every field
        let mut total: u8 = 1 + self.fields.len() as u8;
        // 1 replica for every 'And' condition
        for condition in &self.conditions {
            match condition {
                SupportedCondition::And(_) => total += 1,
                SupportedCondition::Or(_) => {}
            }
        }
        total
    }
}

pub type MetadataDimensions = Vec<i32>;

#[cfg(test)]
mod tests {
    use serde_cbor::{from_slice, to_vec};

    use super::*;

    fn hashset(xs: Vec<&str>) -> HashSet<String> {
        xs.into_iter()
            .map(String::from)
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
        assert_eq!(&1, m.get(&a).unwrap());
        assert_eq!(&2, m.get(&b).unwrap());
        assert_eq!(&3, m.get(&d).unwrap());
        assert_eq!(&4, m.get(&n).unwrap());
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
        // Even though there are 4 values, we need 3 dimensions
        // because ids start from 1 and not 0. In other words, 0 is
        // not used to represent any value.
        assert_eq!(3, m.num_dims);
        let vi = m.value_index;
        assert_eq!(&1, vi.get(&a).unwrap());
        assert_eq!(&2, vi.get(&b).unwrap());
        assert_eq!(&3, vi.get(&d).unwrap());
        assert_eq!(&4, vi.get(&n).unwrap());
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
    fn test_metadata_schema_new_invalid_fields_in_conds() {
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
    fn test_metadata_schema_new_duplicate_conds() {
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
            SupportedCondition::And(vec!["age", "group"].into_iter().map(String::from).collect()),
        ];
        let schema = MetadataSchema::new(vec![age, group], conditions).unwrap();
        // Assert that conditions are deduplicated
        assert_eq!(2, schema.conditions.len());
    }

    // Following fns that are prefixed with `ifc_` are test util for
    // testing the fn `input_field_combinations`.

    fn ifc_input(kvs: Vec<(&str, i32)>) -> MetadataFields {
        kvs.into_iter()
            .map(|(k, v)| (k.to_owned(), FieldValue::Int(v)))
            .collect()
    }

    fn ifc_sort_result(result: &mut Vec<HashMap<&str, &FieldValue>>) {
        result.sort_by_key(|c| {
            c.values()
                .map(|v| match v {
                    FieldValue::Int(i) => *i,
                    FieldValue::String(_) => 0,
                })
                .sum::<i32>()
        })
    }

    #[test]
    fn test_input_field_combinations() {
        let a_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
        let a = MetadataField::new("a".to_owned(), a_values).unwrap();

        let b_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
        let b = MetadataField::new("b".to_owned(), b_values).unwrap();

        let c_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
        let c = MetadataField::new("c".to_owned(), c_values).unwrap();

        let conditions = vec![
            SupportedCondition::And(hashset(vec!["a", "b"])),
            SupportedCondition::And(hashset(vec!["a", "c"])),
            SupportedCondition::Or(hashset(vec!["a", "b"])),
        ];

        let schema = MetadataSchema::new(vec![a, b, c], conditions).unwrap();

        // Test case 1: As the input contains only a single field,
        // only the following replicas are relevant to this vector:
        //
        //  1. matches "a"
        //
        // Hence the result is 1 combination i.e. 1 replica
        let fs = ifc_input(vec![("a", 1)]);
        let cs = schema.input_field_combinations(&fs);
        assert_eq!(1, cs.len());
        let mut ec1 = HashMap::with_capacity(1);
        ec1.insert("a", &FieldValue::Int(1));
        assert_eq!(ec1, cs[0]);

        // Test case 2: As the input contains only "a" and "b", only
        // the following replicas are relevant to this vector
        //
        //   1. matches "a"
        //   2. matches "b"
        //   3. matches "a AND b"
        //
        // Hence the result is 3 combinations i.e. 3 replicas.
        let fs = ifc_input(vec![("a", 1), ("b", 2)]);
        let mut cs = schema.input_field_combinations(&fs);

        // To make the order of returned combinations deterministic,
        // sort them by sum of the field values.
        ifc_sort_result(&mut cs);

        assert_eq!(3, cs.len());
        let mut ec1 = HashMap::with_capacity(1);
        ec1.insert("a", &FieldValue::Int(1));
        let mut ec2 = HashMap::with_capacity(1);
        ec2.insert("b", &FieldValue::Int(2));
        let mut ec3 = HashMap::with_capacity(2);
        ec3.insert("a", &FieldValue::Int(1));
        ec3.insert("b", &FieldValue::Int(2));
        assert_eq!(vec![ec1, ec2, ec3], cs);

        // Test case 3: The input contains fields "b" and "c"
        // only. There is no supported `AND` condition that includes
        // these 2 fields. But we need the "all fields" replica is
        // created any way. Hence the following replicas are relevant
        // to this vector
        //
        //   1. matches "b"
        //   2. matches "c"
        let fs = ifc_input(vec![("b", 3), ("c", 2)]);
        let mut cs = schema.input_field_combinations(&fs);
        // To make the order of returned combinations deterministic,
        // sort them by sum of the field values.
        ifc_sort_result(&mut cs);
        assert_eq!(2, cs.len());
        let mut ec1 = HashMap::with_capacity(1);
        ec1.insert("c", &FieldValue::Int(2));
        let mut ec2 = HashMap::with_capacity(1);
        ec2.insert("b", &FieldValue::Int(3));
        assert_eq!(vec![ec1, ec2], cs);

        // Test case 4: The input contains all 3 fields. So
        // considering the supported conditions, 5 combinations or
        // replicas are relevant to this vector
        //
        //   1. matches "a"
        //   2. matches "b"
        //   3. matches "c"
        //   4. matches "a AND b"
        //   5. to match "a AND c"
        let fs = ifc_input(vec![("a", 4), ("b", 5), ("c", 6)]);
        let mut cs = schema.input_field_combinations(&fs);
        // To make the order of returned combinations deterministic,
        // sort them by sum of the field values.
        ifc_sort_result(&mut cs);
        assert_eq!(5, cs.len());

        let mut ec1 = HashMap::new();
        ec1.insert("a", &FieldValue::Int(4));

        let mut ec2 = HashMap::new();
        ec2.insert("b", &FieldValue::Int(5));

        let mut ec3 = HashMap::new();
        ec3.insert("c", &FieldValue::Int(6));

        let mut ec4 = HashMap::new();
        ec4.insert("a", &FieldValue::Int(4));
        ec4.insert("b", &FieldValue::Int(5));

        let mut ec5 = HashMap::new();
        ec5.insert("a", &FieldValue::Int(4));
        ec5.insert("c", &FieldValue::Int(6));

        assert_eq!(vec![ec1, ec2, ec3, ec4, ec5], cs);

        // Test case 5: When the input contains no fields
        let fs = HashMap::new();
        let cs = schema.input_field_combinations(&fs);
        assert!(cs.is_empty());
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

        let mut fields = HashMap::with_capacity(1);
        fields.insert("age".to_owned(), FieldValue::Int(5));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();
        let exp_dim1 = vec![
            0, 1024, 0, 1024, // 5 (original value: 5)
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
        let exp = vec![
            vec![
                0, 1024, 0, 1024, // 5 (original value: 5)
                0, 0, // no high weights
                0, 0, // no high weights
            ],
            vec![
                0, 0, 0, 0, // no high weights
                0, 1024, // 1 (original value: a)
                0, 0, // no high weights
            ],
            vec![
                0, 1024, 0, 1024, // 5 (original value: 5)
                0, 1024, // 1 (original value: a)
                0, 0, // (not specified)
            ],
        ];
        // Convert both to sets for comparing unordered values
        let wd_set = wd.into_iter().collect::<HashSet<Vec<i32>>>();
        let exp_set = exp.into_iter().collect::<HashSet<Vec<i32>>>();
        assert_eq!(exp_set, wd_set);

        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(3));
        fields.insert("group".to_owned(), FieldValue::String("c".to_owned()));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();
        let exp = vec![
            vec![
                0, 0, 1024, 1024, // 3 (original value: 3)
                0, 0, // no high weights
                0, 0, // no high weights
            ],
            vec![
                0, 0, 0, 0, // no high weights
                1024, 1024, // 3 (original value: c)
                0, 0, // no high weights
            ],
            vec![
                0, 0, 1024, 1024, // 3 (original value: 3)
                1024, 1024, // 3 (original value: c)
                0, 0, // (not specified)
            ],
        ];
        // Convert both to sets for comparing unordered values
        let wd_set = wd.into_iter().collect::<HashSet<Vec<i32>>>();
        let exp_set = exp.into_iter().collect::<HashSet<Vec<i32>>>();
        assert_eq!(exp_set, wd_set);

        // Case 3: When all fields are specified in input vector

        let mut fields = HashMap::with_capacity(3);
        fields.insert("age".to_owned(), FieldValue::Int(5));
        fields.insert("group".to_owned(), FieldValue::String("a".to_owned()));
        fields.insert("level".to_owned(), FieldValue::String("third".to_owned()));
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();

        let exp = vec![
            // dimensions to support queries for individual field age
            vec![
                0, 1024, 0, 1024, // 5 (original value: 5)
                0, 0, // no high weights
                0, 0, // no high weights
            ],
            // dimensions to support queries for individual field group
            vec![
                0, 0, 0, 0, // no high weights
                0, 1024, // 0 (original value: a)
                0, 0, // no high weights
            ],
            // dimensions to support queries for individual field level
            vec![
                0, 0, 0, 0, // no high weights
                0, 0, // no high weights
                1024, 1024, // 3 (original value: third)
            ],
            // dimensions to support AND(age, group)
            vec![
                0, 1024, 0, 1024, // 5 (original value: 5)
                0, 1024, // 1 (original value: a)
                0, 0, // no high weights
            ],
            // dimensions to support AND(age, level)
            vec![
                0, 1024, 0, 1024, // 5 (original value: 5)
                0, 0, // no high weights
                1024, 1024, // 3 (original value: third)
            ],
        ];
        // Convert both to sets for comparing unordered values
        let wd_set = wd.into_iter().collect::<HashSet<Vec<i32>>>();
        let exp_set = exp.into_iter().collect::<HashSet<Vec<i32>>>();
        assert_eq!(exp_set, wd_set);

        // Case 4: When no fields are specified in input vector
        let fields = HashMap::new();
        let wd = schema.weighted_dimensions(&fields, 1024).unwrap();
        assert!(wd.is_empty())
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
        let conditions = vec![SupportedCondition::And(hashset(vec!["age", "group"]))];
        let schema = MetadataSchema::new(vec![age, group], conditions).unwrap();
        let mut fields = HashMap::with_capacity(2);
        fields.insert("age".to_owned(), FieldValue::Int(100));
        fields.insert("group".to_owned(), FieldValue::String("c".to_owned()));
        match schema.weighted_dimensions(&fields, 1024) {
            Err(Error::InvalidFieldValue(_)) => {}
            _ => panic!(),
        }
    }

    /// Test serialization to and deserialization from cbor (as we've
    /// implemented custom serializer and deserializer for `FieldValue`)
    #[test]
    fn test_cbor_serde() {
        let age_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
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

        let slice = to_vec(&schema).unwrap();
        let orig: MetadataSchema = from_slice(&slice).unwrap();

        assert_eq!(schema.fields.len(), orig.fields.len());
    }

    #[test]
    fn test_pseudo_dimensions() {
        let age_values = (1..=2).map(FieldValue::Int).collect();
        let age = MetadataField::new("age".to_owned(), age_values).unwrap();

        let color_values: HashSet<FieldValue> = vec!["red", "yellow", "green"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let color = MetadataField::new("color".to_owned(), color_values).unwrap();

        let group_values: HashSet<FieldValue> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let group = MetadataField::new("group".to_owned(), group_values).unwrap();

        let conditions = vec![SupportedCondition::And(hashset(vec!["age", "color"]))];

        let schema = MetadataSchema::new(vec![age, color, group], conditions).unwrap();

        let weight = 1024;

        let total_bits = schema.num_total_dims();
        // Actual cardinalities = 2, 3, 4
        // Considering 0 value for not-specified case = 3, 4, 5
        // No. of bits required = 2, 2, 3 (total = 7)
        assert_eq!(7, total_bits);

        let root_dims = schema.pseudo_root_dimensions(weight);
        assert_eq!(vec![weight; 7], root_dims);

        let nonroot_dims = schema.pseudo_nonroot_dimensions(weight);
        // No. of values supported in 7 bits = 2^2, 4, 8
        // No. of non-zero values supported = 3, 3, 7 (total 13)
        let exp_total = 22; // 13 (individual) + 9 (age x color)
        assert_eq!(exp_total, nonroot_dims.len());

        let num_total_dims = schema.num_total_dims();
        assert_eq!(7, num_total_dims);

        // The result doesn't contain the pseudo root dims
        assert!(!nonroot_dims.iter().any(|dims| **dims == root_dims));

        // Test with only 1 field
        let status_values: HashSet<FieldValue> = vec!["todo", "done"]
            .into_iter()
            .map(|x| FieldValue::String(String::from(x)))
            .collect();
        let status = MetadataField::new("status".to_owned(), status_values).unwrap();

        let schema = MetadataSchema::new(vec![status], vec![]).unwrap();

        let root_dims = schema.pseudo_root_dimensions(1);
        assert_eq!(vec![1, 1], root_dims);

        let nonroot_dims = schema.pseudo_nonroot_dimensions(1);
        assert_eq!(2, nonroot_dims.len());
        // [1, 0]
        // [0, 1]
    }

    #[test]
    fn test_max_num_replicas() {
        let age_values: HashSet<FieldValue> = (1..=10).map(FieldValue::Int).collect();
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
        assert_eq!(6, schema.max_num_replicas());
    }
}
