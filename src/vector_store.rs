#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::config_loader::Config;
use crate::distance::DistanceFunction;
use crate::indexes::hnsw::offset_counter::HNSWIndexFileOffsetCounter;
use crate::indexes::hnsw::offset_counter::IndexFileId;
use crate::indexes::hnsw::types::HNSWHyperParams;
use crate::indexes::hnsw::types::QuantizedDenseVectorEmbedding;
use crate::indexes::hnsw::types::RawDenseVectorEmbedding;
use crate::indexes::hnsw::DenseInputEmbedding;
use crate::indexes::hnsw::HNSWIndex;
use crate::indexes::InternalSearchResult;
use crate::metadata;
use crate::metadata::fields_to_dimensions;
use crate::metadata::pseudo_level_probs;
use crate::metadata::MetadataFields;
use crate::metadata::MetadataSchema;
use crate::metadata::HIGH_WEIGHT;
use crate::models::cache_loader::HNSWIndexCache;
use crate::models::collection::Collection;
use crate::models::collection::RawVectorEmbedding;
use crate::models::common::*;
use crate::models::dot_product::dot_product_f32;
use crate::models::file_persist::*;
use crate::models::fixedset::PerformantFixedSet;
use crate::models::lazy_item::LazyItem;
use crate::models::prob_node::LatestNode;
use crate::models::prob_node::ProbNode;
use crate::models::prob_node::SharedLatestNode;
use crate::models::types::*;
use crate::models::versioning::VersionNumber;
use crate::quantization::{Quantization, StorageType};
use crate::storage::Storage;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::ptr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::RwLock;

#[allow(clippy::too_many_arguments)]
pub fn create_root_node(
    quantization_metric: &QuantizationMetric,
    storage_type: StorageType,
    dim: usize,
    prop_file: &RwLock<File>,
    version: VersionNumber,
    offset_counter: &HNSWIndexFileOffsetCounter,
    cache: &HNSWIndexCache,
    values_range: (f32, f32),
    hnsw_params: &HNSWHyperParams,
    distance_metric: DistanceMetric,
    metadata_schema: Option<&MetadataSchema>,
) -> Result<SharedLatestNode, WaCustomError> {
    let vec = (0..dim)
        .map(|_| {
            let mut rng = rand::thread_rng();

            let random_number: f32 = rng.gen_range(values_range.0..values_range.1);
            random_number
        })
        .collect::<Vec<f32>>();
    let vec_hash = InternalId::from(u32::MAX);

    let vector_list = Arc::new(quantization_metric.quantize(&vec, storage_type, values_range)?);

    let mut prop_file_guard = prop_file.write().unwrap();
    let location = write_prop_value_to_file(&vec_hash, &vector_list, &mut prop_file_guard)?;

    let prop_value = Arc::new(NodePropValue {
        id: vec_hash,
        vec: vector_list,
        location,
    });

    let prop_metadata = match metadata_schema {
        Some(schema) => {
            let mbits = schema.base_dimensions();
            let metadata = Arc::new(Metadata { mag: 0.0, mbits });
            // A root node is similar to a base replica node. Hence
            // the replica_id and prop_value's id are same.
            let replica_id = vec_hash;
            let location =
                write_prop_metadata_to_file(replica_id, metadata.clone(), &mut prop_file_guard)?;
            Some(Arc::new(NodePropMetadata {
                replica_id: vec_hash,
                vec: metadata,
                location,
            }))
        }
        None => None,
    };

    drop(prop_file_guard);

    let file_id = offset_counter.file_id();

    let mut root = LazyItem::new(
        ProbNode::new(
            HNSWLevel(0),
            version,
            prop_value.clone(),
            prop_metadata.clone(),
            ptr::null_mut(),
            ptr::null_mut(),
            hnsw_params.level_0_neighbors_count,
            distance_metric,
        ),
        file_id,
        offset_counter.next_level_0_offset(),
    );
    let mut root_ptr = LatestNode::new(root, offset_counter.next_latest_version_link_offset());

    let mut nodes = Vec::new();
    nodes.push(root_ptr);

    for l in 1..=hnsw_params.num_layers {
        let current_node = ProbNode::new(
            HNSWLevel(l),
            version,
            prop_value.clone(),
            prop_metadata.clone(),
            ptr::null_mut(),
            root_ptr,
            hnsw_params.neighbors_count,
            distance_metric,
        );

        let lazy_node = LazyItem::new(current_node, file_id, offset_counter.next_offset());

        let lazy_node_ptr =
            LatestNode::new(lazy_node, offset_counter.next_latest_version_link_offset());

        if let Some(prev_node) = unsafe { &*root }.get_lazy_data() {
            prev_node.set_parent(lazy_node_ptr);
        }
        root = lazy_node;
        root_ptr = lazy_node_ptr;

        nodes.push(lazy_node_ptr);
    }

    for item in nodes {
        write_lazy_item_latest_ptr_to_file(cache, item, file_id)?;
    }

    Ok(root_ptr)
}

/// Creates a pseudo root node in the index
///
/// This node is an independent node just like the main root node and
/// it's not connected to the main root node. All metadata nodes and
/// other pseudo nodes will be indexed under the pseudo root node. And
/// any queries that contain metadata filters will use this node as
/// the root when traversing the index.
#[allow(clippy::too_many_arguments)]
pub fn create_pseudo_root_node(
    quantization_metric: &QuantizationMetric,
    storage_type: StorageType,
    prop_file: &RwLock<File>,
    version_hash: VersionNumber,
    offset_counter: &HNSWIndexFileOffsetCounter,
    cache: &HNSWIndexCache,
    values_range: (f32, f32),
    hnsw_params: &HNSWHyperParams,
    distance_metric: DistanceMetric,
    metadata_schema: &MetadataSchema,
    vec: Vec<f32>,
    vec_hash: InternalId,
) -> Result<SharedLatestNode, WaCustomError> {
    let vector_list = Arc::new(quantization_metric.quantize(&vec, storage_type, values_range)?);

    let mut prop_file_guard = prop_file.write().unwrap();
    let location = write_prop_value_to_file(&vec_hash, &vector_list, &mut prop_file_guard)?;

    let prop_value = Arc::new(NodePropValue {
        id: vec_hash,
        vec: vector_list,
        location,
    });

    let prop_metadata = {
        let mbits = metadata_schema.pseudo_root_dimensions(HIGH_WEIGHT);
        let metadata = Arc::new(Metadata::from(mbits));
        // A pseudo root node is similar to a base replica node. Hence
        // the replica_id and prop_value's id are same.
        let replica_id = vec_hash;
        let location =
            write_prop_metadata_to_file(replica_id, metadata.clone(), &mut prop_file_guard)?;
        Some(Arc::new(NodePropMetadata {
            replica_id: vec_hash,
            vec: metadata,
            location,
        }))
    };

    drop(prop_file_guard);

    let file_id = offset_counter.file_id();

    let mut root = LazyItem::new(
        ProbNode::new(
            HNSWLevel(0),
            version_hash,
            prop_value.clone(),
            prop_metadata.clone(),
            ptr::null_mut(),
            ptr::null_mut(),
            hnsw_params.level_0_neighbors_count,
            distance_metric,
        ),
        file_id,
        offset_counter.next_level_0_offset(),
    );
    let mut root_ptr = LatestNode::new(root, offset_counter.next_latest_version_link_offset());

    let mut nodes = Vec::new();
    nodes.push(root_ptr);

    for l in 1..=hnsw_params.num_layers {
        let current_node = ProbNode::new(
            HNSWLevel(l),
            version_hash,
            prop_value.clone(),
            prop_metadata.clone(),
            ptr::null_mut(),
            root_ptr,
            hnsw_params.neighbors_count,
            distance_metric,
        );

        let lazy_node = LazyItem::new(current_node, file_id, offset_counter.next_offset());

        let lazy_node_ptr =
            LatestNode::new(lazy_node, offset_counter.next_latest_version_link_offset());

        if let Some(prev_node) = unsafe { &*root }.get_lazy_data() {
            prev_node.set_parent(lazy_node_ptr);
        }
        root = lazy_node;
        root_ptr = lazy_node_ptr;

        nodes.push(lazy_node_ptr);
    }

    for item in nodes {
        write_lazy_item_latest_ptr_to_file(cache, item, file_id)?;
    }

    Ok(root_ptr)
}

pub fn ann_search(
    config: &Config,
    hnsw_index: &HNSWIndex,
    vector_emb: QuantizedDenseVectorEmbedding,
    query_filter_dims: Option<&Vec<metadata::QueryFilterDimensions>>,
    current_lazy_item_latest_ptr: SharedLatestNode,
    cur_level: HNSWLevel,
    hnsw_params: &HNSWHyperParams,
) -> Result<Vec<(SharedLatestNode, MetricResult)>, WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = PerformantFixedSet::new(if cur_level.0 == 0 {
        hnsw_params.level_0_neighbors_count
    } else {
        hnsw_params.neighbors_count
    });
    skipm.insert(*vector_emb.hash_vec);

    let z = match query_filter_dims {
        Some(qf_dims) => {
            let mut z_candidates: Vec<(SharedLatestNode, MetricResult)> = vec![];
            // @TODO: Can we compute the z_candidates in parallel?
            for qfd in qf_dims {
                let mdims = Metadata::from(qfd);
                let z_with_mdims = traverse_find_nearest(
                    config,
                    hnsw_index,
                    current_lazy_item_latest_ptr,
                    &fvec,
                    None,
                    Some(&mdims),
                    &mut 0,
                    &mut skipm,
                    &hnsw_index.distance_metric.read().unwrap(),
                    false,
                    hnsw_params.ef_search,
                )?;

                for (node, dist) in z_with_mdims {
                    match dist {
                        MetricResult::CosineSimilarity(cs) => {
                            if cs.0 == -1.0 {
                                continue;
                            } else {
                                z_candidates.push((node, dist));
                            }
                        }
                        _ => z_candidates.push((node, dist)),
                    }
                }
            }

            // Sort candidates by distance (asc)
            z_candidates.sort_by_key(|c| Reverse(c.1));
            z_candidates
                .into_iter()
                .take(100) // Limit the number of results
                .collect::<Vec<_>>()
        }
        None => traverse_find_nearest(
            config,
            hnsw_index,
            current_lazy_item_latest_ptr,
            &fvec,
            None,
            None,
            &mut 0,
            &mut skipm,
            &hnsw_index.distance_metric.read().unwrap(),
            false,
            hnsw_params.ef_search,
        )?,
    };

    let mut z = if z.is_empty() {
        let current_lazy_item = unsafe { &*current_lazy_item_latest_ptr }.latest;
        let current_node = unsafe { &*current_lazy_item }.try_get_data(&hnsw_index.cache)?;
        let cur_node_id = &current_node.get_id();
        let dist = match query_filter_dims {
            // In case of metadata filters in query, we calculate the
            // distances between the cur_node and all query filter
            // dimensions and take the strongest match
            //
            // @TODO: Not sure if this additional computation is
            // required because eventually the same node is being
            // returned. Also need to consider performing the
            // following in parallel.
            Some(qf_dims) => {
                let cur_node_metadata = current_node.prop_metadata.clone().map(|pm| pm.vec.clone());
                let cur_node_data = VectorData {
                    id: Some(cur_node_id),
                    quantized_vec: &current_node.prop_value.vec,
                    metadata: cur_node_metadata.as_deref(),
                };
                let mut dists = vec![];
                for qfd in qf_dims {
                    let fvec_metadata = Metadata::from(qfd);
                    let fvec_data = VectorData {
                        id: None,
                        quantized_vec: &fvec,
                        metadata: Some(&fvec_metadata),
                    };
                    let d = hnsw_index.distance_metric.read().unwrap().calculate(
                        &fvec_data,
                        &cur_node_data,
                        false,
                    )?;
                    dists.push(d)
                }
                dists.into_iter().max().unwrap()
            }
            None => {
                let fvec_data = VectorData::without_metadata(None, &fvec);
                let cur_node_data =
                    VectorData::without_metadata(Some(cur_node_id), &current_node.prop_value.vec);
                hnsw_index.distance_metric.read().unwrap().calculate(
                    &fvec_data,
                    &cur_node_data,
                    false,
                )?
            }
        };
        vec![(current_lazy_item_latest_ptr, dist)]
    } else {
        z
    };

    let top_lazy_item_latest_ptr = z[0].0;
    let top_lazy_item = unsafe { &*top_lazy_item_latest_ptr }.latest;
    let top_node = unsafe { &*top_lazy_item }.try_get_data(&hnsw_index.cache)?;
    let child = top_node.get_child();

    if cur_level.0 != 0 {
        let results = ann_search(
            config,
            hnsw_index,
            vector_emb,
            query_filter_dims,
            child,
            HNSWLevel(cur_level.0 - 1),
            hnsw_params,
        )?;

        z.extend(results);
    };

    Ok(z)
}

pub fn finalize_ann_results(
    collection: &Collection,
    hnsw_index: &HNSWIndex,
    results: Vec<(SharedLatestNode, MetricResult)>,
    query: &[f32],
    top_k: Option<usize>,
    return_raw_text: bool,
) -> Result<Vec<InternalSearchResult>, WaCustomError> {
    let filtered = remove_duplicates_and_filter(hnsw_index, results, top_k, &hnsw_index.cache);
    let mut results = Vec::with_capacity(top_k.unwrap_or(filtered.len()));
    let mag_query = query.iter().map(|x| x * x).sum::<f32>().sqrt();

    for (internal_id, _) in filtered {
        let raw_emb = collection
            .get_raw_emb_by_internal_id(&internal_id)
            .ok_or_else(|| {
                WaCustomError::NotFound(format!("raw embedding not found for id={internal_id:?}"))
            })?;
        let dense_values = raw_emb.dense_values.as_ref().ok_or_else(|| {
            WaCustomError::NotFound("dense values not found for raw embedding".to_string())
        })?;
        let dp = dot_product_f32(query, dense_values);
        let mag_raw = dense_values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cs = dp / (mag_query * mag_raw);
        results.push((
            internal_id,
            Some(raw_emb.id.clone()),
            raw_emb.document_id.clone(),
            cs,
            if return_raw_text {
                raw_emb.text.clone()
            } else {
                None
            },
        ));
    }
    results.sort_unstable_by(|(_, _, _, a, _), (_, _, _, b, _)| b.total_cmp(a));
    if let Some(k) = top_k {
        results.truncate(k);
    }
    Ok(results)
}

/// Intermediate representation of the embedding in a form that's
/// ready for indexing.
///
/// i.e. with quantization performed and property values and metadata
/// fields converted into appropriate types.
struct IndexableEmbedding {
    prop_value: Arc<NodePropValue>,
    prop_metadata: Option<Arc<NodePropMetadata>>,
    overridden_level_probs: Option<Vec<(f64, u8)>>,
}

impl IndexableEmbedding {
    /// Returns the kind of replica node that will be created in the
    /// index
    fn node_kind(&self) -> ReplicaNodeKind {
        if self.overridden_level_probs.is_some() {
            ReplicaNodeKind::Pseudo
        } else {
            match &self.prop_metadata {
                Some(m) => {
                    if m.vec.mag == 0.0 {
                        ReplicaNodeKind::Base
                    } else {
                        ReplicaNodeKind::Metadata
                    }
                }
                None => ReplicaNodeKind::Base,
            }
        }
    }

    /// Returns the kind of root node that this embedding must be
    /// indexed under
    fn root_node_kind(&self) -> RootNodeKind {
        self.node_kind().root_node_kind()
    }
}

/// Computes "metadata replica sets" i.e. all metadata dimensions
/// along based on the metadata `schema`. This includes both base and
/// metadata dimensions.
///
/// Note that the first item in the result will correspond to the base
/// dimensions. This can be used as an assumption.
///
/// If `fields` is None or an empty map, it will return a vector with
/// a single item i.e. the base dimensions.
fn metadata_replica_set(
    schema: &MetadataSchema,
    fields: Option<&MetadataFields>,
) -> Result<Vec<Metadata>, WaCustomError> {
    let dims = fields_to_dimensions(schema, fields).map_err(WaCustomError::MetadataError)?;
    let replicas = dims.into_iter().map(Metadata::from).collect();
    Ok(replicas)
}

/// Returns a vector of `NodePropMetadata` instances based on the
/// collection `schema` and `metadata_fields` as per the following
/// cases:
///
///   - If both `schema` and `metadata_fields` are not None, then it
///     computes the metadata dimensions and returns
///     `NodePropMetadata` instances based on those.
///
///   - If `metadata_fields` is None but `schema` is not None
///     (i.e. the collection supports metadata filtering but the
///     vector being inserted doesn't specify any fields), then a
///     single `NodePropMetadata` is returned corresponding to the
///     base dimensions.
///
///   - If schema is None, None is returned
///
/// Note that this function performs IO by writing metadata to the
/// prop_file
fn prop_metadata_replicas(
    schema: Option<&MetadataSchema>,
    metadata_fields: Option<&MetadataFields>,
    prop_file: &RwLock<File>,
    base_id: &InternalId,
) -> Result<Option<Vec<NodePropMetadata>>, WaCustomError> {
    if schema.is_none() {
        return Ok(None);
    }

    let replica_set = if metadata_fields.is_some() {
        let rs = metadata_replica_set(schema.unwrap(), metadata_fields)?
            .into_iter()
            .enumerate()
            .map(|(i, r)| (InternalId::from(**base_id + i as u32), r))
            .collect::<Vec<(InternalId, Metadata)>>();
        Some(rs)
    } else {
        // If the collection supports metadata schema and
        // even if no metadata fields are specified with
        // the input vector, we create one replica with
        // base dimensions.
        match schema {
            Some(s) => {
                let mrset = metadata_replica_set(s, None)?;
                debug_assert_eq!(1, mrset.len());
                let result = mrset
                    .into_iter()
                    .enumerate()
                    .map(|(i, r)| (InternalId::from(**base_id + i as u32), r))
                    .collect::<Vec<(InternalId, Metadata)>>();
                Some(result)
            }
            // Following is unreachable as the case of schema being
            // None has already been handled
            None => None,
        }
    };

    if let Some(replicas) = replica_set {
        let mut result = Vec::with_capacity(replicas.len());
        for (id, m) in replicas {
            let mvalue = Arc::new(m);

            // Write metadata to the same prop file
            let mut prop_file_guard = prop_file.write().map_err(|_| {
                WaCustomError::LockError(
                    "Failed to acquire lock to write prop metadata".to_string(),
                )
            })?;
            let location = write_prop_metadata_to_file(id, mvalue.clone(), &mut prop_file_guard)?;
            drop(prop_file_guard);

            let prop_metadata = NodePropMetadata {
                replica_id: id,
                vec: mvalue,
                location,
            };
            result.push(prop_metadata);
        }
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

fn pseudo_metadata_replicas(
    schema: &MetadataSchema,
    prop_file: &RwLock<File>,
    base_id: &InternalId,
) -> Result<Vec<NodePropMetadata>, WaCustomError> {
    let dims = schema.pseudo_nonroot_dimensions(HIGH_WEIGHT);
    let replicas = dims
        .into_iter()
        .map(Metadata::from)
        .collect::<Vec<Metadata>>();
    // As pseudo_replicas will be created only at the time of index
    // initialization, it's ok to hold a single rw lock for writing
    // metadata for all replicas to the prop file
    let mut prop_file_guard = prop_file.write().map_err(|_| {
        WaCustomError::LockError("Failed to acquire lock to write prop metadata".to_string())
    })?;
    let mut result = Vec::with_capacity(replicas.len());
    for (i, m) in replicas.into_iter().enumerate() {
        let mvalue = Arc::new(m);
        // @NOTE: Why is an extra 1 added below - The base_id is of
        // pseudo root node which has already been created
        let replica_id = InternalId::from(**base_id + i as u32 + 1);
        let location =
            write_prop_metadata_to_file(replica_id, mvalue.clone(), &mut prop_file_guard)?;
        let prop_metadata = NodePropMetadata {
            replica_id,
            vec: mvalue,
            location,
        };
        result.push(prop_metadata);
    }
    drop(prop_file_guard);
    Ok(result)
}

/// Converts raw embeddings into `IndexableEmbedding` i.e. ready to be
/// indexed - with quantization performed and property values and
/// metadata fields converted into appropriate types.
///
/// If metadata filtering is supported for the collection, then one
/// input raw embedding may result in multiple `IndexableEmbedding`
/// instances.
fn preprocess_embedding(
    collection: &Collection,
    hnsw_index: &HNSWIndex,
    quantization_metric: &RwLock<QuantizationMetric>,
    raw_emb: &RawDenseVectorEmbedding,
) -> Result<Vec<IndexableEmbedding>, WaCustomError> {
    let quantization = quantization_metric.read().unwrap();
    let quantized_vec = Arc::new(quantization.quantize(
        &raw_emb.raw_vec,
        *hnsw_index.storage_type.read().unwrap(),
        *hnsw_index.values_range.read().unwrap(),
    )?);

    let base_id = raw_emb.hash_vec;

    let metadata_schema = collection.meta.metadata_schema.as_ref();
    let prop_file = &hnsw_index.cache.prop_file;

    let embeddings = if raw_emb.is_pseudo {
        let replicas = pseudo_metadata_replicas(metadata_schema.unwrap(), prop_file, &base_id)?;
        let num_levels = hnsw_index.levels_prob.len() - 1;
        let plp = pseudo_level_probs(num_levels as u8, replicas.len() as u16);

        // Find the pseudo root node so that we can reuse it's
        // prop_value in rest of the pseudo nodes
        let pseudo_root = hnsw_index.pseudo_root_vec.unwrap();
        let pseudo_root_lazy = unsafe { &*pseudo_root }.latest;
        let pseudo_root_node = unsafe { &*pseudo_root_lazy }.try_get_data(&hnsw_index.cache)?;

        let mut embeddings: Vec<IndexableEmbedding> = vec![];
        for prop_metadata in replicas.into_iter() {
            let emb = IndexableEmbedding {
                prop_value: pseudo_root_node.prop_value.clone(),
                prop_metadata: Some(Arc::new(prop_metadata)),
                overridden_level_probs: Some(plp.clone()),
            };
            embeddings.push(emb);
        }
        embeddings
    } else {
        // Write props to the prop file
        let mut prop_file_guard = hnsw_index.cache.prop_file.write().unwrap();
        let location = write_prop_value_to_file(&base_id, &quantized_vec, &mut prop_file_guard)
            .expect("failed to write prop");
        drop(prop_file_guard);

        let prop_value = Arc::new(NodePropValue {
            id: base_id,
            vec: quantized_vec.clone(),
            location,
        });

        let metadata_replicas = prop_metadata_replicas(
            collection.meta.metadata_schema.as_ref(),
            raw_emb.raw_metadata.as_ref(),
            &hnsw_index.cache.prop_file,
            &base_id,
        )?;
        match metadata_replicas {
            Some(replicas) => {
                let mut embeddings: Vec<IndexableEmbedding> = vec![];
                for prop_metadata in replicas.into_iter() {
                    let emb = IndexableEmbedding {
                        prop_value: prop_value.clone(),
                        prop_metadata: Some(Arc::new(prop_metadata)),
                        overridden_level_probs: None,
                    };
                    embeddings.push(emb);
                }
                embeddings
            }
            None => {
                let emb = IndexableEmbedding {
                    prop_value,
                    prop_metadata: None,
                    overridden_level_probs: None,
                };
                vec![emb]
            }
        }
    };

    Ok(embeddings)
}

pub fn index_embeddings(
    config: &Config,
    collection: &Collection,
    hnsw_index: &HNSWIndex,
    version: VersionNumber,
    vecs: Vec<DenseInputEmbedding>,
) -> Result<(), WaCustomError> {
    let hnsw_params_guard = hnsw_index.hnsw_params.read().unwrap();
    let embeddings = vecs
        .into_iter()
        .map(|vec| {
            let DenseInputEmbedding(id, values, metadata, is_pseudo) = vec;
            RawDenseVectorEmbedding {
                hash_vec: id,
                raw_vec: Arc::new(values),
                raw_metadata: metadata,
                is_pseudo,
            }
        })
        .map(|emb| {
            preprocess_embedding(
                collection,
                hnsw_index,
                &hnsw_index.quantization_metric,
                &emb,
            )
        })
        .collect::<Result<Vec<Vec<IndexableEmbedding>>, WaCustomError>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let offset_counter = hnsw_index.offset_counter.read().unwrap();
    let file_id = offset_counter.file_id();

    for emb in embeddings {
        let max_level = match &emb.overridden_level_probs {
            Some(lp) => get_max_insert_level(rand::random::<f32>().into(), lp),
            None => get_max_insert_level(rand::random::<f32>().into(), &hnsw_index.levels_prob),
        };
        // Start from root at highest level
        let root_entry = match &emb.root_node_kind() {
            RootNodeKind::Pseudo => hnsw_index.get_pseudo_root_vec().unwrap(),
            RootNodeKind::Main => hnsw_index.get_root_vec(),
        };
        let highest_level = HNSWLevel(hnsw_params_guard.num_layers);

        index_embedding(
            config,
            hnsw_index,
            ptr::null_mut(),
            emb.prop_value,
            emb.prop_metadata,
            root_entry,
            highest_level,
            version,
            file_id,
            &hnsw_params_guard,
            max_level, // Pass max_level to let index_embedding control node creation
            &offset_counter,
            *hnsw_index.distance_metric.read().unwrap(),
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn index_embedding(
    config: &Config,
    hnsw_index: &HNSWIndex,
    parent_lazy_item_latest_ptr: SharedLatestNode,
    prop_value: Arc<NodePropValue>,
    prop_metadata: Option<Arc<NodePropMetadata>>,
    current_lazy_item_latest_ptr: SharedLatestNode,
    cur_level: HNSWLevel,
    version: VersionNumber,
    file_id: IndexFileId,
    hnsw_params: &HNSWHyperParams,
    max_level: u8,
    offset_counter: &HNSWIndexFileOffsetCounter,
    distance_metric: DistanceMetric,
) -> Result<(), WaCustomError> {
    let fvec = &prop_value.vec;
    let mut skipm = PerformantFixedSet::new(if cur_level.0 == 0 {
        hnsw_params.level_0_neighbors_count
    } else {
        hnsw_params.neighbors_count
    });
    let new_node_id = match &prop_metadata {
        Some(metadata) => *metadata.replica_id,
        None => *prop_value.id,
    };
    skipm.insert(new_node_id);

    let current_lazy_item = unsafe { &*current_lazy_item_latest_ptr }.latest;
    let current_node = unsafe { &*current_lazy_item }.try_get_data(&hnsw_index.cache)?;

    let cur_node_id = &current_node.get_id();

    let mdims = prop_metadata.clone().map(|pm| pm.vec.clone());

    let z = traverse_find_nearest(
        config,
        hnsw_index,
        current_lazy_item_latest_ptr,
        fvec,
        Some(&prop_value.id),
        mdims.as_deref(),
        &mut 0,
        &mut skipm,
        &distance_metric,
        true,
        hnsw_params.ef_construction,
    )?;

    let z = if z.is_empty() {
        let fvec_data = VectorData {
            id: None,
            quantized_vec: fvec,
            metadata: mdims.as_deref(),
        };
        let cur_node_metadata = current_node.prop_metadata.clone().map(|pm| pm.vec.clone());
        let cur_node_data = VectorData {
            id: Some(cur_node_id),
            quantized_vec: &current_node.prop_value.vec,
            metadata: cur_node_metadata.as_deref(),
        };
        let dist = hnsw_index.distance_metric.read().unwrap().calculate(
            &fvec_data,
            &cur_node_data,
            true,
        )?;
        vec![(current_lazy_item_latest_ptr, dist)]
    } else {
        z
    };

    let top_lazy_item_latest_ptr = z[0].0;
    let top_lazy_item = unsafe { &*top_lazy_item_latest_ptr }.latest;
    let top_node = unsafe { &*top_lazy_item }.try_get_data(&hnsw_index.cache)?;
    let child = top_node.get_child();

    if cur_level.0 > max_level {
        // Just traverse down without creating nodes
        if cur_level.0 != 0 {
            index_embedding(
                config,
                hnsw_index,
                ptr::null_mut(),
                prop_value.clone(),
                prop_metadata.clone(),
                child,
                HNSWLevel(cur_level.0 - 1),
                version,
                file_id,
                hnsw_params,
                max_level,
                offset_counter,
                distance_metric,
            )?;
        }
    } else {
        let neighbors_count = if cur_level.0 == 0 {
            hnsw_params.level_0_neighbors_count
        } else {
            hnsw_params.neighbors_count
        };

        // Create node and edges at max_level and below
        let lazy_item_latest_ptr = create_node(
            version,
            file_id,
            cur_level,
            prop_value.clone(),
            prop_metadata.clone(),
            parent_lazy_item_latest_ptr,
            ptr::null_mut(),
            neighbors_count,
            offset_counter,
            distance_metric,
        );

        if let Some(parent_lazy_item_latest_ptr) = unsafe { parent_lazy_item_latest_ptr.as_ref() } {
            let parent_lazy_item = parent_lazy_item_latest_ptr.latest;
            unsafe { &*parent_lazy_item }
                .try_get_data(&hnsw_index.cache)
                .unwrap()
                .set_child(lazy_item_latest_ptr);
        }

        if cur_level.0 != 0 {
            index_embedding(
                config,
                hnsw_index,
                lazy_item_latest_ptr,
                prop_value.clone(),
                prop_metadata.clone(),
                child,
                HNSWLevel(cur_level.0 - 1),
                version,
                file_id,
                hnsw_params,
                max_level,
                offset_counter,
                distance_metric,
            )?;
        }

        create_node_edges(
            hnsw_index,
            lazy_item_latest_ptr,
            z,
            version,
            file_id,
            if cur_level.0 == 0 {
                hnsw_params.level_0_neighbors_count
            } else {
                hnsw_params.neighbors_count
            },
            offset_counter,
            distance_metric,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn create_node(
    version: VersionNumber,
    file_id: IndexFileId,
    hnsw_level: HNSWLevel,
    prop_value: Arc<NodePropValue>,
    prop_metadata: Option<Arc<NodePropMetadata>>,
    parent: SharedLatestNode,
    child: SharedLatestNode,
    neighbors_count: usize,
    offset_counter: &HNSWIndexFileOffsetCounter,
    distance_metric: DistanceMetric,
) -> SharedLatestNode {
    let offset = if hnsw_level.0 == 0 {
        offset_counter.next_level_0_offset()
    } else {
        offset_counter.next_offset()
    };
    let node = ProbNode::new(
        hnsw_level,
        version,
        prop_value,
        prop_metadata,
        parent,
        child,
        neighbors_count,
        distance_metric,
    );

    let lazy_item = LazyItem::new(node, file_id, offset);
    LatestNode::new(lazy_item, offset_counter.next_latest_version_link_offset())
}

#[allow(clippy::too_many_arguments)]
fn create_node_edges(
    hnsw_index: &HNSWIndex,
    lazy_item_latest_ptr: SharedLatestNode,
    neighbors: Vec<(SharedLatestNode, MetricResult)>,
    version: VersionNumber,
    file_id: IndexFileId,
    max_edges: usize,
    offset_counter: &HNSWIndexFileOffsetCounter,
    distance_metric: DistanceMetric,
) -> Result<(), WaCustomError> {
    let mut successful_edges = 0;
    let mut neighbors_to_update = Vec::new();

    let lazy_item = unsafe { &*lazy_item_latest_ptr }.latest;
    let node = unsafe { &*lazy_item }.try_get_data(&hnsw_index.cache)?;

    let node_id = node.get_id();

    // First loop: Handle neighbor connections and collect updates
    for (neighbor_lazy_item_latest_ptr, dist) in neighbors {
        if successful_edges >= max_edges {
            break;
        }

        let (neighbor_lazy_item, newly_created) = LatestNode::get_or_create_version(
            neighbor_lazy_item_latest_ptr,
            version,
            &hnsw_index.versions_synchronization_map,
            &hnsw_index.cache,
            file_id,
            offset_counter,
        )?;

        let neighbor_node = unsafe { &*neighbor_lazy_item }.try_get_data(&hnsw_index.cache)?;
        let neighbor_node_id = neighbor_node.get_id();

        assert_eq!(neighbor_node.version.load(Ordering::Relaxed), *version);

        // Ensure that a metadata node gets connected to a pseudo node
        // only if there's a perfect match
        match (
            neighbor_node.replica_node_kind(),
            node.replica_node_kind(),
            &dist,
        ) {
            (
                ReplicaNodeKind::Pseudo,
                ReplicaNodeKind::Metadata,
                MetricResult::CosineSimilarity(cs),
            ) => {
                if cs.0 != 1.0 {
                    continue;
                }
            }
            (
                ReplicaNodeKind::Metadata,
                ReplicaNodeKind::Metadata,
                MetricResult::CosineSimilarity(cs),
            ) => {
                if cs.0 == -1.0 {
                    continue;
                }
            }
            _ => {}
        }

        let neighbor_inserted_idx = node.add_neighbor(
            neighbor_node_id,
            neighbor_lazy_item_latest_ptr,
            dist,
            &hnsw_index.cache,
            distance_metric,
        );

        let neighbour_update_info = if let Some(neighbor_inserted_idx) = neighbor_inserted_idx {
            let node_inserted_idx = neighbor_node.add_neighbor(
                node_id,
                lazy_item_latest_ptr,
                dist,
                &hnsw_index.cache,
                distance_metric,
            );
            if let Some(idx) = node_inserted_idx {
                successful_edges += 1;
                Some((idx, dist))
            } else {
                node.remove_neighbor_by_index_and_id(neighbor_inserted_idx, neighbor_node_id);
                None
            }
        } else {
            None
        };

        if newly_created {
            write_lazy_item_to_file(&hnsw_index.cache, neighbor_lazy_item, file_id)?;
        } else if let Some((idx, dist)) = neighbour_update_info {
            neighbors_to_update.push((neighbor_lazy_item_latest_ptr, idx, dist));
        }
    }

    // Second loop: Batch process file operations for updated neighbors
    if !neighbors_to_update.is_empty() {
        let bufman = hnsw_index.cache.bufmans.get(file_id)?;
        let cursor = bufman.open_cursor()?;
        let mut current_node_link = Vec::with_capacity(8);
        current_node_link.extend(node.get_id().to_le_bytes());
        current_node_link.extend(
            unsafe { &*lazy_item_latest_ptr }
                .file_offset
                .0
                .to_le_bytes(),
        );

        for (neighbor_lazy_item_latest_ptr, neighbor_idx, dist) in neighbors_to_update {
            let neighbor_lazy_item = unsafe { &*neighbor_lazy_item_latest_ptr }.latest;
            let offset = unsafe { &*neighbor_lazy_item }.file_index.offset;
            let mut current_node_link_with_dist = Vec::with_capacity(13);
            current_node_link_with_dist.clone_from(&current_node_link);
            let (tag, value) = dist.get_tag_and_value();
            current_node_link_with_dist.push(tag);
            current_node_link_with_dist.extend(value.to_le_bytes());

            let neighbor_offset = (offset.0 + 31) + neighbor_idx as u32 * 13;
            bufman.seek_with_cursor(cursor, neighbor_offset as u64)?;
            bufman.update_with_cursor(cursor, &current_node_link_with_dist)?;
        }

        bufman.close_cursor(cursor)?;
    }

    write_lazy_item_latest_ptr_to_file(&hnsw_index.cache, lazy_item_latest_ptr, file_id)?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn traverse_find_nearest(
    config: &Config,
    hnsw_index: &HNSWIndex,
    start_lazy_item_latest_ptr: SharedLatestNode,
    fvec: &Storage,
    fvec_id: Option<&InternalId>,
    mdims: Option<&Metadata>,
    nodes_visited: &mut u32,
    skipm: &mut PerformantFixedSet,
    distance_metric: &DistanceMetric,
    is_indexing: bool,
    ef: u32,
) -> Result<Vec<(SharedLatestNode, MetricResult)>, WaCustomError> {
    let mut candidate_queue = BinaryHeap::new();
    let mut results = Vec::new();

    let start_lazy_item = unsafe { &*start_lazy_item_latest_ptr }.latest;
    let start_node = unsafe { &*start_lazy_item }.try_get_data(&hnsw_index.cache)?;

    let fvec_data = VectorData {
        id: fvec_id,
        quantized_vec: fvec,
        metadata: mdims,
    };

    let start_metadata = start_node.prop_metadata.clone().map(|pm| pm.vec.clone());
    let start_node_id = start_node.get_id();
    let start_vec_data = VectorData {
        id: Some(&start_node_id),
        quantized_vec: &start_node.prop_value.vec,
        metadata: start_metadata.as_deref(),
    };
    let start_dist = distance_metric.calculate(&fvec_data, &start_vec_data, is_indexing)?;

    let start_id = *start_node.get_id();
    skipm.insert(start_id);
    candidate_queue.push((start_dist, start_lazy_item_latest_ptr));

    while let Some((dist, current_lazy_item_latest_ptr)) = candidate_queue.pop() {
        if *nodes_visited >= ef {
            break;
        }
        *nodes_visited += 1;

        let current_lazy_item = unsafe { &*current_lazy_item_latest_ptr }.latest;
        results.push((dist, current_lazy_item_latest_ptr));
        let current_node = unsafe { &*current_lazy_item }.try_get_data(&hnsw_index.cache)?;

        let _lock = current_node.freeze();
        for neighbor in current_node
            .get_neighbors_raw()
            .iter()
            .take(config.search.shortlist_size)
        {
            let (neighbor_id, neighbor_lazy_item_latest_ptr) = unsafe {
                if let Some((id, node, _)) = neighbor.load(Ordering::Relaxed).as_ref() {
                    (*id, *node)
                } else {
                    continue;
                }
            };

            if !skipm.is_member(*neighbor_id) {
                let neighbor_lazy_item = unsafe { &*neighbor_lazy_item_latest_ptr }.latest;
                let neighbor_node =
                    unsafe { &*neighbor_lazy_item }.try_get_data(&hnsw_index.cache)?;
                let neighbor_metadata =
                    neighbor_node.prop_metadata.clone().map(|pm| pm.vec.clone());
                let neighbor_node_id = neighbor_node.get_id();
                let neighbor_vec_data = VectorData {
                    id: Some(&neighbor_node_id),
                    quantized_vec: &neighbor_node.prop_value.vec,
                    metadata: neighbor_metadata.as_deref(),
                };
                let dist =
                    distance_metric.calculate(&fvec_data, &neighbor_vec_data, is_indexing)?;
                skipm.insert(*neighbor_id);
                candidate_queue.push((dist, neighbor_lazy_item_latest_ptr));
            }
        }
    }

    let final_len = if is_indexing { 64 } else { 100 };

    if results.len() > final_len {
        results.select_nth_unstable_by(final_len, |(a, _), (b, _)| b.cmp(a));
        results.truncate(final_len);
    }

    results.sort_unstable_by(|(a, _), (b, _)| b.cmp(a));

    Ok(results.into_iter().map(|(sim, node)| (node, sim)).collect())
}

pub fn delete_embedding(
    config: &Config,
    hnsw_index: &HNSWIndex,
    version: VersionNumber,
    id: InternalId,
    raw_emb: &RawVectorEmbedding,
) -> Result<(), WaCustomError> {
    let Some(raw_vec) = &raw_emb.dense_values else {
        return Ok(());
    };

    let quantized_vec = hnsw_index.quantization_metric.read().unwrap().quantize(
        raw_vec,
        *hnsw_index.storage_type.read().unwrap(),
        *hnsw_index.values_range.read().unwrap(),
    )?;

    let mut cur_entry = hnsw_index.get_root_vec();

    let hnsw_params = hnsw_index.hnsw_params.read().unwrap();
    let distance_metric = *hnsw_index.distance_metric.read().unwrap();
    let offset_counter = hnsw_index.offset_counter.read().unwrap();
    let file_id = offset_counter.file_id();

    for level in (0..=hnsw_params.num_layers).rev() {
        let mut skipm = PerformantFixedSet::new(if level == 0 {
            hnsw_params.level_0_neighbors_count
        } else {
            hnsw_params.neighbors_count
        });
        let mut results = traverse_find_nearest(
            config,
            hnsw_index,
            cur_entry,
            &quantized_vec,
            Some(&id),
            None,
            &mut 0,
            &mut skipm,
            &distance_metric,
            false,
            512,
        )?;

        if results.is_empty() {
            cur_entry = unsafe { &*(*cur_entry).latest }
                .try_get_data(&hnsw_index.cache)?
                .get_child();
            continue;
        } else {
            cur_entry = unsafe { &*(*results[0].0).latest }
                .try_get_data(&hnsw_index.cache)?
                .get_child();
        }

        let mut idx = None;

        for (i, (lazy_item_latest_ptr, _)) in results.iter().enumerate() {
            let node =
                unsafe { &*(**lazy_item_latest_ptr).latest }.try_get_data(&hnsw_index.cache)?;
            if node.get_id() == id {
                idx = Some(i);
                break;
            }
        }

        let Some(idx) = idx else {
            continue;
        };

        let (lazy_item_latest_ptr, _) = results.swap_remove(idx);
        let lazy_item = unsafe { &*lazy_item_latest_ptr }.latest;
        let node = unsafe { &*lazy_item }.try_get_data(&hnsw_index.cache)?;

        let _lock = node.lowest_index.write();

        for neighbor in node.get_neighbors_raw() {
            let neighbor_lazy_item_latest_ptr = unsafe {
                if let Some((_, neighbor_lazy_item_latest_ptr, _)) =
                    neighbor.load(Ordering::Relaxed).as_ref()
                {
                    *neighbor_lazy_item_latest_ptr
                } else {
                    continue;
                }
            };

            let (neighbor_lazy_item, newly_created) = LatestNode::get_or_create_version(
                neighbor_lazy_item_latest_ptr,
                version,
                &hnsw_index.versions_synchronization_map,
                &hnsw_index.cache,
                file_id,
                &offset_counter,
            )?;

            let neighbor_node = unsafe { &*neighbor_lazy_item }.try_get_data(&hnsw_index.cache)?;

            let removed = neighbor_node.remove_neighbor_by_id(id);

            if removed && neighbor_node.is_neighbors_empty() {
                let neighbor_id = neighbor_node.get_id();
                let neighbor_metadata =
                    neighbor_node.prop_metadata.clone().map(|pm| pm.vec.clone());
                let neighbor_vec = VectorData {
                    id: Some(&neighbor_id),
                    quantized_vec: &neighbor_node.prop_value.vec,
                    metadata: neighbor_metadata.as_deref(),
                };
                let mut results = results.clone();
                let mut neighbor_idx = None;
                for (idx, (lazy_item_latest_ptr, _)) in results.iter().enumerate() {
                    let node = unsafe { &*(**lazy_item_latest_ptr).latest }
                        .try_get_data(&hnsw_index.cache)?;
                    if node.get_id() == neighbor_node.get_id() {
                        neighbor_idx = Some(idx);
                        break;
                    }
                }
                if let Some(idx) = neighbor_idx {
                    results.swap_remove(idx);
                }
                for (lazy_item_latest_ptr, score) in results.iter_mut() {
                    let node = unsafe { &*(**lazy_item_latest_ptr).latest }
                        .try_get_data(&hnsw_index.cache)?;
                    let metadata = node.prop_metadata.clone().map(|pm| pm.vec.clone());
                    let node_id = node.get_id();
                    let vec_data = VectorData {
                        id: Some(&node_id),
                        quantized_vec: &node.prop_value.vec,
                        metadata: metadata.as_deref(),
                    };
                    *score = distance_metric.calculate(&neighbor_vec, &vec_data, false)?;
                }

                results.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));

                create_node_edges(
                    hnsw_index,
                    neighbor_lazy_item_latest_ptr,
                    results,
                    version,
                    file_id,
                    if level == 0 {
                        hnsw_params.level_0_neighbors_count
                    } else {
                        hnsw_params.neighbors_count
                    },
                    &offset_counter,
                    distance_metric,
                )?;
            }

            if newly_created {
                write_lazy_item_to_file(&hnsw_index.cache, neighbor_lazy_item, file_id)?;
            }
        }

        unsafe {
            drop(Box::from_raw(lazy_item));
            drop(Box::from_raw(lazy_item_latest_ptr));
        }
    }

    Ok(())
}
