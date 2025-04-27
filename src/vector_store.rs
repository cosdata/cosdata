#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::config_loader::Config;
use crate::config_loader::VectorsIndexingMode;
use crate::distance::DistanceFunction;
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
use crate::models::buffered_io::*;
use crate::models::collection::Collection;
use crate::models::collection_transaction::CollectionTransaction;
use crate::models::common::*;
use crate::models::dot_product::dot_product_f32;
use crate::models::file_persist::*;
use crate::models::fixedset::PerformantFixedSet;
use crate::models::prob_lazy_load::lazy_item::FileIndex;
use crate::models::prob_lazy_load::lazy_item::ProbLazyItem;
use crate::models::prob_node::ProbNode;
use crate::models::prob_node::SharedNode;
use crate::models::types::*;
use crate::models::versioning::Hash;
use crate::quantization::{Quantization, StorageType};
use crate::storage::Storage;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
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
    hash: Hash,
    index_manager: &BufferManagerFactory<Hash>,
    level_0_index_manager: &BufferManagerFactory<Hash>,
    values_range: (f32, f32),
    hnsw_params: &HNSWHyperParams,
    distance_metric: DistanceMetric,
    metadata_schema: Option<&MetadataSchema>,
) -> Result<SharedNode, WaCustomError> {
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
            let location = write_prop_metadata_to_file(metadata.clone(), &mut prop_file_guard)?;
            Some(Arc::new(NodePropMetadata {
                vec: metadata,
                location,
            }))
        }
        None => None,
    };

    drop(prop_file_guard);

    let mut root = ProbLazyItem::new(
        ProbNode::new(
            HNSWLevel(0),
            prop_value.clone(),
            prop_metadata.clone(),
            ptr::null_mut(),
            ptr::null_mut(),
            hnsw_params.level_0_neighbors_count,
            distance_metric,
        ),
        hash,
        0,
        true,
        FileOffset(0),
    );

    let mut nodes = Vec::new();
    nodes.push(root);

    let mut offset = 0;
    let node_size = ProbNode::get_serialized_size(hnsw_params.neighbors_count) as u32;

    for l in 1..=hnsw_params.num_layers {
        let current_node = ProbNode::new(
            HNSWLevel(l),
            prop_value.clone(),
            prop_metadata.clone(),
            ptr::null_mut(),
            root,
            hnsw_params.neighbors_count,
            distance_metric,
        );

        let lazy_node = ProbLazyItem::new(current_node, hash, 0, false, FileOffset(offset));
        offset += node_size;

        if let Some(prev_node) = unsafe { &*root }.get_lazy_data() {
            prev_node.set_parent(lazy_node);
        }
        root = lazy_node;

        nodes.push(lazy_node);
    }

    for item in nodes {
        write_node_to_file(item, index_manager, level_0_index_manager, hash)?;
    }

    Ok(root)
}

pub fn ann_search(
    config: &Config,
    hnsw_index: &HNSWIndex,
    vector_emb: QuantizedDenseVectorEmbedding,
    query_filter_dims: Option<&Vec<metadata::QueryFilterDimensions>>,
    cur_entry: SharedNode,
    cur_level: HNSWLevel,
    hnsw_params: &HNSWHyperParams,
) -> Result<Vec<(SharedNode, MetricResult)>, WaCustomError> {
    let fvec = vector_emb.quantized_vec.clone();
    let mut skipm = PerformantFixedSet::new(if cur_level.0 == 0 {
        hnsw_params.level_0_neighbors_count
    } else {
        hnsw_params.neighbors_count
    });
    skipm.insert(*vector_emb.hash_vec);

    let z = match query_filter_dims {
        Some(qf_dims) => {
            let mut z_candidates: Vec<(SharedNode, MetricResult)> = vec![];
            // @TODO: Can we compute the z_candidates in parallel?
            for qfd in qf_dims {
                let mdims = Metadata::from(qfd);
                let mut z_with_mdims = traverse_find_nearest(
                    config,
                    hnsw_index,
                    cur_entry,
                    &fvec,
                    None,
                    Some(&mdims),
                    &mut 0,
                    &mut skipm,
                    &hnsw_index.distance_metric.read().unwrap(),
                    false,
                    hnsw_params.ef_search,
                )?;
                // @NOTE: We're considering nearest neighbors computed
                // for all metadata dims. Here we're relying on
                // `traverse_find_nearest` to deduplicate the results
                // (thanks to the `skipm` argument)
                z_candidates.append(&mut z_with_mdims);
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
            cur_entry,
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
        let cur_node = unsafe { &*cur_entry }.try_get_data(&hnsw_index.cache)?;
        let cur_node_id = &cur_node.prop_value.id;
        let dist = match query_filter_dims {
            // In case of metadata filters in query, we calculate the
            // distances between the cur_node and all query filter
            // dimensions and take the minimum.
            //
            // @TODO: Not sure if this additional computation is
            // required because eventually the same node is being
            // returned. Also need to consider performing the
            // following in parallel.
            Some(qf_dims) => {
                let cur_node_metadata = cur_node.prop_metadata.clone().map(|pm| pm.vec.clone());
                let cur_node_data = VectorData {
                    id: Some(cur_node_id),
                    quantized_vec: &cur_node.prop_value.vec,
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
                dists.into_iter().min().unwrap()
            }
            None => {
                let fvec_data = VectorData::without_metadata(None, &fvec);
                let cur_node_data =
                    VectorData::without_metadata(Some(cur_node_id), &cur_node.prop_value.vec);
                hnsw_index.distance_metric.read().unwrap().calculate(
                    &fvec_data,
                    &cur_node_data,
                    false,
                )?
            }
        };
        vec![(cur_entry, dist)]
    } else {
        z
    };

    if cur_level.0 != 0 {
        let results = ann_search(
            config,
            hnsw_index,
            vector_emb,
            query_filter_dims,
            unsafe { &*z[0].0 }
                .try_get_data(&hnsw_index.cache)?
                .get_child(),
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
    results: Vec<(SharedNode, MetricResult)>,
    query: &[f32],
    top_k: Option<usize>,
    return_raw_text: bool,
) -> Result<Vec<InternalSearchResult>, WaCustomError> {
    let filtered = remove_duplicates_and_filter(hnsw_index, results, top_k, &hnsw_index.cache);
    let mut results = Vec::with_capacity(top_k.unwrap_or(filtered.len()));
    let mag_query = query.iter().map(|x| x * x).sum::<f32>().sqrt();

    for (orig_id, _) in filtered {
        let raw_emb = collection
            .internal_to_external_map
            .get_latest(&orig_id)
            .ok_or_else(|| WaCustomError::NotFound("raw embedding not found".to_string()))?;
        let dense_values = raw_emb.dense_values.as_ref().ok_or_else(|| {
            WaCustomError::NotFound("dense values not found for raw embedding".to_string())
        })?;
        let dp = dot_product_f32(query, dense_values);
        let mag_raw = dense_values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cs = dp / (mag_query * mag_raw);
        results.push((
            orig_id,
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

/// Computes "metadata replica sets" i.e. all metadata dimensions
/// along with an id for the provided metadata `fields` and based on
/// the metadata `schema`. If `fields` is None or an empty map, it
/// will return a vector with a single item i.e. the base dimensions.
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
) -> Result<Option<Vec<NodePropMetadata>>, WaCustomError> {
    if schema.is_none() {
        return Ok(None);
    }

    let replica_set = if metadata_fields.is_some() {
        Some(metadata_replica_set(schema.unwrap(), metadata_fields)?)
    } else {
        // If the collection supports metadata schema and
        // even if no metadata fields are specified with
        // the input vector, we create one replica with
        // base dimensions.
        match schema {
            Some(s) => {
                let mrset = metadata_replica_set(s, None)?;
                debug_assert_eq!(1, mrset.len());
                Some(mrset)
            }
            // Following is unreachable as the case of schema being
            // None has already been handled
            None => None,
        }
    };

    if let Some(replicas) = replica_set {
        let mut result = Vec::with_capacity(replicas.len());
        for m in replicas {
            let mvalue = Arc::new(m);

            // Write metadata to the same prop file
            let mut prop_file_guard = prop_file.write().map_err(|_| {
                WaCustomError::LockError(
                    "Failed to acquire lock to write prop metadata".to_string(),
                )
            })?;
            let location = write_prop_metadata_to_file(mvalue.clone(), &mut prop_file_guard)?;
            drop(prop_file_guard);

            let prop_metadata = NodePropMetadata {
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
) -> Result<Vec<NodePropMetadata>, WaCustomError> {
    let dims = schema.pseudo_weighted_dimensions(HIGH_WEIGHT);
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
    for m in replicas {
        let mvalue = Arc::new(m);
        let location = write_prop_metadata_to_file(mvalue.clone(), &mut prop_file_guard)?;
        let prop_metadata = NodePropMetadata {
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
) -> Vec<IndexableEmbedding> {
    let quantization = quantization_metric.read().unwrap();
    let quantized_vec = Arc::new(
        quantization
            .quantize(
                &raw_emb.raw_vec,
                *hnsw_index.storage_type.read().unwrap(),
                *hnsw_index.values_range.read().unwrap(),
            )
            .expect("Quantization failed"),
    );

    // Write props to the prop file
    let mut prop_file_guard = hnsw_index.cache.prop_file.write().unwrap();
    let location =
        write_prop_value_to_file(&raw_emb.hash_vec, &quantized_vec, &mut prop_file_guard)
            .expect("failed to write prop");
    drop(prop_file_guard);

    let base_id = if raw_emb.is_pseudo {
        raw_emb.hash_vec
    } else {
        raw_emb.hash_vec * hnsw_index.max_replica_per_node as u32
    };

    let prop_value = Arc::new(NodePropValue {
        id: base_id,
        vec: quantized_vec.clone(),
        location,
    });

    let metadata_schema = collection.meta.metadata_schema.as_ref();
    let prop_file = &hnsw_index.cache.prop_file;

    // @TODO(vineet): Remove unwraps
    if raw_emb.is_pseudo {
        let replicas = pseudo_metadata_replicas(metadata_schema.unwrap(), prop_file).unwrap();
        // @TODO(vineet): This is hacky
        let num_levels = hnsw_index.levels_prob.len() - 1;
        let plp = pseudo_level_probs(num_levels as u8, replicas.len() as u16);
        let mut embeddings: Vec<IndexableEmbedding> = vec![];
        let mut is_first_overrideen = false;
        for (replica_id, prop_metadata) in replicas.into_iter().enumerate() {
            let overridden_level_probs = if !is_first_overrideen {
                is_first_overrideen = true;
                plp.iter()
                    .map(|(_, lev)| (0.0, *lev))
                    .collect::<Vec<(f64, u8)>>()
            } else {
                plp.clone()
            };
            let emb = IndexableEmbedding {
                prop_value: Arc::new(NodePropValue {
                    id: InternalId::from(*base_id + replica_id as u32 + 1),
                    vec: quantized_vec.clone(),
                    location,
                }),
                prop_metadata: Some(Arc::new(prop_metadata)),
                overridden_level_probs: Some(overridden_level_probs),
            };
            embeddings.push(emb);
        }
        embeddings
    } else {
        let metadata_replicas = prop_metadata_replicas(
            collection.meta.metadata_schema.as_ref(),
            raw_emb.raw_metadata.as_ref(),
            &hnsw_index.cache.prop_file,
        )
        .unwrap();
        match metadata_replicas {
            Some(replicas) => {
                let mut embeddings: Vec<IndexableEmbedding> = vec![];
                for (replica_id, prop_metadata) in replicas.into_iter().enumerate() {
                    let emb = IndexableEmbedding {
                        prop_value: Arc::new(NodePropValue {
                            id: InternalId::from(*base_id + replica_id as u32 + 1),
                            vec: quantized_vec.clone(),
                            location,
                        }),
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
    }
}

pub fn index_embeddings(
    config: &Config,
    collection: &Collection,
    hnsw_index: &HNSWIndex,
    transaction: &CollectionTransaction,
    vecs: Vec<DenseInputEmbedding>,
) -> Result<(), WaCustomError> {
    let hnsw_params_guard = hnsw_index.hnsw_params.read().unwrap();
    let index = |vecs: Vec<DenseInputEmbedding>| {
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
            .flat_map(|emb| {
                preprocess_embedding(
                    collection,
                    hnsw_index,
                    &hnsw_index.quantization_metric,
                    &emb,
                )
            })
            .collect::<Vec<IndexableEmbedding>>();
        for emb in embeddings {
            let max_level =
                get_max_insert_level(rand::random::<f32>().into(), &hnsw_index.levels_prob);
            // Start from root at highest level
            let root_entry = hnsw_index.get_root_vec();
            let highest_level = HNSWLevel(hnsw_params_guard.num_layers);

            index_embedding(
                config,
                hnsw_index,
                ptr::null_mut(),
                emb.prop_value,
                emb.prop_metadata,
                root_entry,
                highest_level,
                transaction.id,
                transaction.version_number,
                transaction.lazy_item_versions_table.clone(),
                &hnsw_params_guard,
                max_level, // Pass max_level to let index_embedding control node creation
                &mut || transaction.get_new_node_offset(),
                &mut || transaction.get_new_level_0_node_offset(),
                *hnsw_index.distance_metric.read().unwrap(),
            )?;
        }
        Ok::<_, WaCustomError>(())
    };

    match config.indexing.mode {
        VectorsIndexingMode::Sequential => {
            index(vecs)?;
        }
        VectorsIndexingMode::Batch { batch_size } => {
            vecs.into_par_iter()
                .chunks(batch_size)
                .map(index)
                .collect::<Result<(), _>>()?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn index_embedding(
    config: &Config,
    hnsw_index: &HNSWIndex,
    parent: SharedNode,
    prop_value: Arc<NodePropValue>,
    prop_metadata: Option<Arc<NodePropMetadata>>,
    cur_entry: SharedNode,
    cur_level: HNSWLevel,
    version: Hash,
    version_number: u16,
    lazy_item_versions_table: Arc<TSHashTable<(InternalId, u16, u8), SharedNode>>,
    hnsw_params: &HNSWHyperParams,
    max_level: u8,
    offset_fn: &mut impl FnMut() -> u32,
    level_0_offset_fn: &mut impl FnMut() -> u32,
    distance_metric: DistanceMetric,
) -> Result<(), WaCustomError> {
    let fvec = &prop_value.vec;
    let mut skipm = PerformantFixedSet::new(if cur_level.0 == 0 {
        hnsw_params.level_0_neighbors_count
    } else {
        hnsw_params.neighbors_count
    });
    skipm.insert(*prop_value.id);

    let cur_node = unsafe { &*cur_entry }.try_get_data(&hnsw_index.cache)?;

    let cur_node_id = &cur_node.prop_value.id;

    let mdims = prop_metadata.clone().map(|pm| pm.vec.clone());

    let z = traverse_find_nearest(
        config,
        hnsw_index,
        cur_entry,
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
        let cur_node_metadata = cur_node.prop_metadata.clone().map(|pm| pm.vec.clone());
        let cur_node_data = VectorData {
            id: Some(cur_node_id),
            quantized_vec: &cur_node.prop_value.vec,
            metadata: cur_node_metadata.as_deref(),
        };
        let dist = hnsw_index.distance_metric.read().unwrap().calculate(
            &fvec_data,
            &cur_node_data,
            true,
        )?;
        vec![(cur_entry, dist)]
    } else {
        z
    };
    if cur_level.0 > max_level {
        // Just traverse down without creating nodes
        if cur_level.0 != 0 {
            index_embedding(
                config,
                hnsw_index,
                ptr::null_mut(),
                prop_value.clone(),
                prop_metadata.clone(),
                unsafe { &*z[0].0 }
                    .try_get_data(&hnsw_index.cache)?
                    .get_child(),
                HNSWLevel(cur_level.0 - 1),
                version,
                version_number,
                lazy_item_versions_table.clone(),
                hnsw_params,
                max_level,
                offset_fn,
                level_0_offset_fn,
                distance_metric,
            )?;
        }
    } else {
        let (neighbors_count, is_level_0, offset) = if cur_level.0 == 0 {
            (
                hnsw_params.level_0_neighbors_count,
                true,
                level_0_offset_fn(),
            )
        } else {
            (hnsw_params.neighbors_count, false, offset_fn())
        };

        // Create node and edges at max_level and below
        let lazy_node = create_node(
            version,
            version_number,
            cur_level,
            prop_value.clone(),
            prop_metadata.clone(),
            parent,
            ptr::null_mut(),
            neighbors_count,
            is_level_0,
            offset,
            distance_metric,
        );

        let node = unsafe { &*lazy_node }.get_lazy_data().unwrap();

        if let Some(parent) = unsafe { parent.as_ref() } {
            parent
                .try_get_data(&hnsw_index.cache)
                .unwrap()
                .set_child(lazy_node);
        }

        if cur_level.0 != 0 {
            index_embedding(
                config,
                hnsw_index,
                lazy_node,
                prop_value.clone(),
                prop_metadata.clone(),
                unsafe { &*z[0].0 }
                    .try_get_data(&hnsw_index.cache)?
                    .get_child(),
                HNSWLevel(cur_level.0 - 1),
                version,
                version_number,
                lazy_item_versions_table.clone(),
                hnsw_params,
                max_level,
                offset_fn,
                level_0_offset_fn,
                distance_metric,
            )?;
        }

        let (is_level_0, offset_fn): (bool, &mut dyn FnMut() -> u32) = if cur_level.0 == 0 {
            (true, level_0_offset_fn)
        } else {
            (false, offset_fn)
        };

        create_node_edges(
            hnsw_index,
            lazy_node,
            node,
            z,
            version,
            version_number,
            lazy_item_versions_table,
            if cur_level.0 == 0 {
                hnsw_params.level_0_neighbors_count
            } else {
                hnsw_params.neighbors_count
            },
            is_level_0,
            offset_fn,
            distance_metric,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn create_node(
    version_id: Hash,
    version_number: u16,
    hnsw_level: HNSWLevel,
    prop_value: Arc<NodePropValue>,
    prop_metadata: Option<Arc<NodePropMetadata>>,
    parent: SharedNode,
    child: SharedNode,
    neighbors_count: usize,
    is_level_0: bool,
    offset: u32,
    distance_metric: DistanceMetric,
) -> SharedNode {
    let node = ProbNode::new(
        hnsw_level,
        prop_value,
        prop_metadata,
        parent,
        child,
        neighbors_count,
        distance_metric,
    );
    ProbLazyItem::new(
        node,
        version_id,
        version_number,
        is_level_0,
        FileOffset(offset),
    )
}

#[allow(clippy::too_many_arguments)]
fn get_or_create_version(
    hnsw_index: &HNSWIndex,
    lazy_item_versions_table: Arc<TSHashTable<(InternalId, u16, u8), SharedNode>>,
    root_version_item: SharedNode,
    version_id: Hash,
    version_number: u16,
    is_level_0: bool,
    offset_fn: &mut dyn FnMut() -> u32,
    distance_metric: DistanceMetric,
) -> Result<(SharedNode, bool), WaCustomError> {
    let root_version_item_ref = unsafe { &*root_version_item };
    let root_node = root_version_item_ref.try_get_data(&hnsw_index.cache)?;

    lazy_item_versions_table.get_or_try_create_with_flag(
        (root_node.get_id(), version_number, root_node.hnsw_level.0),
        || {
            let (latest_version_item, mut guard) =
                ProbLazyItem::get_absolute_latest_version_write_access(
                    root_version_item,
                    &hnsw_index.cache,
                )?;
            let latest_version_item_ref = unsafe { &*latest_version_item };
            let latest_node = latest_version_item_ref.try_get_data(&hnsw_index.cache)?;
            if latest_version_item_ref.file_index.version_number == version_number {
                return Ok(latest_version_item);
            }

            let new_node = ProbNode::new_with_neighbors_and_versions_and_root_version(
                latest_node.hnsw_level,
                latest_node.prop_value.clone(),
                latest_node.prop_metadata.clone(),
                latest_node.clone_neighbors(),
                latest_node.get_parent(),
                latest_node.get_child(),
                root_version_item,
                true,
                distance_metric,
            );

            let new_node_offset = offset_fn();

            let version = ProbLazyItem::new(
                new_node,
                version_id,
                version_number,
                is_level_0,
                FileOffset(new_node_offset),
            );

            *guard = (version, false);
            drop(guard);

            let bufmans: &BufferManagerFactory<Hash> = if is_level_0 {
                &hnsw_index.cache.level_0_bufmans
            } else {
                &hnsw_index.cache.bufmans
            };

            let bufman = bufmans.get(root_version_item_ref.file_index.version_id)?;

            let cursor = bufman.open_cursor()?;
            let file_index = root_version_item_ref.file_index;
            let offset = file_index.offset.0;

            bufman.seek_with_cursor(cursor, offset as u64 + 37)?;
            bufman.update_u32_with_cursor(cursor, new_node_offset)?;
            bufman.update_u16_with_cursor(cursor, version_number)?;
            bufman.update_u32_with_cursor(cursor, *version_id)?;

            bufman.close_cursor(cursor)?;

            if latest_version_item_ref.file_index.version_number
                != root_version_item_ref.file_index.version_number
            {
                hnsw_index.cache.unload(latest_version_item)?;
            }

            Ok(version)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn create_node_edges(
    hnsw_index: &HNSWIndex,
    lazy_node: SharedNode,
    node: &ProbNode,
    neighbors: Vec<(SharedNode, MetricResult)>,
    version: Hash,
    version_number: u16,
    lazy_item_versions_table: Arc<TSHashTable<(InternalId, u16, u8), SharedNode>>,
    max_edges: usize,
    is_level_0: bool,
    offset_fn: &mut dyn FnMut() -> u32,
    distance_metric: DistanceMetric,
) -> Result<(), WaCustomError> {
    let mut successful_edges = 0;
    let mut neighbors_to_update = Vec::new();

    lazy_item_versions_table.insert(
        (node.get_id(), version_number, node.hnsw_level.0),
        lazy_node,
    );
    hnsw_index.cache.insert_lazy_object(lazy_node);

    // First loop: Handle neighbor connections and collect updates
    for (neighbor, dist) in neighbors {
        if successful_edges >= max_edges {
            break;
        }

        let (new_lazy_neighbor, found_in_map) = get_or_create_version(
            hnsw_index,
            lazy_item_versions_table.clone(),
            neighbor,
            version,
            version_number,
            is_level_0,
            offset_fn,
            distance_metric,
        )?;

        let new_neighbor = unsafe { &*new_lazy_neighbor }.try_get_data(&hnsw_index.cache)?;
        let neighbor_inserted_idx = node.add_neighbor(
            new_neighbor.get_id(),
            neighbor,
            dist,
            &hnsw_index.cache,
            distance_metric,
        );

        let neighbour_update_info = if let Some(neighbor_inserted_idx) = neighbor_inserted_idx {
            let node_inserted_idx = new_neighbor.add_neighbor(
                node.get_id(),
                lazy_node,
                dist,
                &hnsw_index.cache,
                distance_metric,
            );
            if let Some(idx) = node_inserted_idx {
                successful_edges += 1;
                Some((idx, dist))
            } else {
                node.remove_neighbor_by_index_and_id(neighbor_inserted_idx, new_neighbor.get_id());
                None
            }
        } else {
            None
        };

        if !found_in_map {
            write_node_to_file(
                new_lazy_neighbor,
                &hnsw_index.cache.bufmans,
                &hnsw_index.cache.level_0_bufmans,
                version,
            )?;
        } else if let Some((idx, dist)) = neighbour_update_info {
            neighbors_to_update.push((new_lazy_neighbor, idx, dist));
        }
    }

    // Second loop: Batch process file operations for updated neighbors
    if !neighbors_to_update.is_empty() {
        let bufman = if is_level_0 {
            hnsw_index.cache.level_0_bufmans.get(version)?
        } else {
            hnsw_index.cache.bufmans.get(version)?
        };
        let cursor = bufman.open_cursor()?;
        let mut current_node_link = Vec::with_capacity(14);
        current_node_link.extend(node.get_id().to_le_bytes());

        let node = unsafe { &*lazy_node };

        let FileIndex {
            offset: node_offset,
            version_number: node_version_number,
            version_id: node_version_id,
        } = node.file_index;
        current_node_link.extend(node_offset.0.to_le_bytes());
        current_node_link.extend(node_version_number.to_le_bytes());
        current_node_link.extend(node_version_id.to_le_bytes());

        for (neighbor, neighbor_idx, dist) in neighbors_to_update {
            let offset = unsafe { &*neighbor }.file_index.offset;
            let mut current_node_link_with_dist = Vec::with_capacity(19);
            current_node_link_with_dist.clone_from(&current_node_link);
            let (tag, value) = dist.get_tag_and_value();
            current_node_link_with_dist.push(tag);
            current_node_link_with_dist.extend(value.to_le_bytes());

            let neighbor_offset = (offset.0 + 49) + neighbor_idx as u32 * 19;
            bufman.seek_with_cursor(cursor, neighbor_offset as u64)?;
            bufman.update_with_cursor(cursor, &current_node_link_with_dist)?;
        }

        bufman.close_cursor(cursor)?;
    }

    write_node_to_file(
        lazy_node,
        &hnsw_index.cache.bufmans,
        &hnsw_index.cache.level_0_bufmans,
        version,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn traverse_find_nearest(
    config: &Config,
    hnsw_index: &HNSWIndex,
    start_node: SharedNode,
    fvec: &Storage,
    fvec_id: Option<&InternalId>,
    mdims: Option<&Metadata>,
    nodes_visited: &mut u32,
    skipm: &mut PerformantFixedSet,
    distance_metric: &DistanceMetric,
    is_indexing: bool,
    ef: u32,
) -> Result<Vec<(SharedNode, MetricResult)>, WaCustomError> {
    let mut candidate_queue = BinaryHeap::new();
    let mut results = Vec::new();

    let (start_version, guard) =
        ProbLazyItem::get_absolute_latest_version(start_node, &hnsw_index.cache)?;
    let start_data = unsafe { &*start_version }.try_get_data(&hnsw_index.cache)?;

    let fvec_data = VectorData {
        id: fvec_id,
        quantized_vec: fvec,
        metadata: mdims,
    };

    let start_metadata = start_data.prop_metadata.clone().map(|pm| pm.vec.clone());
    let start_vec_data = VectorData {
        id: Some(&start_data.prop_value.id),
        quantized_vec: &start_data.prop_value.vec,
        metadata: start_metadata.as_deref(),
    };
    let start_dist = distance_metric.calculate(&fvec_data, &start_vec_data, is_indexing)?;

    let start_id = *start_data.get_id();
    skipm.insert(start_id);
    candidate_queue.push((start_dist, start_node));
    drop(guard);

    while let Some((dist, current_node)) = candidate_queue.pop() {
        if *nodes_visited >= ef {
            break;
        }
        *nodes_visited += 1;

        let (current_version, _guard) =
            ProbLazyItem::get_absolute_latest_version(current_node, &hnsw_index.cache)?;
        results.push((dist, current_node));
        let node = unsafe { &*current_version }.try_get_data(&hnsw_index.cache)?;

        let _lock = node.freeze();
        for neighbor in node
            .get_neighbors_raw()
            .iter()
            .take(config.search.shortlist_size)
        {
            let (neighbor_id, neighbor_node) = unsafe {
                if let Some((id, node, _)) = neighbor.load(Ordering::Relaxed).as_ref() {
                    (*id, *node)
                } else {
                    continue;
                }
            };

            if !skipm.is_member(*neighbor_id) {
                let neighbor_data = unsafe { &*neighbor_node }.try_get_data(&hnsw_index.cache)?;
                let neighbor_metadata =
                    neighbor_data.prop_metadata.clone().map(|pm| pm.vec.clone());
                let neighbor_vec_data = VectorData {
                    id: Some(&neighbor_data.prop_value.id),
                    quantized_vec: &neighbor_data.prop_value.vec,
                    metadata: neighbor_metadata.as_deref(),
                };
                let dist =
                    distance_metric.calculate(&fvec_data, &neighbor_vec_data, is_indexing)?;
                skipm.insert(*neighbor_id);
                candidate_queue.push((dist, neighbor_node));
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
