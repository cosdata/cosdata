use crate::app_context::AppContext;
use crate::indexes::hnsw::offset_counter::{HNSWIndexFileOffsetCounter, IndexFileId};
use crate::indexes::hnsw::types::HNSWHyperParams;
use crate::indexes::hnsw::{DenseInputEmbedding, HNSWIndex};
use crate::indexes::inverted::InvertedIndex;
use crate::indexes::tf_idf::TFIDFIndex;
use crate::indexes::IndexOps;
use crate::metadata::{pseudo_level_probs, pseudo_node_vector, pseudo_root_id};
use crate::models::buffered_io::{BufferManagerFactory, FilelessBufferManager};
use crate::models::cache_loader::HNSWIndexCache;
use crate::models::collection::Collection;
use crate::models::collection_transaction::BackgroundExplicitTransaction;
use crate::models::common::*;
use crate::models::meta_persist::store_values_range;
use crate::models::prob_node::ProbNode;
use crate::models::types::*;
use crate::models::versioning::VersionNumber;
use crate::quantization::StorageType;
use crate::vector_store::*;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// creates a dense index for a collection
#[allow(clippy::too_many_arguments)]
pub async fn init_hnsw_index_for_collection(
    ctx: Arc<AppContext>,
    collection: Arc<Collection>,
    values_range: Option<(f32, f32)>,
    hnsw_params: HNSWHyperParams,
    quantization_metric: QuantizationMetric,
    distance_metric: DistanceMetric,
    storage_type: StorageType,
    sample_threshold: usize,
    is_configured: bool,
) -> Result<Arc<HNSWIndex>, WaCustomError> {
    let collection_name = &collection.meta.name;
    let collection_path: Arc<Path> = collection.get_path();
    let index_path = collection_path.join("dense_hnsw");
    // ensuring that the index has a separate directory created inside the collection directory
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let env = ctx.ain_env.persist.clone();

    let lmdb = MetaDb::from_env(env.clone(), collection_name)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    //
    // what is the prop file exactly?
    // a file that stores the quantized version of raw vec
    let prop_file = RwLock::new(
        fs::OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(index_path.join("prop.data"))
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    );

    let index_manager = BufferManagerFactory::new(
        index_path.clone().into(),
        |root, ver: &IndexFileId| root.join(format!("{}.index", **ver)),
        ProbNode::get_serialized_size(hnsw_params.neighbors_count) * 1000,
    );
    let latest_version_links_bufman = FilelessBufferManager::new(8192)?;

    let distance_metric = Arc::new(RwLock::new(distance_metric));

    let cache = HNSWIndexCache::new(
        index_manager,
        latest_version_links_bufman,
        index_path.clone(),
        ctx.config.enable_context_history,
        prop_file,
        distance_metric.clone(),
    );
    if let Some(values_range) = values_range {
        store_values_range(&lmdb, values_range).map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to store values range to LMDB: {}", e))
        })?;
    }
    let values_range = values_range.unwrap_or((-1.0, 1.0));
    let offset_counter = HNSWIndexFileOffsetCounter::new(
        ctx.config.index_file_min_size,
        hnsw_params.level_0_neighbors_count,
        hnsw_params.neighbors_count,
    );

    let root = create_root_node(
        &quantization_metric,
        storage_type,
        collection.meta.dense_vector.dimension,
        &cache.prop_file,
        *collection.current_version.read(),
        &offset_counter,
        &cache,
        values_range,
        &hnsw_params,
        *distance_metric.read().unwrap(),
        collection.meta.metadata_schema.as_ref(),
    )?;

    cache.flush_all(VersionNumber::from(0))?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 4.0;

    // If metadata schema is supported, the level_probs needs to be
    // adjusted to accommodate only pseudo nodes in the higher layers
    let lp = match &collection.meta.metadata_schema {
        Some(metadata_schema) => {
            // @TODO(vineet): Unnecessary computation of
            // pseudo_weighted_dimensions. Just the no. of pseudo
            // replicas should be sufficient.
            let replica_dims = metadata_schema.pseudo_weighted_dimensions(1);
            let plp = pseudo_level_probs(hnsw_params.num_layers, replica_dims.len() as u16);
            // @TODO(vineet): Super hacky
            let num_lower_layers = plp.iter().filter(|(p, _)| *p == 0.0).count() - 1;
            let num_higher_layers = hnsw_params.num_layers - (num_lower_layers as u8);
            let mut lp = vec![];
            for i in 0..num_higher_layers {
                // no actual replica nodes in higher layers
                lp.push((1.0, hnsw_params.num_layers - i))
            }
            let mut lower_lp = generate_level_probs(factor_levels, num_lower_layers as u8);
            lp.append(&mut lower_lp);
            lp
        }
        None => generate_level_probs(factor_levels, hnsw_params.num_layers),
    };

    // If the collection has metadata_schema defined, we create pseudo
    // nodes. But first, we create the pseudo root node separately as
    // it's an independent root node much like the main root
    // node. Once pseudo root is created, it can be passed when
    // instantiating the HNSWIndex. And the rest of the non-root
    // pseudo nodes can be created through the index's methods
    let pseudo_root = match &collection.meta.metadata_schema {
        Some(metadata_schema) => {
            let num_dims = collection.meta.dense_vector.dimension;
            let pseudo_vals = pseudo_node_vector(num_dims);
            let pseudo_root_id = pseudo_root_id();
            let node = create_pseudo_root_node(
                &quantization_metric,
                storage_type,
                &cache.prop_file,
                *collection.current_version.read(),
                &offset_counter,
                &cache,
                values_range,
                &hnsw_params,
                *distance_metric.read().unwrap(),
                metadata_schema,
                pseudo_vals.clone(),
                pseudo_root_id,
            )?;
            Some(node)
        }
        None => None,
    };

    let hnsw_index = Arc::new(HNSWIndex::new(
        root,
        pseudo_root,
        lp,
        collection.meta.dense_vector.dimension,
        quantization_metric,
        distance_metric,
        storage_type,
        hnsw_params,
        cache,
        values_range,
        sample_threshold,
        is_configured,
        collection
            .meta
            .metadata_schema
            .as_ref()
            .map_or(1, |schema| schema.max_num_replicas()),
        offset_counter,
    ));

    ctx.ain_env
        .collections_map
        .insert_hnsw_index(&collection, hnsw_index.clone())?;

    // If the collection has metadata schema, we create pseudo replica
    // nodes to ensure that the query vectors with metadata dimensions
    // are reachable from the root node.
    if collection.meta.metadata_schema.is_some() {
        let num_dims = collection.meta.dense_vector.dimension;
        let pseudo_vals = pseudo_node_vector(num_dims);
        // base id for nonroot pseudo nodes is 1 more than the pseudo node
        let pseudo_vec = DenseInputEmbedding(pseudo_root_id(), pseudo_vals, None, true);
        let version_number = *collection.current_version.read();
        let transaction =
            BackgroundExplicitTransaction::from_version_id_and_number(&collection, version_number);
        // NOTE: We're directly calling `index_embeddings` instead of
        // `run_upload` because we want to skip sampling for pseudo
        // nodes
        hnsw_index.index_embeddings(
            &collection,
            vec![pseudo_vec],
            transaction.version,
            &ctx.config,
        )?;
        transaction.pre_commit(&collection, &ctx.config)?;
    }

    Ok(hnsw_index)
}

/// creates an inverted index for a collection
pub async fn init_inverted_index_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
    quantization_bits: u8,
    sample_threshold: usize,
) -> Result<Arc<InvertedIndex>, WaCustomError> {
    let collection_path: Arc<Path> = collection.get_path();
    let index_path = collection_path.join("sparse_inverted_index");
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let index = Arc::new(InvertedIndex::new(
        index_path.clone(),
        quantization_bits,
        sample_threshold,
    )?);

    ctx.ain_env
        .collections_map
        .insert_inverted_index(collection, index.clone())?;
    Ok(index)
}

/// creates an inverted index for a collection
pub async fn init_tf_idf_index_for_collection(
    ctx: Arc<AppContext>,
    collection: &Collection,
    sample_threshold: usize,
    k1: f32,
    b: f32,
) -> Result<Arc<TFIDFIndex>, WaCustomError> {
    let collection_path: Arc<Path> = collection.get_path();
    let index_path = collection_path.join("tf_idf_index");
    fs::create_dir_all(&index_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

    let index = Arc::new(TFIDFIndex::new(
        index_path.clone(),
        sample_threshold,
        k1,
        b,
    )?);

    ctx.ain_env
        .collections_map
        .insert_tf_idf_index(collection, index.clone())?;
    Ok(index)
}
