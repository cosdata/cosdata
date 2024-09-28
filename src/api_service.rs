use crate::config_loader::Config;
use crate::models::buffered_io::*;
use crate::models::common::*;
use crate::models::file_persist::*;
use crate::models::lazy_load::*;
use crate::models::meta_persist::*;
use crate::models::rpc::VectorIdValue;
use crate::models::types::*;
use crate::models::user::Statistics;
use crate::models::versioning::VersionControl;
use crate::quantization::{Quantization, StorageType};
use crate::vector_store::*;
use arcshift::ArcShift;
use lmdb::{DatabaseFlags, Transaction};
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::array::TryFromSliceError;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Arc;

pub fn auto_config_storage_type(dim: usize) -> StorageType {
    if dim < 256 {
        StorageType::HalfPrecisionFP
    } else if dim < 512 {
        StorageType::UnsignedByte
    } else if dim < 1024 {
        StorageType::SubByte(3) // Octal
    } else if dim < 2048 {
        StorageType::SubByte(2) // Quaternary
    } else {
        StorageType::SubByte(1) // Binary
    }
}

pub async fn init_vector_store(
    name: String,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    num_layers: u8,
) -> Result<Arc<VectorStore>, WaCustomError> {
    if name.is_empty() {
        return Err(WaCustomError::InvalidParams);
    }

    let quantization_metric = QuantizationMetric::Scalar;
    let storage_type = auto_config_storage_type(size);
    let ain_env = get_app_env().map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let denv = ain_env.persist.clone();

    let metadata_db = denv
        .create_db(Some("metadata"), DatabaseFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let embeddings_db = denv
        .create_db(Some("embeddings"), DatabaseFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let vcs = Arc::new(
        VersionControl::new(denv.clone())
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
    );

    let lmdb = MetaDb {
        env: denv.clone(),
        metadata_db: Arc::new(metadata_db),
        embeddings_db: Arc::new(embeddings_db),
    };

    let hash = store_current_version(&lmdb, vcs.clone(), "main", 0)?;

    let min = lower_bound.unwrap_or(-1.0);
    let max = upper_bound.unwrap_or(1.0);
    let vec = (0..size)
        .map(|_| {
            let mut rng = rand::thread_rng();

            let random_number: f32 = rng.gen_range(min..max);
            random_number
        })
        .collect::<Vec<f32>>();
    let vec_hash = VectorId::Int(-1);

    let exec_queue_nodes: ExecQueueUpdate = STM::new(Vec::new(), 1, true);
    let vector_list = Arc::new(quantization_metric.quantize(&vec, storage_type)?);

    // Note that setting .write(true).append(true) has the same effect
    // as setting only .append(true)
    let prop_file = Arc::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("prop.data")
            .map_err(|e| WaCustomError::FsError(e.to_string()))?,
    );

    let mut root: LazyItemRef<MergedNode> = LazyItemRef::new_invalid();
    let mut prev: LazyItemRef<MergedNode> = LazyItemRef::new_invalid();

    let mut nodes = Vec::new();
    for l in (0..=num_layers).rev() {
        let prop = Arc::new(NodeProp {
            id: vec_hash.clone(),
            value: vector_list.clone(),
            location: Some((FileOffset(0), BytesToRead(0))),
        });
        let mut current_node = ArcShift::new(MergedNode {
            hnsw_level: HNSWLevel(l as u8),
            prop: ArcShift::new(PropState::Ready(prop.clone())),
            neighbors: EagerLazyItemSet::new(),
            parent: LazyItemRef::new_invalid(),
            child: LazyItemRef::new_invalid(),
        });

        let lazy_node = LazyItem::from_arcshift(hash, current_node.clone());
        let nn = LazyItemRef::from_arcshift(hash, current_node.clone());

        if let Some(prev_node) = prev.item.get().get_lazy_data() {
            current_node
                .get()
                .set_parent(prev.clone().item.get().clone());
            prev_node.set_child(lazy_node.clone());
        }
        prev = nn.clone();

        if l == 0 {
            root = nn.clone();
            let _prop_location = write_prop_to_file(&prop, &prop_file);
            current_node.get().set_prop_ready(prop);
        }
        nodes.push(nn.clone());
    }
    // TODO: include db name in the path
    let bufmans = Arc::new(BufferManagerFactory::new(
        Path::new(".").into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));
    for (l, nn) in nodes.iter_mut().enumerate() {
        match persist_node_update_loc(bufmans.clone(), &mut nn.item) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Failed node persist (init) for node {}: {}", l, e);
            }
        };
    }

    bufmans.flush_all()?;
    // ---------------------------
    // -- TODO level entry ratio
    // ---------------------------
    let factor_levels = 10.0;
    let lp = Arc::new(generate_tuples(factor_levels, num_layers));

    let vec_store = Arc::new(VectorStore::new(
        exec_queue_nodes,
        name.clone(),
        root,
        lp,
        size,
        prop_file,
        lmdb,
        ArcShift::new(hash),
        ArcShift::new(quantization_metric),
        ArcShift::new(DistanceMetric::Cosine),
        ArcShift::new(storage_type),
        vcs,
        num_layers,
    ));

    ain_env
        .vector_store_map
        .insert(name.clone(), vec_store.clone());

    Ok(vec_store)
}

pub fn run_upload(
    vec_store: Arc<VectorStore>,
    vecs: Vec<(VectorIdValue, Vec<f32>)>,
    config: Arc<Config>,
) -> Result<(), WaCustomError> {
    let current_version = vec_store.get_current_version();
    let next_version = vec_store
        .vcs
        .add_next_version("main")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    vec_store.set_current_version(next_version);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(format!("{}.vec_raw", *current_version))
        .map_err(|e| WaCustomError::FsError(e.to_string()))?;
    let bufman = Arc::new(BufferManager::new(file).map_err(BufIoError::Io)?);
    let cursor = bufman.open_cursor()?;
    bufman.write_u32_with_cursor(cursor, *next_version)?;
    bufman.close_cursor(cursor)?;

    vecs.into_par_iter()
        .map(|(id, vec)| {
            let hash_vec = convert_value(id);
            let vec_emb = RawVectorEmbedding {
                raw_vec: vec,
                hash_vec,
            };

            insert_embedding(bufman.clone(), vec_store.clone(), &vec_emb, current_version)
        })
        .collect::<Result<Vec<_>, _>>()?;
    bufman.flush()?;

    let env = vec_store.lmdb.env.clone();
    let metadata_db = vec_store.lmdb.metadata_db.clone();

    let txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let count_unindexed = txn
        .get(*metadata_db, &"count_unindexed")
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))
        .and_then(|bytes| {
            let bytes = bytes.try_into().map_err(|e: TryFromSliceError| {
                WaCustomError::DeserializationError(e.to_string())
            })?;
            Ok(u32::from_le_bytes(bytes))
        })?;

    txn.abort();

    if count_unindexed >= config.upload_threshold {
        index_embeddings(vec_store.clone(), config.upload_process_batch_size)?;
    }

    // TODO: include db name in the path
    let bufmans = Arc::new(BufferManagerFactory::new(
        Path::new(".").into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));

    auto_commit_transaction(vec_store.clone(), bufmans.clone())?;
    bufmans.flush_all()?;

    Ok(())
}

pub async fn ann_vector_query(
    vec_store: Arc<VectorStore>,
    query: Vec<f32>,
) -> Result<Option<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    let vector_store = vec_store.clone();
    let vec_hash = VectorId::Str("query".to_string());
    let root = &vector_store.root_vec;
    let vector_list = vector_store
        .quantization_metric
        .quantize(&query, *vector_store.storage_type.clone().get())?;

    let vec_emb = QuantizedVectorEmbedding {
        quantized_vec: Arc::new(vector_list.clone()),
        hash_vec: vec_hash.clone(),
    };

    let results = ann_search(
        vec_store.clone(),
        vec_emb,
        root.item.clone().get().clone(),
        vec_store
            .hnsw_params
            .clone()
            .get()
            .num_layers
            .try_into()
            .unwrap(),
    )?;
    let output = remove_duplicates_and_filter(results);
    Ok(output)
}

pub async fn fetch_vector_neighbors(
    vec_store: Arc<VectorStore>,
    vector_id: VectorId,
) -> Vec<Option<(VectorId, Vec<(VectorId, MetricResult)>)>> {
    let results = vector_fetch(vec_store.clone(), vector_id);
    return results.expect("Failed fetching vector neighbors");
}

fn calculate_statistics(_: &[i32]) -> Option<Statistics> {
    // Placeholder for calculating statistics
    None
}

fn vector_knn(_vs: &Vec<f32>, _vecs: &Vec<f32>) -> Vec<(i8, i8, String, f64)> {
    // Placeholder for vector KNN
    vec![]
}
