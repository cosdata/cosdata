use actix_web::{web, HttpResponse};

use crate::models::collection_cache::CollectionCacheExt;

use crate::{
    api_service::{ann_vector_query, batch_ann_vector_query},
    app_context::AppContext,
    models::{
        common::WaCustomError,
        rpc::{BatchVectorANN, RPCResponseBody, VectorANN},
    },
};

// Route: `/vectordb/search`
pub(crate) async fn search(
    web::Json(body): web::Json<VectorANN>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
    let _ = ctx.update_collection_for_query(&body.vector_db_name);

    // Try to get the vector store from the environment
    let hnsw_index = match ctx
        .ain_env
        .collections_map
        .get_hnsw_index(&body.vector_db_name)
    {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    };

    let metadata_filter = match body.filter {
        Some(filter) => match filter.to_internal() {
            Ok(mf) => Some(mf),
            Err(WaCustomError::MetadataError(e)) => {
                return HttpResponse::BadRequest().body(format!("Invalid metadata filter: {e}"))
            }
            Err(_) => return HttpResponse::InternalServerError().body("Metadata filter error"),
        },
        None => None,
    };

    let query_result = ann_vector_query(
        ctx.into_inner(),
        hnsw_index.clone(),
        body.vector,
        metadata_filter,
        body.nn_count,
    )
    .await;

    let result = match query_result {
        Ok(result) => result,
        Err(err) => match err {
            WaCustomError::InvalidParams => {
                return HttpResponse::BadRequest().body("Bad request".to_string())
            }
            _ => return HttpResponse::InternalServerError().body(err.to_string()),
        },
    };

    let response_data = RPCResponseBody::RespVectorKNN {
        knn: result.into_iter().map(|(id, dist)| (id.0, dist)).collect(),
    };
    HttpResponse::Ok().json(response_data)
}

// Route: `/vectordb/batch-search`
pub(crate) async fn batch_search(
    web::Json(body): web::Json<BatchVectorANN>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
    let _ = ctx.update_collection_for_query(&body.vector_db_name);

    // Try to get the vector store from the environment
    let hnsw_index = match ctx
        .ain_env
        .collections_map
        .get_hnsw_index(&body.vector_db_name)
    {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    };

    let metadata_filter = match body.filter {
        Some(filter) => match filter.to_internal() {
            Ok(mf) => Some(mf),
            Err(WaCustomError::MetadataError(e)) => {
                return HttpResponse::BadRequest().body(format!("Invalid metadata filter: {e}"))
            }
            Err(_) => return HttpResponse::InternalServerError().body("Metadata filter error"),
        },
        None => None,
    };

    let query_results = batch_ann_vector_query(
        ctx.into_inner(),
        hnsw_index.clone(),
        body.vectors,
        metadata_filter,
        body.nn_count,
    )
    .await;

    let results = match query_results {
        Ok(results) => results,
        Err(err) => match err {
            WaCustomError::InvalidParams => {
                return HttpResponse::BadRequest().body("Bad request".to_string())
            }
            _ => return HttpResponse::InternalServerError().body(err.to_string()),
        },
    };

    let response_data: Vec<_> = results
        .into_iter()
        .map(|result| RPCResponseBody::RespVectorKNN {
            knn: result.into_iter().map(|(id, dist)| (id.0, dist)).collect(),
        })
        .collect();
    HttpResponse::Ok().json(response_data)
}
