//! `GET /v1/models` — OpenAI-compatible model list derived from config.

use axum::extract::State;
use axum::Json;

use crate::types::{Model, ModelsList};
use crate::AppState;

pub async fn list(State(state): State<AppState>) -> Json<ModelsList> {
    let data = state
        .config
        .models
        .iter()
        .map(|m| Model {
            id: m.id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "concerto".to_string(),
        })
        .collect();

    Json(ModelsList {
        object: "list".to_string(),
        data,
    })
}
