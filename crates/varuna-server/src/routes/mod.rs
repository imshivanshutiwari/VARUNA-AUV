//! HTTP route handlers.
use axum::{extract::State, Json};
use serde_json::{json, Value};
use crate::state::AppState;

/// GET /api/status
pub async fn status_handler() -> Json<Value> {
    Json(json!({
        "status": "operational",
        "system": "VARUNA-AUV Naval Operations Center",
        "version": "0.1.0",
        "timestamp": chrono_now()
    }))
}

/// GET /api/spectrum
pub async fn spectrum_handler(State(s): State<AppState>) -> Json<Value> {
    let frame = s.sonar_frame.read().await;
    Json(json!({ "spectrum_db": frame.spectrum_db }))
}

/// GET /api/lofar
pub async fn lofar_handler(State(s): State<AppState>) -> Json<Value> {
    let frame = s.sonar_frame.read().await;
    Json(json!({ "lofar_slice": frame.lofar_slice }))
}

/// GET /api/demon
pub async fn demon_handler(State(s): State<AppState>) -> Json<Value> {
    let frame = s.sonar_frame.read().await;
    Json(json!({
        "power_db": frame.demon_power_db,
        "blade_rate_hz": frame.demon_blade_rate_hz
    }))
}

/// GET /api/mfcc
pub async fn mfcc_handler(State(s): State<AppState>) -> Json<Value> {
    let frame = s.sonar_frame.read().await;
    Json(json!({ "mfcc_frame": frame.mfcc_frame }))
}

/// GET /api/classify
pub async fn classify_handler(State(s): State<AppState>) -> Json<Value> {
    let frame = s.sonar_frame.read().await;
    Json(json!({ "classification": frame.classification }))
}

/// GET /api/tracks
pub async fn tracks_handler(State(s): State<AppState>) -> Json<Value> {
    let frame = s.sonar_frame.read().await;
    Json(json!({ "tracks": frame.tracks }))
}

/// GET /api/beamform
pub async fn beamform_handler(State(s): State<AppState>) -> Json<Value> {
    let frame = s.sonar_frame.read().await;
    Json(json!({ "doa_deg": frame.doa_deg }))
}

fn chrono_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
