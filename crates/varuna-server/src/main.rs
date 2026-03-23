//! VARUNA-AUV Naval Operations Center – main entry point.
use std::net::SocketAddr;
use axum::{Router, routing::get};
use tower_http::services::ServeDir;
use tower_http::cors::{CorsLayer, Any};
use tracing_subscriber::EnvFilter;

mod routes;
mod websocket;
mod state;

use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("varuna_server=info".parse()?))
        .init();

    let state = AppState::new().await?;
    let static_dir = std::env::current_dir()?.join("crates/varuna-server/static");

    let app = Router::new()
        .route("/api/status",    get(routes::status_handler))
        .route("/api/spectrum",  get(routes::spectrum_handler))
        .route("/api/lofar",     get(routes::lofar_handler))
        .route("/api/demon",     get(routes::demon_handler))
        .route("/api/mfcc",      get(routes::mfcc_handler))
        .route("/api/classify",  get(routes::classify_handler))
        .route("/api/tracks",    get(routes::tracks_handler))
        .route("/api/beamform",  get(routes::beamform_handler))
        .route("/ws",            get(websocket::ws_handler))
        .nest_service("/", ServeDir::new(&static_dir).append_index_html_on_directories(true))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("VARUNA Naval Ops Center listening on http://{}", addr);

    // Auto-open browser
    let url = format!("http://{}", addr);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let _ = webbrowser::open(&url);
    });

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
