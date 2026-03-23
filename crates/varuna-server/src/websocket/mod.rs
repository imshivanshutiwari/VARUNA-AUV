//! WebSocket handler for real-time sonar data streaming.
use crate::state::AppState;
use axum::extract::ws::{Message, WebSocket};
use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};

/// GET /ws – upgrades to a WebSocket connection.
pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let sonar_frame = state.sonar_frame.clone();

    // Send sonar frames at 20 Hz
    let send_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(50));
        loop {
            interval.tick().await;
            let frame = sonar_frame.read().await.clone();
            match serde_json::to_string(&frame) {
                Ok(json) => {
                    if sender.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to serialize sonar frame: {}", e);
                }
            }
        }
    });

    // Drain incoming messages (keep connection alive)
    while let Some(Ok(_)) = receiver.next().await {}
    send_task.abort();
}
