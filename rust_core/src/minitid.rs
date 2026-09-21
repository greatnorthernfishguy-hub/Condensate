// ---- Changelog ----
// [2026-06-22] Claude (Sonnet 4.6) — Initial build
// What: CC-native KISS proxy for the Anthropic API.
// Why:  KISS runs inside the NG sidecar (gateway-side, neurograph_rpc.py).
//       CC has no gateway — it calls api.anthropic.com directly. This binary
//       is that gateway, stripped to what CC actually needs: message
//       compression + transparent upstream streaming.
// How:  axum 0.8 HTTP server → KISS message compressor (Rust-native, no
//       Python round-trip) → reqwest upstream → SSE stream passthrough.
//       CC sets ANTHROPIC_BASE_URL=http://127.0.0.1:$MINITID_PORT.
//
// [2026-09-21] Claude (Sonnet 4.6) — Correction pass
// What: Removed the faux-KISS compression path and updated comments.
// Why:  The header still advertised 60-char first-sentence truncation,
//       which is LAW 3 shrapnel: a removed behaviour lying about the
//       current contract.
// How:  b263286 deleted the rewrite; this pass records that in the
//       changelog and replaces the obsolete KISS behaviour header with
//       an honest passthrough description.
// -------------------
//
// KISS behaviour (current contract):
//   This June snapshot no longer truncates message content.
//   /v1/messages forwards the original request bytes unchanged.
//   KissSession and KISS_* constants are retained as dead-code
//   scaffolding for a future pass; they do not affect forwarding.
//
// Session identity: SHA-256 of the first 100 bytes of the first user
// message's content string, truncated to 16 hex chars. Stable across
// turns because the first message never changes.
//
// env vars:
//   MINITID_PORT      — listen port (default: 9090)
//   MINITID_UPSTREAM  — upstream base URL (default: https://api.anthropic.com)
//
// Build:
//   cargo build --release --features minitid
//
// Wire up CC:
//   export ANTHROPIC_BASE_URL=http://127.0.0.1:9090
//   # Add to ~/.bashrc alongside LD_PRELOAD for membrane.

use axum::{Router, extract::State, response::Response};
use axum::body::Body;
use reqwest::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

// ── KISS constants (env-overridable in a future pass) ──────────────────────
const DEFAULT_PORT: u16 = 9090;
const KISS_RECENT_WINDOW: usize = 10;
const KISS_WARMUP_TURNS: u32  = 3;
const KISS_FORCE_FULL_EVERY: u32 = 20;

// Maximum request body buffered before forwarding (20 MB covers any realistic
// CC conversation; Anthropic will reject oversized bodies before we do).
const MAX_BODY: usize = 20 * 1_024 * 1_024;

// Hop-by-hop and body-invalidating headers stripped from the forwarded request.
// content-length is excluded because reqwest recomputes it when we set the body.
const DROP_REQ_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
];

// ── Per-session KISS state ──────────────────────────────────────────────────

struct KissSession {
    turn_count: u32,
    since_full: u32,
}

// ── Shared app state ────────────────────────────────────────────────────────

struct AppState {
    // Fallback upstream when the config file is absent/unreadable.
    upstream_fallback: String,
    client:            Client,
    sessions:          Mutex<HashMap<String, KissSession>>,
}

// Read the live upstream URL from ~/.config/minitid/upstream, falling back to
// `fallback` when the file is absent.  Called per-request so `use-claude` /
// `use-openrouter` can switch destinations without restarting the service.
fn read_upstream(fallback: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let path = format!("{}/.config/minitid/upstream", home);
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Extract plain text from a message content Value, whether it is a bare
/// string or an Anthropic block array. For arrays, returns the first text
/// block's text. Used for session identity — NOT for forwarding.
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks.iter()
            .find_map(|b| {
                if b["type"].as_str() == Some("text") {
                    b["text"].as_str()
                } else {
                    None
                }
            })
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

/// Stable session key: SHA-256 of the first 100 bytes of the first user
/// message's text, rendered as 16 hex chars. Claude Code sends content as a
/// block array, so we extract the first text block rather than assuming a
/// string (else every session hashes to "" and shares KISS state).
fn session_id(messages: &[Value]) -> String {
    let first = messages.iter()
        .find(|m| m["role"].as_str() == Some("user"))
        .map(|m| content_text(&m["content"]))
        .unwrap_or_default();
    let bytes = first.as_bytes();
    let mut h = Sha256::new();
    h.update(&bytes[..bytes.len().min(100)]);
    let hex = format!("{:x}", h.finalize());
    hex[..16].to_string()
}

// ── Proxy handler ────────────────────────────────────────────────────────────

async fn proxy(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> Result<Response<Body>, (axum::http::StatusCode, String)> {
    use axum::http::StatusCode;

    let method  = req.method().clone();
    let uri     = req.uri().clone();
    let headers = req.headers().clone();

    let is_messages = method == axum::http::Method::POST
        && uri.path() == "/v1/messages";

    // Buffer the request body for forwarding.
    let body_bytes = axum::body::to_bytes(req.into_body(), MAX_BODY)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // This June snapshot no longer rewrites /v1/messages. Honest passthrough
    // forwards the original bytes unchanged.

    // Build upstream URL — read config file live so switching providers
    // takes effect immediately without restarting the service.
    let upstream = read_upstream(&state.upstream_fallback);
    let path_and_query = uri.path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(uri.path());
    let upstream_url = format!("{}{}", upstream, path_and_query);

    // Build reqwest request, forwarding safe headers.
    let req_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let mut rb = state.client.request(req_method, &upstream_url);
    for (name, value) in &headers {
        if !DROP_REQ_HEADERS.contains(&name.as_str()) {
            rb = rb.header(name.as_str(), value.as_bytes());
        }
    }
    rb = rb.body(body_bytes.to_vec());

    let upstream_resp = rb.send().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    // Map status and headers, stream body back.
    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let resp_headers = upstream_resp.headers().clone();

    let mut response = Response::new(Body::from_stream(upstream_resp.bytes_stream()));
    *response.status_mut() = status;
    for (k, v) in &resp_headers {
        response.headers_mut().insert(k, v.clone());
    }
    Ok(response)
}

// ── Entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let port: u16 = env::var("MINITID_PORT")
        .ok().and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let upstream = env::var("MINITID_UPSTREAM")
        .unwrap_or_else(|_| "https://api.anthropic.com".to_string());

    let state = Arc::new(AppState {
        client:            Client::builder().build().expect("reqwest client"),
        sessions:          Mutex::new(HashMap::new()),
        upstream_fallback: upstream.clone(),
    });

    let app = Router::new()
        .fallback(proxy)
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    eprintln!("miniTID  {}  →  {}", addr, upstream);

    let listener = TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

// ── Tests ────────────────────────────────────────────────────────────────────
// Run: cargo test --features minitid --bin minitid
//
// Faux KISS compression has been removed from this demo snapshot. The only
// remaining test covers session identity extraction from array-form content.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_id_handles_array_content() {
        let a = vec![json!({"role": "user", "content": [{"type": "text", "text": "hello world alpha"}]})];
        let b = vec![json!({"role": "user", "content": [{"type": "text", "text": "different beta text"}]})];
        let id_a = session_id(&a);
        let id_b = session_id(&b);
        assert_eq!(id_a.len(), 16);
        assert_ne!(id_a, id_b, "distinct first messages must yield distinct session ids");
        assert_eq!(id_a, session_id(&a), "session id must be stable");
    }
}
