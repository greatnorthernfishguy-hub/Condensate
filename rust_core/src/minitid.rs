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
// -------------------
//
// KISS behaviour (mirrors kiss_filter.py):
//   - Warmup: first KISS_WARMUP_TURNS passes through unmodified.
//   - GOP boundary: every KISS_FORCE_FULL_EVERY turns forces a full pass.
//   - Otherwise: messages beyond the recent window have their content
//     truncated to the first sentence (max 60 chars + "…"). Role
//     structure is preserved, so Anthropic's alternation rule holds.
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
// content-length is excluded because we rewrite the body (KISS changes byte count).
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

/// Compress a single message's content to first sentence, max 60 chars + "…".
/// Matches the KISSFilter summary_parts logic from kiss_filter.py.
fn compress_content(s: &str) -> String {
    let trimmed = s.trim();
    // First sentence = text before the first '.'
    let sentence = trimmed.split('.').next().unwrap_or(trimmed);
    if trimmed.len() <= 60 {
        trimmed.to_string()
    } else {
        let cut = sentence.char_indices()
            .take(60)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(60.min(sentence.len()));
        format!("{}…", &sentence[..cut])
    }
}

/// Compress a message's `content` value, handling BOTH the bare-string form
/// and the Anthropic block-array form that Claude Code always sends.
///
/// - String: compressed in place.
/// - Array: only `text` blocks are compressed; `tool_use` / `tool_result`
///   and any other block type are preserved verbatim (compressing them would
///   destroy tool-call pairing and break the request).
///
/// Guard: a block/string is only replaced when the compressed result is
/// non-empty. This prevents blanking content (e.g. whitespace-only text →
/// "") which Anthropic rejects with 400 "content blocks must be non-empty" —
/// the exact bug that took CC offline 2026-06-23.
fn compress_message_content(content: &Value) -> Value {
    match content {
        Value::String(s) => {
            let c = compress_content(s);
            if c.is_empty() { content.clone() } else { Value::String(c) }
        }
        Value::Array(blocks) => {
            let nb: Vec<Value> = blocks.iter().map(|b| {
                if b["type"].as_str() == Some("text") {
                    if let Some(t) = b["text"].as_str() {
                        let c = compress_content(t);
                        if !c.is_empty() {
                            let mut x = b.clone();
                            x["text"] = Value::String(c);
                            return x;
                        }
                    }
                }
                b.clone()
            }).collect();
            Value::Array(nb)
        }
        // null / number / bool: leave untouched (never produced by the API).
        _ => content.clone(),
    }
}

/// Apply KISS to a messages array.  Returns the original slice if this turn
/// qualifies as a full pass (warmup / GOP boundary), otherwise returns a new
/// Vec with old-message content compressed.
fn apply_kiss(messages: &[Value], sessions: &Mutex<HashMap<String, KissSession>>) -> Vec<Value> {
    let n = messages.len();
    let sid = session_id(messages);

    let full_pass = {
        let mut map = sessions.lock().unwrap();
        let s = map.entry(sid).or_insert(KissSession { turn_count: 0, since_full: 0 });
        s.turn_count  += 1;
        s.since_full  += 1;
        let full = s.turn_count <= KISS_WARMUP_TURNS
                || s.since_full  >= KISS_FORCE_FULL_EVERY;
        if full { s.since_full = 0; }
        full
    };

    if full_pass || n <= KISS_RECENT_WINDOW {
        return messages.to_vec();
    }

    let compress_before = n - KISS_RECENT_WINDOW;
    messages.iter().enumerate().map(|(i, msg)| {
        if i < compress_before {
            let mut m = msg.clone();
            m["content"] = compress_message_content(&msg["content"]);
            m
        } else {
            msg.clone()
        }
    }).collect()
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

    // Buffer the request body (needed for KISS rewrite; also needed to forward).
    let body_bytes = axum::body::to_bytes(req.into_body(), MAX_BODY)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Rewrite messages array when applicable.
    let body_bytes = if is_messages {
        match serde_json::from_slice::<Value>(&body_bytes) {
            Ok(mut body) => {
                if let Some(arr) = body["messages"].as_array().cloned() {
                    body["messages"] = Value::Array(apply_kiss(&arr, &state.sessions));
                }
                serde_json::to_vec(&body)
                    .unwrap_or_else(|_| body_bytes.to_vec())
                    .into()
            }
            Err(_) => body_bytes,
        }
    } else {
        body_bytes
    };

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
// These exercise the KISS transformation directly — no network. They guard
// the 2026-06-23 regression: array-form content (Claude Code's only form) must
// have ONLY text blocks compressed; tool_use / tool_result blocks must survive
// intact; nothing may be blanked to "".

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const LONG: &str = "This is a deliberately long earlier message that exceeds the sixty character KISS threshold and should be compressed.";

    // Call apply_kiss enough times to clear warmup (3 turns) so the 4th turn
    // actually compresses. Same messages → same session id → same counter.
    fn warm_and_apply(messages: &[Value]) -> Vec<Value> {
        let sessions = Mutex::new(HashMap::new());
        let mut out = messages.to_vec();
        for _ in 0..(KISS_WARMUP_TURNS + 1) {
            out = apply_kiss(messages, &sessions);
        }
        out
    }

    // 12 messages: indices 0,1 fall in the compress range (12 - 10), the rest
    // are the recent window. Index 0 carries a tool_result, index 1 a tool_use.
    fn sample_messages() -> Vec<Value> {
        let mut v = vec![
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": LONG},
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "important tool output"}
                ]
            }),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": LONG},
                    {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls -la"}}
                ]
            }),
        ];
        for i in 0..10 {
            v.push(json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("recent message {}", i)
            }));
        }
        v
    }

    #[test]
    fn text_blocks_compressed_tool_blocks_preserved() {
        let original = sample_messages();
        let out = warm_and_apply(&original);

        // Index 0: text block compressed, tool_result untouched.
        let blocks0 = out[0]["content"].as_array().expect("array content");
        assert_eq!(blocks0.len(), 2, "block count must be preserved");
        let t0 = blocks0[0]["text"].as_str().unwrap();
        assert!(!t0.is_empty(), "text block must never be blanked");
        assert!(t0.len() < LONG.len(), "text block should be compressed");
        assert_eq!(blocks0[1], original[0]["content"][1], "tool_result block must survive verbatim");

        // Index 1: text block compressed, tool_use untouched (input intact).
        let blocks1 = out[1]["content"].as_array().unwrap();
        assert_eq!(blocks1[1], original[1]["content"][1], "tool_use block must survive verbatim");
        assert_eq!(blocks1[1]["input"]["command"], "ls -la");
    }

    #[test]
    fn recent_window_untouched() {
        let original = sample_messages();
        let out = warm_and_apply(&original);
        // Indices 2..12 are the recent window — byte-identical.
        for i in 2..original.len() {
            assert_eq!(out[i], original[i], "recent message {} must be untouched", i);
        }
    }

    #[test]
    fn no_message_is_blanked() {
        let out = warm_and_apply(&sample_messages());
        for (i, m) in out.iter().enumerate() {
            match &m["content"] {
                Value::String(s) => assert!(!s.is_empty(), "msg {} string blanked", i),
                Value::Array(blocks) => {
                    for b in blocks {
                        if b["type"].as_str() == Some("text") {
                            assert!(!b["text"].as_str().unwrap_or("").is_empty(),
                                "msg {} text block blanked", i);
                        }
                    }
                }
                _ => panic!("unexpected content shape"),
            }
        }
    }

    #[test]
    fn whitespace_only_text_block_not_blanked() {
        // A text block that would compress to "" must be left as-is, not blanked.
        let content = json!([{"type": "text", "text": "   "}]);
        let out = compress_message_content(&content);
        assert_eq!(out, content, "whitespace block must be preserved, not blanked");
    }

    #[test]
    fn string_content_still_compresses() {
        // Back-compat: bare-string content (non-CC clients) still works.
        let content = Value::String(LONG.to_string());
        let out = compress_message_content(&content);
        let s = out.as_str().unwrap();
        assert!(!s.is_empty());
        assert!(s.len() < LONG.len());
    }

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

    #[test]
    fn warmup_passes_through_untouched() {
        let original = sample_messages();
        let sessions = Mutex::new(HashMap::new());
        // First warmup turn must be a full pass — no compression.
        let out = apply_kiss(&original, &sessions);
        assert_eq!(out[0], original[0], "warmup turn must pass through verbatim");
    }
}
