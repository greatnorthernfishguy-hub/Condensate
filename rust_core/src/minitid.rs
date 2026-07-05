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
//
// ---- Changelog ----
// [2026-07-04/05] Claude Code (Sonnet 5 / Haiku 4.5) — CC Gateway Turn-Deposit (Tasks 5-7)
// What: Tee the response stream (relay live to the client + accumulate for parsing),
//       reconstruct the assistant's full text from SSE content_block_delta/text_delta
//       events, extract the genuine last user message before KISS mutates the body,
//       and deposit both sides of the turn as raw ExperienceEntry frames via the
//       ng_tract crate (write_experience/deposit_to_file) to CC_GATEWAY_TRACT_PATH.
// Why:  Claude Code's hook system never exposes CC's own generated response text to
//       any hook script -- only prompts and tool I/O. CC's own NeuroGraph instance
//       (a separate substrate from Syl's) never saw its own words. This closes that
//       gap the same way Anima closes it for Syl: sit in the transport path and
//       deposit raw, unclassified turn text (LAW 7), never a direct call into the
//       Python daemon that drains it (LAW 1) -- deposit/drain via tract file only.
// How:  Client-facing relay and accumulation/deposit run in two independently
//       spawned tokio tasks so neither blocking nor a panic in the deposit path can
//       reach the proxied response. CC_GATEWAY_TRACT_PATH (LAW 5) is read
//       independently here and by the Python drain side (cc_ng_organism.py), same
//       default on both. See docs/superpowers/specs/2026-07-04-cc-gateway-turn-deposit-design.md.
// -------------------

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

/// The genuine last user message, extracted from the request body BEFORE
/// KISS's compression mutates it. KISS only compresses OLDER history, so
/// the last message is always the real one -- this must be captured
/// independent of and before apply_kiss() runs on the messages array.
fn extract_last_user_message(body_bytes: &[u8]) -> Option<String> {
    let body: Value = serde_json::from_slice(body_bytes).ok()?;
    let messages = body["messages"].as_array()?;
    for msg in messages.iter().rev() {
        if msg["role"].as_str() != Some("user") {
            continue;
        }
        if let Some(s) = msg["content"].as_str() {
            return Some(s.to_string());
        }
        if let Some(blocks) = msg["content"].as_array() {
            for b in blocks {
                if b["type"].as_str() == Some("text") {
                    if let Some(t) = b["text"].as_str() {
                        return Some(t.to_string());
                    }
                }
            }
        }
    }
    None
}

/// CC_GATEWAY_TRACT_PATH (LAW 5) -- read independently here and by the
/// Python daemon's drain_ingest_tract(); same default on both sides so
/// they resolve to the same file even if the env var is never set.
fn cc_gateway_tract_path() -> String {
    std::env::var("CC_GATEWAY_TRACT_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        format!("{home}/.claude/plugins/neurograph/tracts/cc_gateway/turns.tract")
    })
}

fn deposit_experience_entry(path: &str, source: &str, content: String) {
    if content.trim().is_empty() {
        return;
    }
    let entry = ng_tract::ExperienceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
        source: source.to_string(),
        content_type: "text".to_string(),
        content: content.into_bytes(),
    };
    let bytes = ng_tract::write::write_experience(&entry);
    let _ = ng_tract::write::deposit_to_file(path, &bytes);
}

/// Deposit both sides of one turn as raw experience (LAW 7) -- no
/// classification, no embedding computed here (deferred to the daemon's
/// ng_embed boundary), no success/failure label (write_experience, never
/// write_outcome). Two independent entries, not one combined/paired
/// record -- the daemon's own dual-pass chains consecutive conversational
/// deposits via a delayed synapse regardless of physical turn boundaries.
/// Fails soft: this must never affect the proxied response to the client.
fn deposit_turn(user_message: Option<String>, assistant_text: String) {
    let path = cc_gateway_tract_path();
    if let Some(dir) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Some(user_text) = user_message {
        deposit_experience_entry(&path, "cc_gateway", user_text);
    }
    deposit_experience_entry(&path, "cc_gateway", assistant_text);
}

/// Reconstruct the assistant's full generated text from accumulated SSE
/// bytes by concatenating every `content_block_delta` event whose
/// `delta.type` is `text_delta`. Malformed/non-text events are skipped,
/// never panicked on -- this runs on every response and must never crash
/// the proxy. Returns "" if nothing could be reconstructed.
fn reconstruct_assistant_text(sse_bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(sse_bytes);
    let mut out = String::new();
    for line in text.lines() {
        let Some(json_str) = line.strip_prefix("data: ") else { continue };
        let Ok(value) = serde_json::from_str::<Value>(json_str) else { continue };
        if value["type"].as_str() != Some("content_block_delta") {
            continue;
        }
        if value["delta"]["type"].as_str() != Some("text_delta") {
            continue;
        }
        if let Some(t) = value["delta"]["text"].as_str() {
            out.push_str(t);
        }
    }
    out
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

    // Extract the last user message BEFORE KISS mutation — the last message is
    // always genuine (uncompressed) since KISS only compresses older history.
    let last_user_message = if is_messages {
        extract_last_user_message(&body_bytes)
    } else {
        None
    };

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

    use futures_util::StreamExt;
    use tokio::sync::mpsc;
    let (tx, rx) = mpsc::unbounded_channel::<Result<axum::body::Bytes, std::io::Error>>();
    let mut upstream_stream = upstream_resp.bytes_stream();
    let accumulator_task = tokio::spawn(async move {
        let mut accumulated: Vec<u8> = Vec::new();
        let mut client_connected = true;
        while let Some(chunk) = upstream_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    accumulated.extend_from_slice(&bytes);
                    if client_connected && tx.send(Ok(bytes)).is_err() {
                        client_connected = false; // client disconnected -- stop relaying, still finish accumulating for deposit
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())));
                    break;
                }
            }
        }
        accumulated
    });

    tokio::spawn(async move {
        let accumulated = match accumulator_task.await {
            Ok(bytes) => bytes,
            Err(_) => return, // task panicked -- nothing to deposit, never crash the proxy
        };
        let assistant_text = reconstruct_assistant_text(&accumulated);
        deposit_turn(last_user_message, assistant_text);
    });

    let out_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(out_stream));
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

    #[tokio::test]
    async fn test_reconstruct_assistant_text_from_sse() {
        let sse = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
                    event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\", world\"}}\n\n\
                    event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let text = reconstruct_assistant_text(sse);
        assert_eq!(text, "Hello, world");
    }

    #[tokio::test]
    async fn test_reconstruct_assistant_text_ignores_non_text_delta() {
        let sse = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n";
        let text = reconstruct_assistant_text(sse);
        assert_eq!(text, "");
    }

    #[tokio::test]
    async fn test_reconstruct_assistant_text_handles_malformed_lines_gracefully() {
        let sse = b"garbage\nnot json at all\ndata: {not even valid json\n\n";
        let text = reconstruct_assistant_text(sse);
        assert_eq!(text, "");
    }

    #[test]
    fn test_extract_last_user_message_returns_most_recent_user_text() {
        let body = br#"{"messages":[
            {"role":"user","content":"first"},
            {"role":"assistant","content":"reply"},
            {"role":"user","content":"second, the real last message"}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("second, the real last message".to_string()));
    }

    #[test]
    fn test_extract_last_user_message_handles_array_content_blocks() {
        let body = br#"{"messages":[
            {"role":"user","content":[{"type":"text","text":"array-form message"}]}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("array-form message".to_string()));
    }

    #[test]
    fn test_extract_last_user_message_returns_none_on_malformed_body() {
        let body = b"not json";
        assert_eq!(extract_last_user_message(body), None);
    }

    #[test]
    fn test_deposit_turn_writes_two_experience_entries() {
        use ng_tract::read::{TractReader, ReadResult};
        use ng_tract::TractEntry;

        let tmp = std::env::temp_dir().join(format!("minitid_test_tract_{}.tract", std::process::id()));
        std::env::set_var("CC_GATEWAY_TRACT_PATH", tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);

        deposit_turn(Some("what is the numpy issue".to_string()), "it's a stray .pth file".to_string());

        let data = std::fs::read(&tmp).expect("tract file should exist");
        let mut reader = TractReader::new(&data);
        let mut count = 0;
        let mut contents = Vec::new();
        while let Some(result) = reader.next_entry() {
            if let Ok(ReadResult::Entry(TractEntry::Experience(exp))) = result {
                count += 1;
                contents.push(String::from_utf8_lossy(&exp.content).to_string());
            }
        }
        assert_eq!(count, 2);
        assert!(contents.iter().any(|c| c.contains("numpy issue")));
        assert!(contents.iter().any(|c| c.contains("stray .pth file")));
        std::fs::remove_file(&tmp).ok();
    }
}
