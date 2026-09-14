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
// [2026-07-16 DudeMan CC] — miniTID substrate peninsula (P1/P2), gated OFF
// What: apply_pith_peninsula() + daemon_compress_history()/message_text()/
//       set_compressed_text(). When MINITID_PITH_PENINSULA=1, older-than-window
//       turns are compressed by the CC daemon's substrate-informed Pith (via
//       daemon.sock compress_history) instead of the 60-char faux-KISS cut.
// Why:  CC Body Architecture Primitive B — move compression from the blind
//       proxy into the substrate-connected peninsula; the real replacement for
//       faux KISS (retires task #49). Law-enforcer PASS (intra-module IPC).
// How:  blocking UnixStream on spawn_blocking + tokio timeout (async runtime
//       never stalls); JSON newline frames; splice preserves tool_use/
//       tool_result; fail-soft → inline faux compression on any daemon failure.
//       Default OFF (unchanged apply_kiss); socket via MINITID_PENINSULA_SOCK.
// [2026-09-12] Codex (GPT-5.6 Sol) — Preserve the live human instruction
// What: Exclude the latest genuine user message, as a complete Value, from
//       both faux-KISS and Pith history compression; use the same genuine-user
//       predicate for turn deposit and compression protection.
// Why:  Five tool-use/result pairs can push the current instruction outside
//       the ten-message numeric window during the same turn. Pith then reduced
//       the live task to a keyframe and the working CC lost its assignment.
// How:  Build a sparse eligible-index mask, send only eligible history to
//       Pith, splice results through that mask, and reuse it for fail-soft faux
//       compression. The previous human turn becomes eligible when a newer
//       genuine human instruction arrives.
// [2026-09-13] Codex (GPT-5.6 Sol) — Bind cadence to genuine human turns
// What: Key KISS/Pith state from Claude's request metadata, partition agent
//       sidechains, and reuse one gate decision throughout each human turn.
// Why:  Tool-loop requests were consuming warmup/GOP cadence and the first-
//       prompt hash could merge sessions or change after native compaction.
// How:  Parse metadata.user_id without logging it, track the latest genuine
//       human marker/count, bypass native compaction requests, and fail open
//       whenever stable request identity is unavailable.
// [2026-09-13] Codex (GPT-5.6 Sol) — Harden cadence identity and state bounds
// What: Detect Claude's native compaction prompt without relying on a header,
//       derive turn identity from genuine human blocks only, and cap session
//       cadence state with deterministic least-recently-used eviction.
// Why:  The localhost miniTID route does not retain Claude's native compaction
//       header, injected hook blocks changed the turn hash, and abandoned
//       sessions otherwise accumulated for the lifetime of the daemon.
// How:  Bypass on the latest user-role compaction prompt before state access,
//       share one filtered human-text extractor with deposits and cadence, and
//       keep an env-bounded access-order store that never evicts the active key.
// [2026-09-13] Codex (GPT-5.6 Sol) — Compose each request from the live CC mind
// What: Replace peninsula history keyframes with one fresh provider_context
//       assembled from CC's live NeuroGraph, the exact current instruction and
//       Quest orientation, and the intact current tool episode.
// Why:  Replaying or periodically restoring transcript history is not Pith.
//       Provider context must come from the substrate's current learned
//       topology and activation on every request, including warmup/GOP turns.
// How:  Extract the already-rendered Quest text from Claude's request (never
//       Quest storage), cue the existing daemon socket verb, evict obsolete
//       history only after a valid fresh response, and fail open unchanged.
// -------------------
//
// KISS behaviour (mirrors kiss_filter.py):
//   - Warmup: first KISS_WARMUP_TURNS passes through unmodified.
//   - GOP boundary: every KISS_FORCE_FULL_EVERY turns forces a full pass.
//   - Otherwise: messages beyond the recent window have their content
//     truncated to the first sentence (max 60 chars + "…"). Role
//     structure is preserved, so Anthropic's alternation rule holds.
//
// Session identity: SHA-256 of metadata.user_id.session_id, optionally
// partitioned by a bounded x-claude-code-agent-id. Missing or malformed
// identity fails open without touching shared cadence state.
//
// env vars:
//   MINITID_PORT      — listen port (default: 9090)
//   MINITID_UPSTREAM  — upstream base URL (default: https://api.anthropic.com)
//   MINITID_SESSION_CAPACITY — retained cadence sessions (default: 4096)
//   MINITID_PITH_TOOL_TAIL_BYTES — exact recent tool-tail target (default: 65536)
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

use axum::body::Body;
use axum::{extract::State, http::HeaderMap, response::Response, Router};
use reqwest::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
// Blocking Unix-socket client to the CC daemon's read-only provider_context
// handler, run on the blocking pool so the async runtime is never stalled.
// Gated off by default (MINITID_PITH_PENINSULA).
use std::env;
use std::io::{Read as _PenRead, Write as _PenWrite};
use std::net::SocketAddr;
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

// ── KISS constants (env-overridable in a future pass) ──────────────────────
const DEFAULT_PORT: u16 = 9090;
const KISS_RECENT_WINDOW: usize = 10;
const KISS_WARMUP_TURNS: u32 = 3;
const KISS_FORCE_FULL_EVERY: u32 = 20;
const DEFAULT_SESSION_CAPACITY: usize = 4_096;
const MIN_SESSION_CAPACITY: usize = 16;
const MAX_SESSION_CAPACITY: usize = 65_536;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GateDecision {
    FullPass,
    Compress,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KissSession {
    turn_count: u32,
    since_full: u32,
    human_marker: [u8; 32],
    visible_human_count: usize,
    decision: GateDecision,
    last_access: u64,
}

struct SessionStore {
    entries: HashMap<String, KissSession>,
    access_clock: u64,
    capacity: usize,
}

impl SessionStore {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_clock: 0,
            capacity: capacity.max(1),
        }
    }

    fn next_access(&mut self) -> u64 {
        self.access_clock = self.access_clock.saturating_add(1);
        self.access_clock
    }

    fn evict_lru_for_new_session(&mut self) {
        if self.entries.len() < self.capacity {
            return;
        }
        let victim = self
            .entries
            .iter()
            .min_by(|(key_a, a), (key_b, b)| {
                a.last_access
                    .cmp(&b.last_access)
                    .then_with(|| key_a.cmp(key_b))
            })
            .map(|(key, _)| key.clone());
        if let Some(key) = victim {
            self.entries.remove(&key);
        }
    }
}

// ── Shared app state ────────────────────────────────────────────────────────

struct AppState {
    // Fallback upstream when the config file is absent/unreadable.
    upstream_fallback: String,
    client: Client,
    sessions: Mutex<SessionStore>,
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

fn session_capacity() -> usize {
    configured_session_capacity(env::var("MINITID_SESSION_CAPACITY").ok().as_deref())
}

fn configured_session_capacity(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .map(|capacity| capacity.clamp(MIN_SESSION_CAPACITY, MAX_SESSION_CAPACITY))
        .unwrap_or(DEFAULT_SESSION_CAPACITY)
}

fn pith_tool_tail_bytes() -> usize {
    configured_pith_tool_tail_bytes(env::var("MINITID_PITH_TOOL_TAIL_BYTES").ok().as_deref())
}

fn configured_pith_tool_tail_bytes(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .map(|bytes| bytes.clamp(MIN_PITH_TOOL_TAIL_BYTES, MAX_PITH_TOOL_TAIL_BYTES))
        .unwrap_or(DEFAULT_PITH_TOOL_TAIL_BYTES)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const MAX_SESSION_ID_BYTES: usize = 512;
const MAX_AGENT_ID_BYTES: usize = 128;
const NEUROGRAPH_SURFACED_MARKER: &str = "[NeuroGraph Surfaced Knowledge]";
const QUEST_TRACKER_BANNER: &str =
    "ACTIVE QUEST TRAIL for this session (injected by the Quest Tracker).";
const MAX_PROVIDER_CONTEXT_CHARS: usize = 40_000;
const MAX_PROVIDER_RESPONSE_BYTES: usize = 256 * 1024;
const DEFAULT_PITH_TOOL_TAIL_BYTES: usize = 64 * 1024;
const MIN_PITH_TOOL_TAIL_BYTES: usize = 8 * 1024;
const MAX_PITH_TOOL_TAIL_BYTES: usize = 1024 * 1024;
const CLAUDE_COMPACTION_PROMPT_PREFIX: &str = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.\n- You already have all the context you need in the conversation above.\n- Tool calls will be REJECTED and will waste your only turn";
const CLAUDE_COMPACTED_PREAMBLE: &str = "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\n";

fn digest_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Stable, opaque cadence key from Claude Code request identity. The full
/// user_id envelope is neither retained nor logged.
fn session_key(body: &Value, headers: &HeaderMap) -> Option<String> {
    let encoded = body.get("metadata")?.get("user_id")?.as_str()?;
    let metadata: Value = serde_json::from_str(encoded).ok()?;
    let session_id = metadata.get("session_id")?.as_str()?;
    if session_id.is_empty() || session_id.len() > MAX_SESSION_ID_BYTES {
        return None;
    }

    let mut key = digest_hex(session_id.as_bytes());
    if let Some(value) = headers.get("x-claude-code-agent-id") {
        let agent_id = value.to_str().ok()?;
        if agent_id.is_empty() || agent_id.len() > MAX_AGENT_ID_BYTES {
            return None;
        }
        key.push(':');
        key.push_str(&digest_hex(agent_id.as_bytes()));
    }
    Some(key)
}

fn has_exact_prefix(text: &str, prefix: &str) -> bool {
    text.as_bytes().get(..prefix.len()) == Some(prefix.as_bytes())
}

fn is_claude_compaction_request_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    has_exact_prefix(trimmed, CLAUDE_COMPACTION_PROMPT_PREFIX)
}

fn is_claude_compacted_summary_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    has_exact_prefix(trimmed, CLAUDE_COMPACTED_PREAMBLE)
}

fn message_has_text_matching(msg: &Value, predicate: impl Fn(&str) -> bool) -> bool {
    match &msg["content"] {
        Value::String(text) => predicate(text),
        Value::Array(blocks) => blocks.iter().any(|block| {
            block["type"].as_str() == Some("text")
                && block["text"].as_str().map(&predicate).unwrap_or(false)
        }),
        _ => false,
    }
}

fn is_compact_summary_message(msg: &Value) -> bool {
    msg["isCompactSummary"].as_bool() == Some(true)
        || message_has_text_matching(msg, is_claude_compacted_summary_text)
}

fn latest_user_role_is_compaction_request(messages: &[Value]) -> bool {
    messages
        .iter()
        .rfind(|msg| msg["role"].as_str() == Some("user"))
        .map(|msg| message_has_text_matching(msg, is_claude_compaction_request_text))
        .unwrap_or(false)
}

fn human_turn_marker(messages: &[Value]) -> Option<([u8; 32], usize)> {
    let mut count = 0;
    let mut latest = None;
    for msg in messages {
        if let Some(text) = genuine_user_text(msg) {
            count += 1;
            latest = Some(text);
        }
    }
    let latest = latest?;
    let mut digest = Sha256::new();
    digest.update(latest.as_bytes());
    Some((digest.finalize().into(), count))
}

fn decision_for_request(
    session_key: String,
    messages: &[Value],
    sessions: &Mutex<SessionStore>,
) -> Option<GateDecision> {
    let (human_marker, visible_human_count) = human_turn_marker(messages)?;
    let mut store = sessions.lock().ok()?;
    let access = store.next_access();
    let Some(session) = store.entries.get_mut(&session_key) else {
        let decision = GateDecision::FullPass;
        store.evict_lru_for_new_session();
        store.entries.insert(
            session_key,
            KissSession {
                turn_count: 1,
                since_full: 0,
                human_marker,
                visible_human_count,
                decision,
                last_access: access,
            },
        );
        return Some(decision);
    };
    session.last_access = access;

    let advances = human_marker != session.human_marker
        || (visible_human_count > session.visible_human_count
            && human_marker == session.human_marker);
    session.visible_human_count = visible_human_count;
    if !advances {
        return Some(session.decision);
    }

    session.human_marker = human_marker;
    session.turn_count += 1;
    session.since_full += 1;
    let full =
        session.turn_count <= KISS_WARMUP_TURNS || session.since_full >= KISS_FORCE_FULL_EVERY;
    if full {
        session.since_full = 0;
    }
    session.decision = if full {
        GateDecision::FullPass
    } else {
        GateDecision::Compress
    };
    Some(session.decision)
}

fn gate_decision_for_body(
    body: &Value,
    headers: &HeaderMap,
    sessions: &Mutex<SessionStore>,
) -> Option<GateDecision> {
    if headers.contains_key("x-cc-compaction-request") {
        return None;
    }
    let messages = body.get("messages")?.as_array()?;
    if latest_user_role_is_compaction_request(messages) {
        return None;
    }
    let key = session_key(body, headers)?;
    decision_for_request(key, messages, sessions)
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
        let cut = sentence
            .char_indices()
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
            if c.is_empty() {
                content.clone()
            } else {
                Value::String(c)
            }
        }
        Value::Array(blocks) => {
            let nb: Vec<Value> = blocks
                .iter()
                .map(|b| {
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
                })
                .collect();
            Value::Array(nb)
        }
        // null / number / bool: leave untouched (never produced by the API).
        _ => content.clone(),
    }
}

/// Indices before `compress_before` that may be compressed. The latest
/// genuine human message is the live instruction for the current top-level
/// turn; tool-use/result traffic can push it outside the numeric recent window
/// without making it history.
fn compression_indices(messages: &[Value], compress_before: usize) -> Vec<usize> {
    let protected = last_genuine_user_message_index(messages);
    (0..compress_before)
        .filter(|i| Some(*i) != protected && !is_compact_summary_message(&messages[*i]))
        .collect()
}

/// Apply the in-process faux compressor to exactly the eligible old messages.
/// Both the normal faux path and the Pith failure fallback use this helper so
/// daemon availability cannot change current-turn continuity.
fn apply_faux_at_indices(messages: &[Value], indices: &[usize]) -> Vec<Value> {
    let mut out = messages.to_vec();
    for &i in indices {
        out[i]["content"] = compress_message_content(&messages[i]["content"]);
    }
    out
}

/// Apply KISS to a messages array.  Returns the original slice if this turn
/// qualifies as a full pass (warmup / GOP boundary), otherwise returns a new
/// Vec with old-message content compressed.
fn apply_kiss(messages: &[Value], decision: GateDecision) -> Vec<Value> {
    let n = messages.len();
    if decision == GateDecision::FullPass || n <= KISS_RECENT_WINDOW {
        return messages.to_vec();
    }

    let compress_before = n - KISS_RECENT_WINDOW;
    let indices = compression_indices(messages, compress_before);
    apply_faux_at_indices(messages, &indices)
}

// ============================================================================
// Substrate peninsula — fresh provider context from CC's living NeuroGraph.
// Gated off by default (MINITID_PITH_PENINSULA unset).  Obsolete history is
// evicted only after the daemon returns a valid fresh context.  Any failure
// leaves the original request untouched.
// ============================================================================

/// Call the CC daemon's read-only `provider_context` handler.  The request
/// carries only attention cues already present in this request; it never sends
/// transcript history.  Accept only the producer's closed fresh states.
async fn daemon_provider_context(
    current_instruction: String,
    quest_focus: String,
) -> Option<String> {
    let sock = std::env::var("MINITID_PENINSULA_SOCK").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/josh".into());
        format!("{}/.claude/plugins/neurograph/daemon.sock", home)
    });
    let fut = tokio::task::spawn_blocking(move || -> Option<String> {
        let mut stream = UnixStream::connect(&sock).ok()?;
        let t = std::time::Duration::from_millis(1500);
        stream.set_read_timeout(Some(t)).ok()?;
        stream.set_write_timeout(Some(t)).ok()?;
        let req = provider_context_request(&current_instruction, &quest_focus);
        let mut line = serde_json::to_vec(&req).ok()?;
        line.push(b'\n');
        stream.write_all(&line).ok()?;
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let nread = stream.read(&mut chunk).ok()?;
            if nread == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..nread]);
            if buf.len() > MAX_PROVIDER_RESPONSE_BYTES {
                return None;
            }
            if buf.last() == Some(&b'\n') {
                break;
            }
        }
        if buf.last() != Some(&b'\n') {
            return None;
        }
        parse_provider_response(&buf)
    });
    match tokio::time::timeout(std::time::Duration::from_millis(2000), fut).await {
        Ok(Ok(v)) => v,
        _ => None,
    }
}

fn provider_context_request(current_instruction: &str, quest_focus: &str) -> Value {
    serde_json::json!({
        "event": "provider_context",
        "data": {
            "current_instruction": current_instruction,
            "quest_focus": quest_focus,
        }
    })
}

fn parse_provider_response(bytes: &[u8]) -> Option<String> {
    let resp: Value = serde_json::from_slice(bytes).ok()?;
    if resp.get("ok")?.as_bool()? != true {
        return None;
    }
    let state = resp.get("state")?.as_str()?;
    if resp.get("source")?.as_str()? != "cc_neurograph_topology" {
        return None;
    }
    resp.get("coherence")?.as_str()?;
    let anchors = resp.get("anchors")?.as_array()?;
    let warnings = resp.get("warnings")?.as_array()?;
    if !anchors.iter().all(Value::is_string) || !warnings.iter().all(Value::is_string) {
        return None;
    }
    let assemblies = resp.get("assemblies")?.as_u64()?;
    if !matches!((state, assemblies), ("ok", 1..) | ("empty", 0)) {
        return None;
    }
    let context = resp.get("context")?.as_str()?;
    let context_chars = context.chars().count();
    if context.trim().is_empty() || context_chars > MAX_PROVIDER_CONTEXT_CHARS {
        return None;
    }
    Some(context.to_string())
}

fn content_texts(content: &Value) -> Vec<&str> {
    match content {
        Value::String(text) => vec![text.as_str()],
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| block["type"].as_str() == Some("text"))
            .filter_map(|block| block["text"].as_str())
            .collect(),
        _ => Vec::new(),
    }
}

/// Pull the latest bounded Quest orientation out of the rendered Claude
/// request.  SessionStart wraps it in a system-reminder; Quest's banner is the
/// stable owned boundary.  The extracted bytes begin at that banner and stop
/// before Claude's wrapper close, so unrelated hook context is not promoted.
fn verified_quest_text<'a>(msg: &Value, text: &'a str) -> Result<Option<&'a str>, ()> {
    if msg["role"].as_str() != Some("user") {
        return Ok(None);
    }
    let trimmed = text.trim_start();
    let verified_envelope =
        trimmed.starts_with("<system-reminder>\nSessionStart hook additional context: ");
    if !verified_envelope && msg["isMeta"].as_bool() != Some(true) {
        return Ok(None);
    }
    let mut starts = text
        .match_indices(QUEST_TRACKER_BANNER)
        .map(|(index, _)| index);
    let Some(start) = starts.next() else {
        return Ok(None);
    };
    if starts.next().is_some() {
        return Err(());
    }
    let remainder = &text[start..];
    let end = remainder
        .find("\n</system-reminder>")
        .unwrap_or(remainder.len());
    Ok(Some(&remainder[..end]))
}

fn extract_quest_focus(messages: &[Value]) -> Result<Option<String>, ()> {
    for msg in messages.iter().rev() {
        for text in content_texts(&msg["content"]).into_iter().rev() {
            if let Some(quest) = verified_quest_text(msg, text)? {
                return Ok(Some(quest.to_string()));
            }
        }
    }
    Ok(None)
}

fn verified_quest_rails(messages: &[Value]) -> Result<Vec<String>, ()> {
    let mut rails = Vec::new();
    for msg in messages {
        for text in content_texts(&msg["content"]) {
            if let Some(quest) = verified_quest_text(msg, text)? {
                rails.push(quest.to_string());
            }
        }
    }
    Ok(rails)
}

fn is_standalone_neurograph_surface_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    if trimmed.starts_with(NEUROGRAPH_SURFACED_MARKER) {
        return true;
    }
    let Some(reminder) = trimmed.strip_prefix("<system-reminder>\n") else {
        return false;
    };
    let reminder = reminder.trim_start();
    let boundary = " hook additional context: ";
    let Some(split) = reminder.find(boundary) else {
        return false;
    };
    let event = &reminder[..split];
    if event.is_empty() || event.len() > 32 || !event.chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    reminder[split + boundary.len()..].starts_with(NEUROGRAPH_SURFACED_MARKER)
}

/// Remove only independently identifiable synthetic NG text blocks.  Every
/// neighboring human/tool/reminder block is cloned byte-for-byte.  If the
/// marker is embedded inside inseparable text, decline the rewrite.
fn strip_neurograph_surface(msg: &Value) -> Result<Option<Value>, ()> {
    if msg["role"].as_str() != Some("user") {
        return Ok(Some(msg.clone()));
    }
    match &msg["content"] {
        Value::String(text) => {
            if is_standalone_neurograph_surface_text(text) {
                Ok(None)
            } else if text.contains(NEUROGRAPH_SURFACED_MARKER) {
                Err(())
            } else {
                Ok(Some(msg.clone()))
            }
        }
        Value::Array(blocks) => {
            let mut kept = Vec::with_capacity(blocks.len());
            for block in blocks {
                if block["type"].as_str() == Some("text") {
                    if let Some(text) = block["text"].as_str() {
                        if is_standalone_neurograph_surface_text(text) {
                            continue;
                        }
                        if text.contains(NEUROGRAPH_SURFACED_MARKER) {
                            return Err(());
                        }
                    }
                }
                kept.push(block.clone());
            }
            if kept.is_empty() {
                Ok(None)
            } else if kept.len() == blocks.len() {
                Ok(Some(msg.clone()))
            } else {
                let mut cleaned = msg.clone();
                cleaned["content"] = Value::Array(kept);
                Ok(Some(cleaned))
            }
        }
        _ => Ok(Some(msg.clone())),
    }
}

fn count_marker(messages: &[Value], marker: &str) -> usize {
    messages
        .iter()
        .flat_map(|msg| content_texts(&msg["content"]))
        .map(|text| text.matches(marker).count())
        .sum()
}

fn tool_ids(msg: &Value, block_type: &str, id_field: &str) -> Vec<String> {
    match &msg["content"] {
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| block["type"].as_str() == Some(block_type))
            .filter_map(|block| block[id_field].as_str())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn message_has_tool_traffic(msg: &Value) -> bool {
    match &msg["content"] {
        Value::Array(blocks) => blocks.iter().any(|block| {
            matches!(
                block["type"].as_str(),
                Some("tool_use") | Some("tool_result")
            )
        }),
        _ => false,
    }
}

fn suffix_has_complete_tool_pairs(episode: &[Value], start: usize) -> bool {
    for (offset, msg) in episode[start..].iter().enumerate() {
        let result_pos = start + offset;
        for result_id in tool_ids(msg, "tool_result", "tool_use_id") {
            if !episode[start..result_pos].iter().any(|candidate| {
                tool_ids(candidate, "tool_use", "id")
                    .iter()
                    .any(|use_id| use_id == &result_id)
            }) {
                return false;
            }
        }
    }
    true
}

fn serialized_messages_bytes(messages: &[Value]) -> usize {
    messages
        .iter()
        .map(|msg| {
            serde_json::to_vec(msg)
                .map(|bytes| bytes.len())
                .unwrap_or(MAX_BODY)
        })
        .sum()
}

/// Keep the exact current human turn plus a small, pair-aware recent suffix.
/// Resolved old pairs fall away together.  A retained result pulls its matching
/// use into the suffix, while every still-unresolved use remains visible.
fn bounded_current_episode(cleaned: &[Value], tool_tail_bytes: usize) -> Option<Vec<Value>> {
    let current = cleaned.first()?.clone();
    let episode = &cleaned[1..];
    if episode.is_empty() {
        return Some(vec![current]);
    }
    let mut start = episode.len().saturating_sub(KISS_RECENT_WINDOW);

    // An unresolved tool call is unfinished business even if it is older than
    // the ordinary recent suffix.
    let mut unresolved_positions = Vec::new();
    for (use_pos, msg) in episode.iter().enumerate() {
        for use_id in tool_ids(msg, "tool_use", "id") {
            let resolved = episode[use_pos + 1..].iter().any(|later| {
                tool_ids(later, "tool_result", "tool_use_id")
                    .iter()
                    .any(|result_id| result_id == &use_id)
            });
            if !resolved {
                unresolved_positions.push(use_pos);
                start = start.min(use_pos);
            }
        }
    }

    // Never retain a result without the assistant message that issued it.
    loop {
        let mut adjusted = start;
        for (offset, msg) in episode[start..].iter().enumerate() {
            let result_pos = start + offset;
            for result_id in tool_ids(msg, "tool_result", "tool_use_id") {
                let use_pos = episode[..result_pos].iter().rposition(|candidate| {
                    tool_ids(candidate, "tool_use", "id")
                        .iter()
                        .any(|use_id| use_id == &result_id)
                })?;
                adjusted = adjusted.min(use_pos);
            }
        }
        if adjusted == start {
            break;
        }
        start = adjusted;
    }

    // A configured byte envelope prevents ten enormous tool messages from becoming
    // another context window.  Admission stays whole-message and pair-aware.
    // The newest indivisible pair (or unresolved call) is the semantic floor:
    // it remains exact even if that one unit alone exceeds this target.
    if serialized_messages_bytes(&episode[start..]) > tool_tail_bytes {
        let safe_candidates: Vec<usize> = (start + 1..episode.len())
            .filter(|candidate| {
                !unresolved_positions.iter().any(|pos| pos < candidate)
                    && suffix_has_complete_tool_pairs(episode, *candidate)
            })
            .collect();
        if let Some(candidate) = safe_candidates
            .iter()
            .copied()
            .find(|candidate| serialized_messages_bytes(&episode[*candidate..]) <= tool_tail_bytes)
            .or_else(|| safe_candidates.last().copied())
        {
            start = candidate;
        }
    }

    let mut out = Vec::with_capacity(1 + episode.len() - start);
    out.push(current);
    out.extend_from_slice(&episode[start..]);
    Some(out)
}

/// Compose a bounded L1 only from the fresh topology response and the live
/// request tail. Retained messages stay exact unless an independently
/// identified NG text block is removed; every neighboring block stays exact.
fn compose_provider_messages(
    messages: &[Value],
    provider_context: &str,
    quest_focus: Option<&str>,
) -> Option<Vec<Value>> {
    if provider_context.trim().is_empty()
        || provider_context.chars().count() > MAX_PROVIDER_CONTEXT_CHARS
        || provider_context.contains(NEUROGRAPH_SURFACED_MARKER)
        || provider_context.contains(QUEST_TRACKER_BANNER)
    {
        return None;
    }
    let current = last_genuine_user_message_index(messages)?;
    let mut cleaned_messages = Vec::new();
    for msg in &messages[current..] {
        if msg["isMeta"].as_bool() == Some(true) || is_compact_summary_message(msg) {
            if message_has_tool_traffic(msg) {
                return None;
            }
            continue;
        }
        if let Some(cleaned) = strip_neurograph_surface(msg).ok()? {
            cleaned_messages.push(cleaned);
        }
    }
    let tail = bounded_current_episode(&cleaned_messages, pith_tool_tail_bytes())?;

    let quest_rails_in_tail = verified_quest_rails(&tail).ok()?;
    if quest_rails_in_tail.len() > 1 {
        return None;
    }
    let mut out = vec![serde_json::json!({
        "role": "user",
        "content": format!("{NEUROGRAPH_SURFACED_MARKER}\n\n{provider_context}"),
    })];
    if quest_rails_in_tail.is_empty() {
        if let Some(quest) = quest_focus.filter(|text| !text.is_empty()) {
            out.push(serde_json::json!({
                "role": "user",
                "content": format!(
                    "<system-reminder>\nSessionStart hook additional context: {quest}\n</system-reminder>"
                ),
            }));
        }
    }
    out.extend(tail);
    if count_marker(&out, NEUROGRAPH_SURFACED_MARKER) != 1 {
        return None;
    }
    let expected_quest = usize::from(quest_focus.map(|text| !text.is_empty()).unwrap_or(false));
    let verified_out = verified_quest_rails(&out).ok()?;
    if verified_out.len() != expected_quest {
        return None;
    }
    if let Some(expected) = quest_focus.filter(|text| !text.is_empty()) {
        if verified_out.first().map(String::as_str) != Some(expected) {
            return None;
        }
    }
    Some(out)
}

/// Build fresh context on every Pith-enabled provider request.  Warmup and GOP
/// cadence never restore raw history.  The old request survives byte-for-byte
/// whenever the daemon or composition boundary is unavailable.
async fn apply_pith_peninsula(messages: &[Value], decision: GateDecision) -> Vec<Value> {
    let Some(current_instruction) = messages.iter().rev().find_map(genuine_user_text) else {
        return messages.to_vec();
    };
    let Ok(quest_focus) = extract_quest_focus(messages) else {
        return messages.to_vec();
    };
    let Some(context) =
        daemon_provider_context(current_instruction, quest_focus.clone().unwrap_or_default()).await
    else {
        return messages.to_vec();
    };
    apply_provider_result(messages, decision, Some(&context), quest_focus.as_deref())
}

fn apply_provider_result(
    messages: &[Value],
    _decision: GateDecision,
    provider_context: Option<&str>,
    quest_focus: Option<&str>,
) -> Vec<Value> {
    provider_context
        .and_then(|context| compose_provider_messages(messages, context, quest_focus))
        .unwrap_or_else(|| messages.to_vec())
}

/// The latest genuine user message, extracted from the request body BEFORE
/// KISS's compression mutates it. Later user-role entries may be tool results
/// or injected harness context, so this uses the same genuine-user predicate
/// as current-turn compression protection.
/// Harness-injected content delivered as a plain `type: "text"` block inside a
/// synthetic user-role turn -- background Task-tool completions, system
/// reminders, and local-command output all arrive this way, not as a
/// `tool_result` block, so the type check alone can't tell them apart from a
/// genuine human message. Confirmed via `laptop_export.jsonl`: ~29% of one
/// exported substrate were raw `<task-notification>...</task-notification>`
/// deliveries deposited verbatim as "user" experience.
fn is_synthetic_harness_text(text: &str) -> bool {
    const MARKERS: &[&str] = &[
        NEUROGRAPH_SURFACED_MARKER,
        "<task-notification>",
        "<system-reminder>",
        "<local-command-stdout>",
        "<local-command-caveat>",
    ];
    let trimmed = text.trim_start();
    MARKERS.iter().any(|m| trimmed.starts_with(m))
}

/// Claude Code represents tool results and injected harness context as
/// user-role messages too. Return only genuine human text blocks so raw
/// last-user deposit, turn cadence, and current-instruction protection all use
/// one semantic boundary. A mixed array keeps every human block and excludes
/// independently injected context blocks.
fn genuine_user_text(msg: &Value) -> Option<String> {
    if msg["role"].as_str() != Some("user")
        || msg["isMeta"].as_bool() == Some(true)
        || is_compact_summary_message(msg)
    {
        return None;
    }
    let is_genuine = |text: &str| {
        !text.trim().is_empty()
            && !is_synthetic_harness_text(text)
            && !is_claude_compaction_request_text(text)
            && !is_claude_compacted_summary_text(text)
    };
    let text = match &msg["content"] {
        Value::String(text) if is_genuine(text) => text.clone(),
        Value::Array(blocks) => {
            // A tool-result delivery may carry neighboring text blocks from
            // the harness. The whole user-role turn remains tool traffic.
            if blocks
                .iter()
                .any(|block| block["type"].as_str() == Some("tool_result"))
            {
                return None;
            }
            blocks
                .iter()
                .filter(|block| block["type"].as_str() == Some("text"))
                .filter_map(|block| block["text"].as_str())
                .filter(|text| is_genuine(text))
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => return None,
    };
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn is_genuine_user_message(msg: &Value) -> bool {
    genuine_user_text(msg).is_some()
}

fn last_genuine_user_message_index(messages: &[Value]) -> Option<usize> {
    messages.iter().rposition(is_genuine_user_message)
}

fn extract_last_user_message(body_bytes: &[u8]) -> Option<String> {
    let body: Value = serde_json::from_slice(body_bytes).ok()?;
    let messages = body["messages"].as_array()?;
    messages.iter().rev().find_map(genuine_user_text)
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
        let Some(json_str) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(json_str) else {
            continue;
        };
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

    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();

    let is_messages = method == axum::http::Method::POST && uri.path() == "/v1/messages";

    // Buffer the request body (needed for KISS rewrite; also needed to forward).
    let body_bytes = axum::body::to_bytes(req.into_body(), MAX_BODY)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Extract the latest genuine user message BEFORE KISS mutation. Trailing
    // user-role entries may be tool results or injected harness context.
    let last_user_message = if is_messages {
        extract_last_user_message(&body_bytes)
    } else {
        None
    };

    // Claude's native summary request must reach the provider byte-for-byte and
    // cannot consume a human-turn gate decision.
    // Rewrite messages only when Claude supplied stable request identity and a
    // genuine human marker. Every other case fails open without shared state.
    let body_bytes = if is_messages {
        let rewritten = 'rewrite: {
            let Ok(mut body) = serde_json::from_slice::<Value>(&body_bytes) else {
                break 'rewrite None;
            };
            let Some(arr) = body["messages"].as_array().cloned() else {
                break 'rewrite None;
            };
            let Some(decision) = gate_decision_for_body(&body, &headers, &state.sessions) else {
                break 'rewrite None;
            };
            body["messages"] = Value::Array(
                if std::env::var("MINITID_PITH_PENINSULA")
                    .map(|v| v == "1")
                    .unwrap_or(false)
                {
                    apply_pith_peninsula(&arr, decision).await
                } else {
                    apply_kiss(&arr, decision)
                },
            );
            break 'rewrite serde_json::to_vec(&body).ok();
        };
        rewritten.map(Into::into).unwrap_or(body_bytes)
    } else {
        body_bytes
    };

    // Build upstream URL — read config file live so switching providers
    // takes effect immediately without restarting the service.
    let upstream = read_upstream(&state.upstream_fallback);
    let path_and_query = uri
        .path_and_query()
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

    let upstream_resp = rb
        .send()
        .await
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
                    let _ = tx.send(Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    )));
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
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let upstream =
        env::var("MINITID_UPSTREAM").unwrap_or_else(|_| "https://api.anthropic.com".to_string());

    let state = Arc::new(AppState {
        client: Client::builder().build().expect("reqwest client"),
        sessions: Mutex::new(SessionStore::new(session_capacity())),
        upstream_fallback: upstream.clone(),
    });

    let app = Router::new().fallback(proxy).with_state(state);

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

    fn compress_with_faux(messages: &[Value]) -> Vec<Value> {
        apply_kiss(messages, GateDecision::Compress)
    }

    fn headers(agent_id: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(agent_id) = agent_id {
            headers.insert(
                "x-claude-code-agent-id",
                agent_id.parse().expect("valid agent header"),
            );
        }
        headers
    }

    fn test_sessions() -> Mutex<SessionStore> {
        Mutex::new(SessionStore::new(DEFAULT_SESSION_CAPACITY))
    }

    fn body(session_id: &str, messages: Vec<Value>) -> Value {
        json!({
            "metadata": {
                "user_id": serde_json::to_string(&json!({
                    "device_id": "device",
                    "account_uuid": "account",
                    "session_id": session_id
                })).unwrap()
            },
            "messages": messages
        })
    }

    fn conversation(human_texts: &[&str]) -> Vec<Value> {
        let mut messages = Vec::new();
        for (index, text) in human_texts.iter().enumerate() {
            if index > 0 {
                messages.push(json!({"role": "assistant", "content": "reply"}));
            }
            messages.push(json!({"role": "user", "content": text}));
        }
        messages
    }

    fn advance_human_turns(
        sessions: &Mutex<SessionStore>,
        session_id: &str,
        human_texts: &[&str],
    ) -> GateDecision {
        let mut decision = GateDecision::FullPass;
        for index in 0..human_texts.len() {
            decision = gate_decision_for_body(
                &body(session_id, conversation(&human_texts[..=index])),
                &headers(None),
                sessions,
            )
            .unwrap();
        }
        decision
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

    const INCIDENT_TASK: &str = "You are the bounded source-and-runtime analyst for the current Claude Code NeuroGraph surfacing mission. Trace the exact path, preserve all restrictions, write the requested PLAN.md, and do not edit repositories or runtime state. This text intentionally exceeds the Pith keyframe budget.";

    fn current_turn_with_tool_pairs(pair_count: usize, array_prompt: bool) -> Vec<Value> {
        let first = if array_prompt {
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": INCIDENT_TASK},
                    {"type": "text", "text": "The second human text block must also survive."}
                ]
            })
        } else {
            json!({"role": "user", "content": INCIDENT_TASK})
        };
        let mut messages = vec![first];
        for i in 0..pair_count {
            messages.push(json!({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": format!("tool_{i}"),
                    "name": "Read",
                    "input": {"file_path": format!("file_{i}.rs")}
                }]
            }));
            messages.push(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": format!("tool_{i}"),
                    "content": format!("tool output {i}")
                }]
            }));
        }
        messages
    }

    #[test]
    fn current_human_task_survives_tool_growth_in_faux_path() {
        // One human task + five tool-use/result pairs = 11 entries. Previously
        // index 0 fell below n-10 and was compressed during the same user turn.
        let original = current_turn_with_tool_pairs(5, false);
        assert_eq!(original.len(), 11);
        let out = compress_with_faux(&original);
        assert_eq!(out[0], original[0]);
        assert!(out[0]["content"].as_str().unwrap().contains("PLAN.md"));
    }

    #[test]
    fn current_multiblock_human_message_is_preserved_as_whole_value() {
        let original = current_turn_with_tool_pairs(6, true);
        let compress_before = original.len() - KISS_RECENT_WINDOW;
        let indices = compression_indices(&original, compress_before);
        assert!(!indices.contains(&0));
        let out = apply_faux_at_indices(&original, &indices);
        assert_eq!(out[0], original[0]);
        assert_eq!(out[0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn provider_composition_evicts_old_history_and_keeps_live_rails_once() {
        let quest = format!(
            "{QUEST_TRACKER_BANNER} This is what you were working on and why.\n\nCurrent: Slice B"
        );
        let wrapped_quest = format!(
            "<system-reminder>\nSessionStart hook additional context: {quest}\n</system-reminder>"
        );
        let current = json!({
            "role": "user",
            "content": [{"type": "text", "text": INCIDENT_TASK}]
        });
        let messages = vec![
            json!({"role": "user", "content": wrapped_quest}),
            json!({"role": "assistant", "content": LONG}),
            current.clone(),
        ];
        let extracted = extract_quest_focus(&messages).unwrap().unwrap();
        assert_eq!(extracted, quest);
        let out = compose_provider_messages(
            &messages,
            "## Who I Am\n- constitutional core\n\n## Learned Situation\n- learned topology",
            Some(&extracted),
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[2], current);
        assert_eq!(count_marker(&out, NEUROGRAPH_SURFACED_MARKER), 1);
        assert_eq!(count_marker(&out, QUEST_TRACKER_BANNER), 1);
        assert_eq!(genuine_user_text(&out[0]), None);
        assert_eq!(genuine_user_text(&out[1]), None);
        assert_eq!(
            out[1]["content"].as_str(),
            Some(
                format!(
                    "<system-reminder>\nSessionStart hook additional context: {quest}\n</system-reminder>"
                )
                .as_str()
            )
        );
        assert!(!serde_json::to_string(&out).unwrap().contains(LONG));
        assert!(!serde_json::to_string(&out).unwrap().contains('…'));
        assert_eq!(
            serde_json::to_string(&out)
                .unwrap()
                .matches(INCIDENT_TASK)
                .count(),
            1
        );
    }

    #[test]
    fn blank_session_receives_fresh_mind_and_exact_instruction() {
        let current = json!({"role": "user", "content": "Begin the active mission"});
        let out = compose_provider_messages(
            &[current.clone()],
            "## Who I Am\n- constitutional core\n\n## Learned Situation\n- active mission",
            None,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1], current);
        assert_eq!(count_marker(&out, NEUROGRAPH_SURFACED_MARKER), 1);
    }

    #[test]
    fn quoted_quest_banner_is_human_text_and_cannot_spoof_focus() {
        let prefixed_quote = json!({
            "role": "user",
            "content": format!("Please explain this phrase: {QUEST_TRACKER_BANNER}")
        });
        let exact_leading_quote = json!({
            "role": "user",
            "content": format!("{QUEST_TRACKER_BANNER} I am quoting this banner, not injecting a Quest.")
        });
        assert!(is_genuine_user_message(&prefixed_quote));
        assert!(is_genuine_user_message(&exact_leading_quote));
        assert_eq!(extract_quest_focus(&[prefixed_quote]), Ok(None));
        assert_eq!(extract_quest_focus(&[exact_leading_quote]), Ok(None));
    }

    #[test]
    fn human_quest_quote_cannot_replace_verified_orientation_rail() {
        let quest = format!("{QUEST_TRACKER_BANNER}\n\nCurrent: real Quest orientation");
        let verified = json!({
            "role": "user",
            "content": format!(
                "<system-reminder>\nSessionStart hook additional context: {quest}\n</system-reminder>"
            )
        });
        let quotation = json!({
            "role": "user",
            "content": format!(
                "{QUEST_TRACKER_BANNER} I am discussing this literal text as the current human request."
            )
        });
        let messages = vec![verified, quotation.clone()];
        let extracted = extract_quest_focus(&messages).unwrap().unwrap();
        assert_eq!(extracted, quest);
        let out = compose_provider_messages(&messages, "## Who I Am\n- identity", Some(&extracted))
            .unwrap();
        assert_eq!(verified_quest_rails(&out).unwrap(), vec![quest]);
        assert_eq!(count_marker(&out, QUEST_TRACKER_BANNER), 2);
        assert_eq!(out.last(), Some(&quotation));
    }

    #[test]
    fn multiple_banners_in_one_trusted_quest_envelope_are_ambiguous() {
        let ambiguous = json!({
            "role": "user",
            "content": format!(
                "<system-reminder>\nSessionStart hook additional context: {QUEST_TRACKER_BANNER}\nfirst\n{QUEST_TRACKER_BANNER}\nsecond\n</system-reminder>"
            )
        });
        assert_eq!(extract_quest_focus(&[ambiguous]), Err(()));
    }

    #[test]
    fn newer_human_turn_releases_prior_human_turn_for_compression() {
        let mut original = vec![
            json!({"role": "user", "content": INCIDENT_TASK}),
            json!({"role": "assistant", "content": LONG}),
            json!({"role": "user", "content": "new genuine instruction that must remain verbatim even after enough later tool traffic pushes it outside the numeric recent window"}),
        ];
        for i in 0..6 {
            original.push(json!({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": format!("later_{i}"), "name": "Read", "input": {}}]
            }));
            original.push(json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": format!("later_{i}"), "content": "result"}]
            }));
        }
        let indices = compression_indices(&original, original.len() - KISS_RECENT_WINDOW);
        assert!(indices.contains(&0));
        assert!(!indices.contains(&2));
        let out = apply_faux_at_indices(&original, &indices);
        assert_ne!(out[0], original[0]);
        assert_eq!(out[2], original[2]);
    }

    #[test]
    fn synthetic_user_entries_do_not_steal_current_human_protection() {
        let mut messages = current_turn_with_tool_pairs(5, false);
        messages.push(json!({
            "role": "user",
            "isMeta": true,
            "content": "meta"
        }));
        messages.push(json!({
            "role": "user",
            "isCompactSummary": true,
            "content": "summary"
        }));
        messages.push(json!({
            "role": "user",
            "content": "<system-reminder>injected</system-reminder>"
        }));
        assert_eq!(last_genuine_user_message_index(&messages), Some(0));
        let indices = compression_indices(&messages, messages.len() - KISS_RECENT_WINDOW);
        assert!(!indices.contains(&0));
    }

    #[test]
    fn text_blocks_compressed_tool_blocks_preserved() {
        let original = sample_messages();
        let out = compress_with_faux(&original);

        // Index 0: text block compressed, tool_result untouched.
        let blocks0 = out[0]["content"].as_array().expect("array content");
        assert_eq!(blocks0.len(), 2, "block count must be preserved");
        let t0 = blocks0[0]["text"].as_str().unwrap();
        assert!(!t0.is_empty(), "text block must never be blanked");
        assert!(t0.len() < LONG.len(), "text block should be compressed");
        assert_eq!(
            blocks0[1], original[0]["content"][1],
            "tool_result block must survive verbatim"
        );

        // Index 1: text block compressed, tool_use untouched (input intact).
        let blocks1 = out[1]["content"].as_array().unwrap();
        assert_eq!(
            blocks1[1], original[1]["content"][1],
            "tool_use block must survive verbatim"
        );
        assert_eq!(blocks1[1]["input"]["command"], "ls -la");
    }

    #[test]
    fn recent_window_untouched() {
        let original = sample_messages();
        let out = compress_with_faux(&original);
        // Indices 2..12 are the recent window — byte-identical.
        for i in 2..original.len() {
            assert_eq!(
                out[i], original[i],
                "recent message {} must be untouched",
                i
            );
        }
    }

    #[test]
    fn no_message_is_blanked() {
        let out = compress_with_faux(&sample_messages());
        for (i, m) in out.iter().enumerate() {
            match &m["content"] {
                Value::String(s) => assert!(!s.is_empty(), "msg {} string blanked", i),
                Value::Array(blocks) => {
                    for b in blocks {
                        if b["type"].as_str() == Some("text") {
                            assert!(
                                !b["text"].as_str().unwrap_or("").is_empty(),
                                "msg {} text block blanked",
                                i
                            );
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
        assert_eq!(
            out, content,
            "whitespace block must be preserved, not blanked"
        );
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
    fn metadata_identity_survives_first_message_replacement() {
        let a = body("session-alpha", conversation(&["original first prompt"]));
        let b = body(
            "session-alpha",
            conversation(&["replacement after compaction"]),
        );
        assert_eq!(
            session_key(&a, &headers(None)),
            session_key(&b, &headers(None))
        );
    }

    #[test]
    fn sessions_with_the_same_first_prompt_are_isolated() {
        let messages = conversation(&["shared opening prompt"]);
        let a = body("session-alpha", messages.clone());
        let b = body("session-beta", messages);
        let sessions = test_sessions();
        assert_eq!(
            gate_decision_for_body(&a, &headers(None), &sessions),
            Some(GateDecision::FullPass)
        );
        assert_eq!(
            gate_decision_for_body(&b, &headers(None), &sessions),
            Some(GateDecision::FullPass)
        );
        assert_eq!(sessions.lock().unwrap().entries.len(), 2);
    }

    #[test]
    fn missing_or_malformed_metadata_fails_open_without_state() {
        let messages = conversation(&["human prompt"]);
        let cases = [
            json!({"messages": messages}),
            json!({"metadata": {"user_id": "not-json"}, "messages": conversation(&["human prompt"])}),
            json!({"metadata": {"user_id": "{}"}, "messages": conversation(&["human prompt"])}),
        ];
        let sessions = test_sessions();
        for body in &cases {
            assert_eq!(
                gate_decision_for_body(body, &headers(None), &sessions),
                None
            );
        }
        assert!(sessions.lock().unwrap().entries.is_empty());
    }

    #[test]
    fn configured_session_capacity_is_finitely_clamped() {
        assert_eq!(configured_session_capacity(None), DEFAULT_SESSION_CAPACITY);
        assert_eq!(
            configured_session_capacity(Some("invalid")),
            DEFAULT_SESSION_CAPACITY
        );
        assert_eq!(configured_session_capacity(Some("0")), MIN_SESSION_CAPACITY);
        assert_eq!(configured_session_capacity(Some("1")), MIN_SESSION_CAPACITY);
        assert_eq!(configured_session_capacity(Some("128")), 128);
        assert_eq!(
            configured_session_capacity(Some("999999999")),
            MAX_SESSION_CAPACITY
        );
    }

    #[test]
    fn configured_pith_tool_tail_bytes_defaults_and_clamps() {
        assert_eq!(
            configured_pith_tool_tail_bytes(None),
            DEFAULT_PITH_TOOL_TAIL_BYTES
        );
        assert_eq!(
            configured_pith_tool_tail_bytes(Some("malformed")),
            DEFAULT_PITH_TOOL_TAIL_BYTES
        );
        assert_eq!(
            configured_pith_tool_tail_bytes(Some("0")),
            MIN_PITH_TOOL_TAIL_BYTES
        );
        assert_eq!(
            configured_pith_tool_tail_bytes(Some("65536")),
            DEFAULT_PITH_TOOL_TAIL_BYTES
        );
        assert_eq!(
            configured_pith_tool_tail_bytes(Some("999999999")),
            MAX_PITH_TOOL_TAIL_BYTES
        );
    }

    #[test]
    fn poisoned_session_lock_fails_open() {
        let sessions = test_sessions();
        let _ = std::panic::catch_unwind(|| {
            let _guard = sessions.lock().unwrap();
            panic!("poison cadence state for regression coverage");
        });
        let request = body("poisoned-session", conversation(&["human prompt"]));
        assert_eq!(
            gate_decision_for_body(&request, &headers(None), &sessions),
            None
        );
    }

    #[test]
    fn agent_header_partitions_concurrent_sidechains() {
        let request = body("parent-session", conversation(&["delegated prompt"]));
        let parent = session_key(&request, &headers(None)).unwrap();
        let agent_a = session_key(&request, &headers(Some("agent-a"))).unwrap();
        let agent_b = session_key(&request, &headers(Some("agent-b"))).unwrap();
        assert_ne!(parent, agent_a);
        assert_ne!(agent_a, agent_b);

        let oversized = "a".repeat(MAX_AGENT_ID_BYTES + 1);
        assert_eq!(session_key(&request, &headers(Some(&oversized))), None);
    }

    #[test]
    fn tool_loop_reuses_one_human_turn_decision() {
        let sessions = test_sessions();
        let mut messages = conversation(&["one human turn"]);
        for i in 0..8 {
            let request = body("tool-loop-session", messages.clone());
            assert_eq!(
                gate_decision_for_body(&request, &headers(None), &sessions),
                Some(GateDecision::FullPass)
            );
            messages.push(json!({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": format!("tool-{i}"), "name": "Read", "input": {}}]
            }));
            messages.push(json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": format!("tool-{i}"), "content": "result"}]
            }));
        }
        let map = sessions.lock().unwrap();
        let state = map.entries.values().next().unwrap();
        assert_eq!(state.turn_count, 1);
    }

    #[test]
    fn changed_and_repeated_identical_human_turns_advance() {
        let sessions = test_sessions();
        let first = body("cadence-session", conversation(&["same prompt"]));
        let repeated = body(
            "cadence-session",
            conversation(&["same prompt", "same prompt"]),
        );
        let changed = body(
            "cadence-session",
            conversation(&["same prompt", "same prompt", "different prompt"]),
        );
        assert_eq!(
            gate_decision_for_body(&first, &headers(None), &sessions),
            Some(GateDecision::FullPass)
        );
        assert_eq!(
            gate_decision_for_body(&repeated, &headers(None), &sessions),
            Some(GateDecision::FullPass)
        );
        assert_eq!(
            gate_decision_for_body(&changed, &headers(None), &sessions),
            Some(GateDecision::FullPass)
        );
        assert_eq!(
            sessions
                .lock()
                .unwrap()
                .entries
                .values()
                .next()
                .unwrap()
                .turn_count,
            3
        );
    }

    #[test]
    fn compaction_count_shrink_does_not_advance() {
        let sessions = test_sessions();
        for turns in [
            vec!["first"],
            vec!["first", "second"],
            vec!["first", "second", "third"],
        ] {
            let request = body("shrink-session", conversation(&turns));
            gate_decision_for_body(&request, &headers(None), &sessions).unwrap();
        }

        let shrunk = body("shrink-session", conversation(&["third"]));
        assert_eq!(
            gate_decision_for_body(&shrunk, &headers(None), &sessions),
            Some(GateDecision::FullPass)
        );
        assert_eq!(
            sessions
                .lock()
                .unwrap()
                .entries
                .values()
                .next()
                .unwrap()
                .turn_count,
            3
        );

        let next = body("shrink-session", conversation(&["third", "fourth"]));
        assert_eq!(
            gate_decision_for_body(&next, &headers(None), &sessions),
            Some(GateDecision::Compress)
        );
    }

    #[test]
    fn first_three_human_turns_are_full_and_fourth_compresses() {
        let sessions = test_sessions();
        let expected = [
            GateDecision::FullPass,
            GateDecision::FullPass,
            GateDecision::FullPass,
            GateDecision::Compress,
        ];
        let all = ["one", "two", "three", "four"];
        for (index, expected) in expected.iter().enumerate() {
            let request = body("warmup-session", conversation(&all[..=index]));
            assert_eq!(
                gate_decision_for_body(&request, &headers(None), &sessions),
                Some(*expected)
            );
        }
    }

    #[test]
    fn force_full_cadence_counts_human_turns() {
        let sessions = test_sessions();
        let human_turns: Vec<String> = (1..=KISS_WARMUP_TURNS + KISS_FORCE_FULL_EVERY)
            .map(|n| format!("turn {n}"))
            .collect();
        for index in 0..human_turns.len() {
            let texts: Vec<&str> = human_turns[..=index].iter().map(String::as_str).collect();
            let request = body("force-full-session", conversation(&texts));
            let decision = gate_decision_for_body(&request, &headers(None), &sessions).unwrap();
            if index + 1 == human_turns.len() {
                assert_eq!(decision, GateDecision::FullPass);
            } else if index >= KISS_WARMUP_TURNS as usize {
                assert_eq!(decision, GateDecision::Compress);
            }
        }
    }

    #[test]
    fn compaction_header_bypasses_without_advancing() {
        let sessions = test_sessions();
        let request = body("compaction-session", conversation(&["human prompt"]));
        let mut request_headers = headers(None);
        request_headers.insert("x-cc-compaction-request", "1".parse().unwrap());
        assert_eq!(
            gate_decision_for_body(&request, &request_headers, &sessions),
            None
        );
        assert!(sessions.lock().unwrap().entries.is_empty());
    }

    #[test]
    fn compaction_prompt_bypasses_without_header_or_state_change() {
        let sessions = test_sessions();
        let normal_messages = conversation(&["one", "two", "three", "four"]);
        let normal = body("compaction-session", normal_messages.clone());
        assert_eq!(
            advance_human_turns(
                &sessions,
                "compaction-session",
                &["one", "two", "three", "four"]
            ),
            GateDecision::Compress
        );

        let key = session_key(&normal, &headers(None)).unwrap();
        let (before_map, before_clock) = {
            let store = sessions.lock().unwrap();
            (store.entries.clone(), store.access_clock)
        };
        let mut compacting_messages = normal_messages.clone();
        compacting_messages.push(json!({"role": "assistant", "content": "ready"}));
        compacting_messages.push(json!({
            "role": "user",
            "content": format!("{CLAUDE_COMPACTION_PROMPT_PREFIX}\n\nSummarize now.")
        }));
        let compacting = body("compaction-session", compacting_messages);
        let original = compacting.clone();
        assert_eq!(
            gate_decision_for_body(&compacting, &headers(None), &sessions),
            None
        );
        assert_eq!(compacting, original, "bypass must not mutate the body");
        let store = sessions.lock().unwrap();
        assert_eq!(store.entries, before_map);
        assert_eq!(store.access_clock, before_clock);
        drop(store);

        let mut tool_loop = normal_messages;
        tool_loop.push(json!({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}}]
        }));
        tool_loop.push(json!({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "result"}]
        }));
        assert_eq!(
            gate_decision_for_body(
                &body("compaction-session", tool_loop),
                &headers(None),
                &sessions
            ),
            Some(GateDecision::Compress)
        );
        assert_eq!(
            sessions
                .lock()
                .unwrap()
                .entries
                .get(&key)
                .unwrap()
                .turn_count,
            4
        );
    }

    #[test]
    fn compacted_summary_preamble_is_not_a_request_bypass() {
        let sessions = test_sessions();
        let mut messages = conversation(&["one", "two", "three", "four"]);
        assert_eq!(
            advance_human_turns(
                &sessions,
                "summary-session",
                &["one", "two", "three", "four"]
            ),
            GateDecision::Compress
        );
        messages.push(json!({"role": "assistant", "content": "summary follows"}));
        messages.push(json!({
            "role": "user",
            "content": format!("{CLAUDE_COMPACTED_PREAMBLE}retained context")
        }));
        assert_eq!(
            gate_decision_for_body(
                &body("summary-session", messages),
                &headers(None),
                &sessions
            ),
            Some(GateDecision::Compress)
        );
    }

    #[test]
    fn injected_blocks_do_not_advance_mixed_human_turn() {
        let sessions = test_sessions();
        assert_eq!(
            advance_human_turns(&sessions, "mixed-block-session", &["one", "two", "three"]),
            GateDecision::FullPass
        );
        let mut messages = conversation(&["one", "two", "three"]);
        messages.push(json!({"role": "assistant", "content": "reply"}));
        messages.push(json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "the real fourth instruction"},
                {"type": "text", "text": "[NeuroGraph Surfaced Knowledge]\nold surface"},
                {"type": "text", "text": "<system-reminder>old reminder</system-reminder>"},
                {"type": "text", "text": "<task-notification>old task</task-notification>"},
                {"type": "text", "text": "<local-command-stdout>old output</local-command-stdout>"},
                {"type": "text", "text": "<local-command-caveat>old caveat</local-command-caveat>"}
            ]
        }));
        let first = body("mixed-block-session", messages.clone());
        assert_eq!(
            gate_decision_for_body(&first, &headers(None), &sessions),
            Some(GateDecision::Compress)
        );

        let blocks = messages.last_mut().unwrap()["content"]
            .as_array_mut()
            .unwrap();
        blocks[1]["text"] = json!("[NeuroGraph Surfaced Knowledge]\nnew surface");
        blocks[2]["text"] = json!("<system-reminder>new reminder</system-reminder>");
        blocks[3]["text"] = json!("<task-notification>new task</task-notification>");
        blocks[4]["text"] = json!("<local-command-stdout>new output</local-command-stdout>");
        blocks[5]["text"] = json!("<local-command-caveat>new caveat</local-command-caveat>");
        assert_eq!(
            gate_decision_for_body(
                &body("mixed-block-session", messages),
                &headers(None),
                &sessions
            ),
            Some(GateDecision::Compress)
        );
        let store = sessions.lock().unwrap();
        let state = store.entries.values().next().unwrap();
        assert_eq!(state.turn_count, 4);
        assert_eq!(state.visible_human_count, 4);
    }

    #[test]
    fn capped_session_store_evicts_stale_and_preserves_active_cadence() {
        let sessions = Mutex::new(SessionStore::new(2));
        let stale = body("stale-session", conversation(&["stale"]));
        gate_decision_for_body(&stale, &headers(None), &sessions).unwrap();

        let active_messages = conversation(&["one", "two", "three", "four"]);
        let active = body("active-session", active_messages.clone());
        assert_eq!(
            advance_human_turns(
                &sessions,
                "active-session",
                &["one", "two", "three", "four"]
            ),
            GateDecision::Compress
        );
        let stale_key = session_key(&stale, &headers(None)).unwrap();
        let active_key = session_key(&active, &headers(None)).unwrap();

        let newcomer = body("new-session", conversation(&["new"]));
        gate_decision_for_body(&newcomer, &headers(None), &sessions).unwrap();
        let newcomer_key = session_key(&newcomer, &headers(None)).unwrap();
        {
            let store = sessions.lock().unwrap();
            assert_eq!(store.entries.len(), 2);
            assert!(!store.entries.contains_key(&stale_key));
            assert!(store.entries.contains_key(&active_key));
            assert!(store.entries.contains_key(&newcomer_key));
        }

        let mut active_tool_loop = active_messages;
        active_tool_loop.push(json!({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool", "name": "Read", "input": {}}]
        }));
        active_tool_loop.push(json!({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool", "content": "result"}]
        }));
        assert_eq!(
            gate_decision_for_body(
                &body("active-session", active_tool_loop),
                &headers(None),
                &sessions
            ),
            Some(GateDecision::Compress)
        );
        assert_eq!(
            sessions
                .lock()
                .unwrap()
                .entries
                .get(&active_key)
                .unwrap()
                .turn_count,
            4
        );
    }

    #[test]
    fn compact_summaries_are_excluded_from_sparse_indices() {
        let mut messages = vec![
            json!({"role": "user", "isCompactSummary": true, "content": LONG}),
            json!({"role": "assistant", "content": LONG}),
        ];
        for i in 0..10 {
            messages.push(json!({"role": "assistant", "content": format!("recent {i}")}));
        }
        assert_eq!(compression_indices(&messages, 2), vec![1]);

        messages[0] = json!({
            "role": "user",
            "content": format!("{CLAUDE_COMPACTED_PREAMBLE}Summary: retained context")
        });
        assert_eq!(compression_indices(&messages, 2), vec![1]);
    }

    #[test]
    fn current_claude_wire_preamble_is_not_genuine_human_text() {
        let summary = json!({
            "role": "user",
            "content": format!("{CLAUDE_COMPACTED_PREAMBLE}Summary: retained context")
        });
        let prompt = json!({
            "role": "user",
            "content": format!("{CLAUDE_COMPACTION_PROMPT_PREFIX} — you will fail the task.")
        });
        assert!(!is_genuine_user_message(&summary));
        assert!(!is_genuine_user_message(&prompt));
        assert_eq!(human_turn_marker(&[summary, prompt]), None);
    }

    #[test]
    fn mixed_human_and_surface_blocks_keep_human_block_exact() {
        let human =
            json!({"type": "text", "text": INCIDENT_TASK, "cache_control": {"type": "ephemeral"}});
        let noise =
            json!({"type": "text", "text": format!("{NEUROGRAPH_SURFACED_MARKER}\nold noise")});
        let current = json!({"role": "user", "content": [human.clone(), noise]});
        let out = compose_provider_messages(&[current], "## Who I Am\n- identity", None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1]["content"].as_array().unwrap(), &[human]);
        assert_eq!(count_marker(&out, NEUROGRAPH_SURFACED_MARKER), 1);
    }

    #[test]
    fn mixed_tool_result_and_surface_keep_tool_block_exact() {
        let human = json!({"role": "user", "content": INCIDENT_TASK});
        let use_message = json!({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool-live", "name": "Read", "input": {"file_path": "x"}}]
        });
        let result = json!({"type": "tool_result", "tool_use_id": "tool-live", "content": "exact output", "is_error": false});
        let delivery = json!({
            "role": "user",
            "content": [
                result.clone(),
                {"type": "text", "text": format!("{NEUROGRAPH_SURFACED_MARKER}\nold noise")}
            ]
        });
        let out = compose_provider_messages(
            &[human.clone(), use_message.clone(), delivery],
            "## Who I Am\n- identity",
            None,
        )
        .unwrap();
        assert_eq!(out[1], human);
        assert_eq!(out[2], use_message);
        assert_eq!(out[3]["content"].as_array().unwrap(), &[result]);
    }

    #[test]
    fn event_prefixed_surface_wrappers_are_removed_but_quotes_are_not() {
        for event in ["UserPromptSubmit", "PreToolUse", "SessionStart"] {
            let wrapped = format!(
                "<system-reminder>\n{event} hook additional context: {NEUROGRAPH_SURFACED_MARKER}\nold\n</system-reminder>"
            );
            assert!(is_standalone_neurograph_surface_text(&wrapped));
        }
        let quote = format!("Human quotation before {NEUROGRAPH_SURFACED_MARKER} must remain");
        assert!(!is_standalone_neurograph_surface_text(&quote));
        let messages = vec![
            json!({"role": "user", "content": INCIDENT_TASK}),
            json!({"role": "user", "content": quote}),
        ];
        assert_eq!(
            apply_provider_result(
                &messages,
                GateDecision::Compress,
                Some("## Who I Am\n- identity"),
                None,
            ),
            messages,
            "an inseparable marker quote must fail open"
        );
    }

    #[test]
    fn meta_and_compact_clutter_leave_l1_but_tool_metadata_fails_open() {
        let human = json!({"role": "user", "content": INCIDENT_TASK});
        let meta = json!({"role": "user", "isMeta": true, "content": "stale harness data"});
        let compact =
            json!({"role": "user", "isCompactSummary": true, "content": "old L3 summary"});
        let out = compose_provider_messages(
            &[human.clone(), meta, compact],
            "## Who I Am\n- identity",
            None,
        )
        .unwrap();
        assert_eq!(
            out,
            vec![
                json!({"role": "user", "content": format!("{NEUROGRAPH_SURFACED_MARKER}\n\n## Who I Am\n- identity")}),
                human.clone(),
            ]
        );

        let suspicious = json!({
            "role": "user",
            "isMeta": true,
            "content": [{"type": "tool_result", "tool_use_id": "live", "content": "result"}]
        });
        let original = vec![human, suspicious];
        assert_eq!(
            apply_provider_result(
                &original,
                GateDecision::Compress,
                Some("## Who I Am\n- identity"),
                None,
            ),
            original
        );
    }

    #[test]
    fn long_tool_turn_is_bounded_pair_complete_and_exact() {
        let original = current_turn_with_tool_pairs(50, true);
        let out = compose_provider_messages(
            &original,
            "## Who I Am\n- identity\n\n## Learned Situation\n- current",
            None,
        )
        .unwrap();
        // Surface + exact human + five complete recent pairs.
        assert_eq!(out.len(), 12);
        assert_eq!(out[1], original[0]);
        assert_eq!(&out[2..], &original[original.len() - KISS_RECENT_WINDOW..]);
        for msg in &out {
            for result_id in tool_ids(msg, "tool_result", "tool_use_id") {
                assert!(out.iter().any(|candidate| {
                    tool_ids(candidate, "tool_use", "id")
                        .iter()
                        .any(|use_id| use_id == &result_id)
                }));
            }
        }
    }

    #[test]
    fn tool_tail_byte_budget_admits_whole_pairs_only() {
        let mut original = vec![json!({"role": "user", "content": INCIDENT_TASK})];
        let large_result = "x".repeat(20_000);
        for i in 0..20 {
            original.push(json!({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": format!("big-{i}"), "name": "Read", "input": {}}]
            }));
            original.push(json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": format!("big-{i}"), "content": large_result.clone()}]
            }));
        }
        let out = compose_provider_messages(&original, "## Who I Am\n- identity", None).unwrap();
        let exact_tail = &out[2..];
        assert!(serialized_messages_bytes(exact_tail) <= DEFAULT_PITH_TOOL_TAIL_BYTES);
        assert_eq!(exact_tail.len() % 2, 0);
        for msg in exact_tail {
            for result_id in tool_ids(msg, "tool_result", "tool_use_id") {
                assert!(exact_tail.iter().any(|candidate| {
                    tool_ids(candidate, "tool_use", "id").contains(&result_id)
                }));
            }
        }
    }

    #[test]
    fn oversized_pairs_reduce_to_newest_indivisible_pair() {
        let mut original = vec![json!({"role": "user", "content": INCIDENT_TASK})];
        let oversized = "z".repeat(DEFAULT_PITH_TOOL_TAIL_BYTES + 1024);
        for i in 0..8 {
            original.push(json!({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": format!("huge-{i}"), "name": "Read", "input": {}}]
            }));
            original.push(json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": format!("huge-{i}"), "content": oversized.clone()}]
            }));
        }
        let out = compose_provider_messages(&original, "## Who I Am\n- identity", None).unwrap();
        assert_eq!(out.len(), 4, "surface + human + newest whole pair");
        assert_eq!(&out[2..], &original[original.len() - 2..]);
        assert!(serialized_messages_bytes(&out[2..]) > DEFAULT_PITH_TOOL_TAIL_BYTES);
    }

    #[test]
    fn provider_socket_request_contains_cues_and_no_history() {
        let request = provider_context_request(INCIDENT_TASK, "current Quest focus");
        assert_eq!(request["event"], "provider_context");
        assert_eq!(request["data"]["current_instruction"], INCIDENT_TASK);
        assert_eq!(request["data"]["quest_focus"], "current Quest focus");
        assert!(request.get("messages").is_none());
        assert!(request["data"].get("turns").is_none());
        assert!(request["data"].get("history").is_none());
    }

    #[test]
    fn provider_response_contract_rejects_every_unavailable_shape() {
        let valid = json!({
            "ok": true,
            "state": "empty",
            "context": "## Who I Am\n- identity",
            "source": "cc_neurograph_topology",
            "coherence": "empty",
            "anchors": [],
            "warnings": ["topology_empty"],
            "assemblies": 0
        });
        assert_eq!(
            parse_provider_response(&serde_json::to_vec(&valid).unwrap()),
            Some("## Who I Am\n- identity".to_string())
        );
        let populated = json!({
            "ok": true,
            "state": "ok",
            "context": "## Who I Am\n- identity\n\n## Learned Situation\n- connected",
            "source": "cc_neurograph_topology",
            "coherence": "shared",
            "anchors": ["commit:abc123"],
            "warnings": ["coherence_unknown"],
            "assemblies": 1
        });
        assert!(parse_provider_response(&serde_json::to_vec(&populated).unwrap()).is_some());
        for invalid in [
            json!({"ok": false, "state": "unavailable", "context": "status", "source": "cc_neurograph_topology", "coherence": "unavailable", "anchors": [], "warnings": ["ng_unavailable"], "assemblies": 0}),
            json!({"ok": true, "state": "other", "context": "context", "source": "cc_neurograph_topology", "coherence": "empty", "anchors": [], "warnings": [], "assemblies": 0}),
            json!({"ok": true, "state": "ok", "context": "", "source": "cc_neurograph_topology", "coherence": "shared", "anchors": [], "warnings": [], "assemblies": 1}),
            json!({"ok": true, "state": "ok", "context": "context", "source": "database", "coherence": "shared", "anchors": [], "warnings": [], "assemblies": 1}),
            json!({"ok": true, "state": "ok", "context": "context", "source": "cc_neurograph_topology", "coherence": "shared", "anchors": [], "warnings": []}),
            json!({"ok": true, "state": "ok", "context": "context", "source": "cc_neurograph_topology", "coherence": 7, "anchors": [], "warnings": [], "assemblies": 1}),
            json!({"ok": true, "state": "ok", "context": "context", "source": "cc_neurograph_topology", "coherence": "shared", "anchors": "path", "warnings": [], "assemblies": 1}),
            json!({"ok": true, "state": "ok", "context": "context", "source": "cc_neurograph_topology", "coherence": "shared", "anchors": [], "warnings": [4], "assemblies": 1}),
            json!({"ok": true, "state": "empty", "context": "context", "source": "cc_neurograph_topology", "coherence": "empty", "anchors": [], "warnings": [], "assemblies": 1}),
            json!({"ok": true, "state": "ok", "context": "context", "source": "cc_neurograph_topology", "coherence": "shared", "anchors": [], "warnings": [], "assemblies": -1}),
        ] {
            assert_eq!(
                parse_provider_response(&serde_json::to_vec(&invalid).unwrap()),
                None
            );
        }
        assert_eq!(parse_provider_response(b"not json"), None);
    }

    #[test]
    fn every_provider_failure_preserves_original_without_faux_keyframes() {
        let original = current_turn_with_tool_pairs(8, false);
        for failed in [
            None,
            Some(""),
            Some(NEUROGRAPH_SURFACED_MARKER),
            Some(QUEST_TRACKER_BANNER),
        ] {
            assert_eq!(
                apply_provider_result(&original, GateDecision::Compress, failed, None),
                original,
            );
        }
        assert!(serde_json::to_string(&original)
            .unwrap()
            .contains(INCIDENT_TASK));
        assert!(!serde_json::to_string(&original).unwrap().contains('…'));
    }

    #[test]
    fn successful_pith_never_replays_history_on_warmup_or_gop() {
        let mut original = conversation(&["one", "two", "three", INCIDENT_TASK]);
        for i in 0..12 {
            original.push(json!({"role": "assistant", "content": format!("obsolete history {i}")}));
        }
        for legacy_decision in [GateDecision::FullPass, GateDecision::Compress] {
            let out = apply_provider_result(
                &original,
                legacy_decision,
                Some("## Who I Am\n- identity\n\n## Learned Situation\n- fresh"),
                None,
            );
            assert_eq!(out[1]["content"], INCIDENT_TASK);
            assert!(!serde_json::to_string(&out)
                .unwrap()
                .contains("obsolete history 0"));
        }
    }

    #[test]
    fn composing_messages_cannot_change_model_system_or_tools() {
        let original = json!({
            "model": "any-provider-model",
            "system": [{"type": "text", "text": "static system"}],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old reply"},
                {"role": "user", "content": INCIDENT_TASK}
            ]
        });
        let mut rewritten = original.clone();
        rewritten["messages"] = Value::Array(
            compose_provider_messages(
                original["messages"].as_array().unwrap(),
                "## Who I Am\n- model-neutral identity",
                None,
            )
            .unwrap(),
        );
        assert_eq!(rewritten["model"], original["model"]);
        assert_eq!(rewritten["system"], original["system"]);
        assert_eq!(rewritten["tools"], original["tools"]);
        assert!(!serde_json::to_string(&rewritten["messages"])
            .unwrap()
            .contains("old reply"));
    }

    #[test]
    fn warmup_passes_through_untouched() {
        let original = sample_messages();
        let out = apply_kiss(&original, GateDecision::FullPass);
        assert_eq!(
            out[0], original[0],
            "warmup turn must pass through verbatim"
        );
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
    fn test_extract_mixed_array_keeps_human_blocks_only() {
        let body = br#"{"messages":[{
            "role":"user",
            "content":[
                {"type":"text","text":"first human block"},
                {"type":"text","text":"[NeuroGraph Surfaced Knowledge]\nchanging surface"},
                {"type":"text","text":"<system-reminder>changing reminder</system-reminder>"},
                {"type":"text","text":"second human block"}
            ]
        }]}"#;
        assert_eq!(
            extract_last_user_message(body),
            Some("first human block\nsecond human block".to_string())
        );
    }

    #[test]
    fn test_extract_last_user_message_returns_none_on_malformed_body() {
        let body = b"not json";
        assert_eq!(extract_last_user_message(body), None);
    }

    #[test]
    fn test_extract_last_user_message_skips_task_notification_array_block() {
        let body = br#"{"messages":[
            {"role":"user","content":"the real question"},
            {"role":"assistant","content":"working on it"},
            {"role":"user","content":[{"type":"text","text":"<task-notification>\n<task-id>a123</task-id>\n<result>some subagent report</result>\n</task-notification>"}]}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("the real question".to_string()));
    }

    #[test]
    fn test_extract_last_user_message_skips_system_reminder_string_content() {
        let body = br#"{"messages":[
            {"role":"user","content":"the real question"},
            {"role":"assistant","content":"working on it"},
            {"role":"user","content":"<system-reminder>some injected reminder text</system-reminder>"}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("the real question".to_string()));
    }

    #[test]
    fn test_extract_last_user_message_returns_none_when_only_synthetic_present() {
        let body = br#"{"messages":[
            {"role":"user","content":[{"type":"text","text":"<local-command-stdout>ls output</local-command-stdout>"}]}
        ]}"#;
        assert_eq!(extract_last_user_message(body), None);
    }

    #[test]
    fn test_extract_skips_ismeta_true_message() {
        let body = br#"{"messages":[
            {"role":"user","content":"the real question"},
            {"role":"assistant","content":"..."},
            {"role":"user","isMeta":true,"content":"injected meta text"}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("the real question".to_string()));
    }

    #[test]
    fn test_extract_skips_iscompactsummary_true_message() {
        let body = br#"{"messages":[
            {"role":"user","content":"the real question"},
            {"role":"assistant","content":"..."},
            {"role":"user","isCompactSummary":true,"content":[{"type":"text","text":"prior conversation summary..."}]}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("the real question".to_string()));
    }

    #[test]
    fn test_extract_skips_tool_result_array_block() {
        let body = br#"{"messages":[
            {"role":"user","content":"the real question"},
            {"role":"assistant","content":"..."},
            {"role":"user","content":[{"type":"tool_result","content":"..."}]}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("the real question".to_string()));
    }

    #[test]
    fn test_extract_skips_tool_result_turn_with_neighboring_text() {
        let body = br#"{"messages":[
            {"role":"user","content":"the real question"},
            {"role":"assistant","content":"..."},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"tool-1","content":"result"},
                {"type":"text","text":"neighboring harness text that is not a human turn"}
            ]}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("the real question".to_string()));
    }

    #[test]
    fn test_extract_ismeta_false_not_skipped() {
        let body = br#"{"messages":[
            {"role":"user","isMeta":false,"content":"still a real question"}
        ]}"#;
        let result = extract_last_user_message(body);
        assert_eq!(result, Some("still a real question".to_string()));
    }

    #[test]
    fn test_deposit_turn_writes_two_experience_entries() {
        use ng_tract::read::{ReadResult, TractReader};
        use ng_tract::TractEntry;

        let tmp =
            std::env::temp_dir().join(format!("minitid_test_tract_{}.tract", std::process::id()));
        std::env::set_var("CC_GATEWAY_TRACT_PATH", tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);

        deposit_turn(
            Some("what is the numpy issue".to_string()),
            "it's a stray .pth file".to_string(),
        );

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
