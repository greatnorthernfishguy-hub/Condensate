# Return — kiss-pith-punchlist-correction-001

## Commit

- Branch: `worktree-demo-reimagining`
- Commit: `de056cb`
- Message: `kiss-pith: correct comments after faux-KISS deletion (R1+R2)`
- Files changed: `rust_core/src/minitid.rs` only

## Diff summary

- Added `[2026-09-21]` changelog entry recording the faux-KISS deletion and honest passthrough (R1).
- Rewrote the `KISS behaviour` header to describe the current contract: this June snapshot no longer truncates; `/v1/messages` forwards original bytes (R2).
- Left `KissSession` / `KISS_*` in place as dead-code scaffolding.
- Rewrote the `DROP_REQ_HEADERS` prose to state that `content-length` is excluded because reqwest recomputes it when the body is set (the body is no longer rewritten).

```diff
diff --git a/rust_core/src/minitid.rs b/rust_core/src/minitid.rs
index 23a4af6..0a2c15d 100644
--- a/rust_core/src/minitid.rs
+++ b/rust_core/src/minitid.rs
@@ -8,14 +8,22 @@
 // How:  axum 0.8 HTTP server → KISS message compressor (Rust-native, no
 //       Python round-trip) → reqwest upstream → SSE stream passthrough.
 //       CC sets ANTHROPIC_BASE_URL=http://127.0.0.1:$MINITID_PORT.
+//
+// [2026-09-21] Claude (Sonnet 4.6) — Correction pass
+// What: Removed the faux-KISS compression path and updated comments.
+// Why:  The header still advertised 60-char first-sentence truncation,
+//       which is LAW 3 shrapnel: a removed behaviour lying about the
+//       current contract.
+// How:  b263286 deleted the rewrite; this pass records that in the
+//       changelog and replaces the obsolete KISS behaviour header with
+//       an honest passthrough description.
 // -------------------
 //
-// KISS behaviour (mirrors kiss_filter.py):
-//   - Warmup: first KISS_WARMUP_TURNS passes through unmodified.
-//   - GOP boundary: every KISS_FORCE_FULL_EVERY turns forces a full pass.
-//   - Otherwise: messages beyond the recent window have their content
-//     truncated to the first sentence (max 60 chars + "…"). Role
-//     structure is preserved, so Anthropic's alternation rule holds.
+// KISS behaviour (current contract):
+//   This June snapshot no longer truncates message content.
+//   /v1/messages forwards the original request bytes unchanged.
+//   KissSession and KISS_* constants are retained as dead-code
+//   scaffolding for a future pass; they do not affect forwarding.
 //
 // Session identity: SHA-256 of the first 100 bytes of the first user
 // message's content string, truncated to 16 hex chars. Stable across
@@ -54,7 +62,7 @@ const KISS_FORCE_FULL_EVERY: u32 = 20;
 const MAX_BODY: usize = 20 * 1_024 * 1_024;
 
 // Hop-by-hop and body-invalidating headers stripped from the forwarded request.
-// content-length is excluded because we rewrite the body (KISS changes byte count).
+// content-length is excluded because reqwest recomputes it when we set the body.
 const DROP_REQ_HEADERS: &[&str] = &[
     "host",
     "content-length",
```

## Git status after commit + push

```
## worktree-demo-reimagining
 M app.py
 M engine.py
 M tests/test_engine.py
 M tests/test_viz.py
 M viz.py
?? libjpeg-turbo-2.1.5/
?? libjpeg-turbo_2.1.5-2ubuntu2.debian.tar.xz
?? libjpeg-turbo_2.1.5-2ubuntu2.dsc
?? libjpeg-turbo_2.1.5.orig.tar.gz
```

Dirty demo Python files (`app.py`, `engine.py`, `viz.py`, `tests/`) remain unstaged and untouched.

## Boundaries observed

- Only `rust_core/src/minitid.rs` edited, staged, committed, and pushed.
- No rebuild or redeploy of live miniTID PID `743492`.
- No worktree removal, no merge, no dispatch, no self-accept.
