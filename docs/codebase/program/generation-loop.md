[Codebase docs](../README.md) › [Program](README.md) › One-shot generation loop

# One-shot generation (prefill, token loop, report)

**Purpose:** what `strata generate` does without `--serve`: read the prompt (batched or token by token), generate
`--max-new` tokens (by the captured per-layer graphs, then optionally by speculative verify windows), print the ids
and rates. This path is the measurement harness the project grew from: Gate C1 compares its `--dump-logits` against
llama.cpp on the same ids.
**Covers:** `src/program/generate.cpp:4216-4302` (head, input, sampler), `:4584-4689` (scratch, token graph),
`:7901-8228` (batched prefill and token loop), `:8662-8843` (report). The speculative continuation (`:8230-8660`) is
on [speculative-decoding.md](speculative-decoding.md).

## Canonical vs native pack

| | Canonical (Q2_0) pack | Native (IQ/GGUF-form) pack |
| --- | --- | --- |
| Per-layer graphs | `session_capture` (`:4171-4180`) | none |
| Token graph | captured when allowed (below) | none |
| Prompt | batched (`--prefill`) or the token loop | batched, required for >1 token |
| Decode | token loop, then verify windows if `--spec > 0` | verify windows only; first window is the last prompt token (`:8010`) |
| Requirements | | `--native`, `--spec >= 2`, `--prefill` (`:2490-2495`) |

Qwen3.6 packs are native packs.

## Token graph and per-layer path (`:4590-4689`)

`SessionLoopScratch` is allocated once (the comment records that per-token allocation used to force a device-wide
sync every token). The **token graph** (`session_capture_token`, one launch for all layers) is captured when nothing
needs a host step between the ring and `post[l]`: captured path, no `--no-token-graph`, no per-layer dumps, the hit
path absent or graph-decided (`thits.on()`), canonical pack, no multi-GPU split. Otherwise `session_loop` runs a
`pre[l]` graph, the CPU pool via the doorbell, then `post[l]`, per layer (the "captured pair" that fixed layer `l`'s
experts being combined with layer `l+1`'s router, LEDGER L100, `:4155-4167`).

## Batched prefill (`:7901-8006`)

When `--prefill > 0` and the prompt has more than one token:

1. Positions batched: `n_batched = n_prompt - 1` (the last prompt token is decoded, its prediction is the first new
   token), or `--prefill-until P` when smaller.
2. Borrowing (`:7914-7945`): with a residency table, `plan_lend(chunk)` chooses the chunk and slot count; the chunk is
   shrunk to what this prompt needs (`request_chunk`, rounded up to 256); with `auto` the chosen chunk is printed. The
   lent slots are marked non-resident and `d_res` re-uploaded. Otherwise the prompt path allocates its own buffers.
3. `Prefill::init(wt, g, ss, srcp, cache, host_res, chunk, main_cs, err, borrow, borrow_bytes)`.
4. With `--mtp`: `mtp.bind(wt, &native_head, nullptr)` and `prefill.on_chunk` feeds each chunk's residual rows and the
   next tokens to the drafter's KV (`Prefill::draft_kv`, else `MtpDrafter::prefill`).
5. `prefill.run(tokens, n_batched, 0, err)`; lent slots are refilled from `srcp` and the table restored; the PLE window
   `ss.ple_prev` is set from the two tokens before `pos_start`; a stats line prints tokens, chunks, ms, tok/s, experts
   streamed/DMA/resident and PLE ms.

## The token loop (`:8008-8228`)

From `pos_start` (0, or `n_batched` after batched prefill). For each position:

1. Native pack: break straight to the speculative loop (`spec_pos = pos`).
2. PLE: `ss.ple_token = tok`, `ple_issue_token` starts the row reads, `put_input` embeds (`embed_row`; position 0 calls
   `session_zero` with the embedding, later positions broadcast it into the `hc` residual streams), then
   `ple_finish_token`.
3. Layers: `session_token` (`--no-capture`), else `doorbell_reset` and `session_run_token` (token graph) or
   `session_loop` (`overlap=true`, the layer-dump stage, the loop scratch). `drive.d.layers` is reset to 0 every
   token - it is the blob's layer index, not a counter (`:8058-8065`).
4. Optional `--dump-final-r` record; pool failure exits with the layer/expert.
5. `run_head` (`lm_head`, or `lm_head_mix` + native head). Without `--stream-token` a `cudaDeviceSynchronize` follows
   so an asynchronous fault is reported here, not at the next token's `embed_row` (`:8114-8123`).
6. Logits are read back when not `--stream-token`, or `--check-logits`, or this row is dumped; every value must be
   finite (a non-finite logit exits 1). Dump rows are selected by `logits_selection::selected(pos, dump_positions,
   stride)` and the header is written with the first row ([diagnostics.md](diagnostics.md)).
7. **Sampling:** `sp.counter = pos` and `sample_tokens(d_logits, 1, n_vocab, nullptr, 0, sp, d_next, stream)`: the
   draw is Philox(seed, position), the same as a verify window's row so a seed gives the same text on either path
   (`:8169-8173`). The sampled id is range-checked; `--native-flash-attn-short` checks each attention layer's status.
8. Timing: decode positions (`pos >= n_prompt - 1`) add to `total_ms`, earlier ones to `prefill_ms`; TTFT is taken at
   the last prompt position. Six host phases (`ms_ple`, `ms_embed`, `ms_layers`, `ms_head`, `ms_readback`,
   `ms_sample`) accumulate for `--stats`.
9. From `n_prompt - 1` on the pick goes into `produced`; stop at `--max-new` or (with `--stop-eos`) an EOS id.
   **Teacher forcing over the prompt:** while the prompt lasts the next input is the prompt's token, not the pick.
   The PLE window shifts (`ple_prev`).
10. With `--spec > 0`, after the first generated token the loop hands over to the speculative loop at
    `spec_pos = pos + 1` (`:8227`).

## Report (`:8662-8843`)

Closes dump files (and prints what was written), prints `prompt  :` and `output  :` id lists, `decode N tokens in X
ms -> tok/s` (wall clock, everything from PLE advance to sampled id, A7), `prefill ... (time to first token ...)`,
`--dump-mixed` and `--dump-residual` files with magnitude summaries. `--stats` adds per-token ms, the six host
phases, PLE I/O report, CPU pool ms per position (denominator = positions the pool ran, `drive.calls / n_layers`, A6),
pool phase split (wait-park / drain / re-park), blob reads, helper-GPU counts, R4 overlap and hit rate
(`hits / (hits + admitted + refused)`), token-graph wait/pool, ring-latency and host-after-ring figures.

## Configuration

Flags: `--tokens`, `--tokens-file`, `--max-new`, `--max-context`, sampling flags, `--prefill`, `--prefill-until`,
`--no-prefill-borrow`, `--stream-token`, `--check-logits`, `--no-capture`, `--no-pool`, `--no-token-graph`,
`--stop-eos`, `--eos-ids`, `--stats`, the dump flags. Env: `STRATA_PREFILL_LEND_PCT`, `STRATA_REFILL_BLOCKING`.

## Gotchas recorded in code comments

- `--no-capture` runs `session_token`, which has no CPU pool hook; it is refused unless `--no-pool` (`:4182-4197`).
- The logits header row count is `n_prompt - 1 + max_new` (corrected once, `tools/logits_identical.py` caught it).
- The first generated token is the prediction at the last **prompt** position; sampling at every position and recording
  from `n_prompt - 1` avoids a special case (`:8211-8216`).
- Two device-wide syncs happen per sampled token (`sample_tokens(nullptr)` and the 4-byte read; `:8198-8200`).

## See also

- [speculative-decoding.md](speculative-decoding.md), [diagnostics.md](diagnostics.md), [expert-tiers.md](expert-tiers.md)
- [../core/README.md](../core/README.md) (session graphs, token graph), [../prefill/README.md](../prefill/README.md),
  [../kernels/README.md](../kernels/README.md) (sampler, PLE)
