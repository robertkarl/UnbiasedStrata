# Qwen3.6 port, step 5, first part: serving on the 4090, and what broke below full residency

Branch `qwen36-step4`, 2026-10-13, VM 221 (RTX 4090, 24 GB; the user handed the box to this work at 02:05 UTC and the
Flash-Next service on it was stopped).  Tree `/models/work/qwen36-step4`, binary `build/unbiased-strata`, logs under
`/models/work/serve/`.  The 1080 Ti (VM 211) is off limits: another session's service runs there.

## The serve command

```
./build/unbiased-strata --model /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --pack /models/work/packs/qwen36-q4 \
    --mtp /models/work/packs/qwen36-mtp-draft-full --ctx 262144 --threads 8 --host 127.0.0.1 --port 18090 -- --kv q4_0
```

`--mtp` takes a Qwen3.6 draft pack (`tools/mtp_qwen36_pack.py`); `qwen36-mtp-draft` carries a `draft_vocab.bin`
subset, `-full` has none.  Everything after `--` goes to the engine: `--vram-cap MiB`, `--ram-cap GiB`, `--kv
fp16|q4_0|int8`, `--expert-cache N|auto`, `--pcie-frac F`, `--prompt-cache N`, `--vram-reserve-mib`.

## Faults found in order, all fixed on the branch

| # | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| 1 | `unbiased-strata: missing file: .../qwen36-mtp-draft-full/draft_vocab.bin` | the front end required the Flash-Next draft vocabulary subset | optional in the front end as it already was in the engine (3a806b5) |
| 2 | first request: 220 tokens generated, then `saving a conversation checkpoint failed` (`conversation snapshot: invalid model geometry`, then `invalid indexer running-state target`) | the prompt-cache checkpoint and the snapshot rejected a geometry with no indexer (idx heads / key dim 0) and no hyper-connection rate, and demanded the indexer's tail / dead / block-position / pooled buffers | zeros allowed where `has_indexer()` is false and `hc` is 1, no indexer buffers or block positions for dense attention, pooled rows 0 (3a806b5).  Two-turn conversation: 1 then 2 checkpoints, prefix reused |
| 3 | Q4 at 262,144: `2, 3! 5, 7, 11, !!!!!!...` (token 0 forever); IQ3 under any `--vram-cap` the same; both files fine at 8K / 32K; IQ3 at 256K fine | **the pool's native-pack phases iterated the canonical Q2_0 widths (FF 640, H 2560) instead of the layer's (512, 2048)**: every expert miss computed on the CPU wrote 512 rows past its output row.  The runner's and drafter's tests all ran with every expert resident, so the miss path had never executed on Qwen3.6 | `run_split_multi_native` at `NativeFmt::n_ff / n_embd`; `native_fmt` refuses wider layers; `pool_native_widths_test` (22d72e6) |

How fault 3 was isolated (each row one serve, the prime-numbers prompt, greedy, `bangs` = count of `!` in 120-200
tokens):

| Run | file | ctx | KV | slots resident | misses on | result |
| --- | --- | --- | --- | --- | --- | --- |
| e1 | Q4 | 8192 | fp16 | 10,240 / 10,240 | - | correct, 362 tok/s |
| e3 | Q4 | 262,144 | q4_0 | 9,980 | CPU + DMA | `2, 3, 5, 7, !!!` |
| e2 | Q4 | 262,144 | q4_0, `--vram-cap 20480` | 8,132 | CPU + DMA | `!!!` from the first token |
| e4 | Q4 | 262,144 | fp16 | 8,120 | CPU + DMA | `!!!` |
| e5 / e6 | Q4 | 8192 / 32,768 | q4_0 | 10,240 | - | correct, 354 / 356 tok/s |
| t1 | IQ3 | 262,144 | q4_0, `--vram-cap 15360` | 10,212 | CPU + DMA | `!!!` |
| t2 | Q4 | 262,144 | q4_0, `--vram-reserve-mib 1200` | 9,838 | CPU + DMA | `!!!` (so not VRAM pressure) |
| t3 | Q4 | 8192 | fp16, `--expert-cache 8000` | 9,587 | CPU + DMA | `!!!` (the cheap reproduction) |
| t4 | Q4 | 8192 | as t3, `--pcie-frac 1.0` | 9,587 | DMA only (GPU computes) | **correct**, 305 tok/s |
| t5 | Q4 | 8192 | as t3, `--pcie-frac 0.0` | 9,587 | CPU only | `!!!` |
| t6 / t7 / t8 | Q4 ggml dots only / IQ3 / one worker | 8192 | as t5 | 9,587 | CPU only | `!!!` all three: not the kernels |
| fix | Q4 | 8192 | as t5 | 9,587 | CPU only | correct, 328 tok/s |
| fix | Q4 | 262,144 | q4_0 | 9,980 | CPU + DMA | correct, 322 tok/s |
| fix | IQ3 | 262,144 | q4_0, `--vram-cap 11264` | 5,762 (56%) | CPU + DMA | correct, 169 tok/s; 63,526 VRAM hits of 73,553 lookups |

| 4 | Pascal host (VM 211, step5-1080ti, REGINA-106): with CPU-computed misses iq3-test20 51/64 top-1, KL 0.84; Q4 52-54/64, KL 0.25; DMA misses correct; deterministic; `--spec 1` changes it | **the multi-token CPU dispatch addressed token t's activation at t x 2560 and its output row at i x 2560** (the canonical widths pulled in by `using namespace`), while the window publishes at the layer's 2048: every token after a group's first used another token's activation.  Named by the Codex review of 22d72e6; the 4090's serve looked right because its text was checked, not its logits | the dispatch at the layer's width with refusals (the commit after 22d72e6); after it on VM 211 all three IQ3 prompts 64/64 (KL 1.4e-3 / 4.8e-3 / 2.0e-3), q4-test20 62/64 KL 6e-3, q4-prime 64/64 KL 8e-4, CPU misses forced.  The ladder points taken with the old binary are rerun |
| 5 | the 4090 sweep's `--ram-cap 12` points: "ArenaExpertSource: the expert arena needs 18.32 GiB ... memory limit is 12.00 GiB ... --mmap-experts" | the plan chose RAM copies but the arena source still pinned the whole file | the engine picks the mapped tier with profile-ranked RAM copies itself when the arena exceeds the limit or available RAM (same commit) |
| 6 | planner (Codex review): a full-head draft pack charged 515 MiB the drafter never allocates; k8v4 undercounted the drafter's INT8 K/V by 60 MiB; a lowered context was printed but not applied | | all three in the same commit; multi-GPU helpers refused for this geometry |

| 7 | second Codex review of fa69e82: the front end kept validating and advertising the requested `--ctx` when the engine's plan lowered it; the automatic mapped tier ignored an explicit `--shared-expert-arena` and the multi-GPU residency rules; the plan's session term keeps the RoPE table at the requested context | | the front end takes the context from the engine's `READY <n>` line and bounds requests by it; the automatic choice is skipped with a shared arena or the helpers.  The RoPE table at a lowered context is left as a conservative overcount (at most 32 MiB from 262,144 to 131,072), noted here rather than threaded through the plan |

## What the 4090 serves now

| file | ctx | fixed set | experts resident | VRAM free after load | short answer decode | drafts accepted |
| --- | --- | --- | --- | --- | --- | --- |
| IQ3_XXS, `-full` draft | 262,144 | 4,331 MiB (weights 1,596, drafter 831, KV 1,440, session 162, reserve 300) | 10,240 / 10,240 (10.35 GiB) | 8,758 MiB | 319-359 tok/s | 157 / 181, 223 / 237 |
| Q4_K_XL, subset draft | 262,144 | 4,823 MiB (weights 2,038, drafter 882, KV 1,440, session 162, reserve 300) | 9,980 / 10,240 | 434 MiB | 322 tok/s | 149 / 152 |

Also checked through the HTTP front end: streaming (SSE chunks, `reasoning_content` deltas), a two-turn
conversation with checkpoint reuse, a tool call (`<tool_call><function=read>` emitted; the earlier "!!!" inside its
arguments was fault 3).  The response's `model` field is now the GGUF's base name (was the Flash-Next constant).

## Observations for the measurements

- The RAM tier reports `expert copies in RAM: 0` on this host: with 62 GB the whole expert arena is pinned host
  memory (`cudaHostRegister PORTABLE`, 18.3 GiB for Q4), and a miss is served from it by DMA or the CPU; `ram_reads`
  and `file_reads` on `/health` stay 0.  On a 12 GiB host the arena cannot hold the file and the mapped-from-disk
  tier is what the ladder measures (`--ram-cap` emulates it here).
- Checkpoint reuse: the second turn reused 27 of 278 prompt tokens.  Cause and the opt-in that lifts it: the A5
  section below, check (d).
- `expert cache ... policy is` prints an empty policy name for native packs (cosmetic).
- At 262,144 the Q4 fixed set leaves 19.3 GiB for experts; 9,980 slots fit; the planner's "10240 of 10240 (100%)"
  line and the cache's "9980 slots" disagree by the 700 MiB runtime reserve the cache keeps; the plan line should
  say what the cache will actually hold.

## A5 on the 4090: the serving checks, after the streaming fix (REGINA-110)

**The fault.**  With `"stream": true` the answer arrived as one `content` delta before the final chunk; only
`reasoning_content` streamed.  Not the parser: `run()` in `frontend/main.cpp` cleared the content of every partial
parse together with its tool calls, a rule that came with 46c46ec so that partial tool-call JSON (closed and
normalized by the parser on each token) could never change text already sent.  That holds for a call's arguments,
not for the answer.  c3b41cc: reasoning and content are published as they arrive, tool calls once from the complete
parse (`StreamParse`); a partial parse whose content does not extend what was sent publishes no content until it
does; a non-streaming request is parsed once instead of after every token.  **A tool call still arrives whole**, in
one delta (name, id and full arguments) when the message ends; the text before a call streams.
`make test-chat-parser` now feeds seven responses a character at a time (reasoning on and off, tools declared and
not called, text before a call, one and two calls) and requires the pieces to add up to the whole parse, the answer
to arrive in many deltas and each call in one; it passes on the IQ3_XXS and Q4_K_XL files.  The ticket's request on
the final binary: 200 content chunks of 203 (was 1 of 3).

**The checks.**  VM 221, binary sha256 `46c5747f23dcbdeb` (c3b41cc), `--mtp qwen36-mtp-draft --ctx 262144 --threads
8 ... -- --kv q4_0`, greedy, client and server on the VM (`/models/work/serve/a5-20261013/`: `a5.py`, `serve.sh`,
one `.json` and `.log` per configuration).  All nine checks pass in all three configurations.

| check | Q4_K_XL, no cap (9,980 slots) | Q4_K_XL, `--vram-cap 11264` (2,910 slots) | IQ3_XXS, `--vram-cap 6144` (809 slots) |
| --- | --- | --- | --- |
| (a) chat, not streamed, thinking off: "capital of France" | pass, 0.15 s | pass, 0.54 s | pass, 0.54 s |
| (a) chat, not streamed, thinking on: 17 x 23 = 391 | pass, 314 tokens at 313 tok/s | pass, 314 at 76 | pass, 410 at 70 |
| (b) stream, thinking on: reasoning deltas, first, largest gap | 1,087, 0.081 s, 14.1 ms | 467, 0.293 s, 92.8 ms | 418, 0.348 s, 78.6 ms |
| (b) the same request: content deltas, first, largest gap | 89, 3.616 s, 13.8 ms | 82, 5.803 s, 61.6 ms | 53, 6.181 s, 82.3 ms |
| (b) the same request: tokens, decode, largest gap between any two deltas | 1,179 at 303 tok/s, 14.1 ms | 552 at 83, 92.8 ms | 474 at 72, 82.3 ms |
| (b) stream, thinking off (the ticket's request): content deltas, first, largest gap | 212 of 213 tokens, 0.076 s, 11.5 ms | 203 of 204, 0.330 s, 53.8 ms | 209 of 210, 0.345 s, 64.2 ms |
| (c) tool round trip, not streamed: call, result, answer | pass: `get_weather {"city":"Paris"}`, `tool_calls`, then "17 C with light rain", `stop` | pass | pass |
| (c) tool round trip, streamed | pass: reasoning 658 deltas, the call in 1 delta at 2.40 s; the answer in 14 content deltas | pass: 517 reasoning deltas, call in 1 at 5.40 s; 15 content deltas | pass: 173, call in 1 at 2.86 s; 18 content deltas |
| (c) tools declared, not called, streamed, thinking on: content deltas, first, largest gap | 73, 2.597 s, 14.0 ms | 74, 10.318 s, 62.2 ms | 81, 12.442 s, 81.1 ms |
| (c) the same, thinking off | 76, 0.032 s, 11.4 ms | 69, 0.067 s, 41.7 ms | 66, 0.093 s, 41.9 ms |
| (d) three-turn chat, tokens reused of the prompt per turn | 0 / 23, 18 / 129, 124 / 172 | 0 / 23, 18 / 133, 128 / 176 | 0 / 23, 18 / 178, 173 / 246 |

"First" is from the request to that delta, so with thinking on the first content delta comes after the reasoning.
Deltas come in bursts of an accepted draft window (median gap 0.05-0.1 ms); the largest gap is one verify round,
longer under a cap where it waits on expert misses.  No keepalive comment was needed in any run.

**(d) Why the reused prefix is short, and what lifts it.**  The engine keeps a checkpoint at the last
`<|im_start|>` of each prompt and one at the end of each reply.  The next request reuses the second only if it
renders the reply as it was generated, and in a plain chat Qwen3.6's template does not: a reply before the last
user message is rendered without its `<think>` block (with thinking off, without the empty `<think>\n\n</think>`
the generation prompt carried), whether or not the client sends `reasoning_content` back.  The tokens leave the
generated ones right after the assistant header, and the reuse is the previous prompt less its 5-token header: 18
of 129, 124 of 172, and the 27 of 278 above.  Each turn rereads the last reply and the new message, never the older
history.  Not an engine fault.  Two cases reuse the reply too:

- a tool loop: a reply after the last user message keeps its reasoning in the template.  pi's three requests (check
  e): 0 of 1,604, then 1,867 of 1,948 and 2,047 of 2,118;
- `"chat_template_kwargs": {"preserve_thinking": true}` with `reasoning_content` sent back, which the front end now
  passes to the template (3aa9eed; binary `e4d8ec0c07473e13`): four turns reuse 0 / 22, 551 / 569, 1,010 / 1,027,
  1,365 / 1,382.  Opt-in, because the earlier reasoning then stays in the context the model reads.  Not every turn
  hits: in two of eight such turns the re-rendered reply did not match the generated tokens (578 of 1,330; 18 of
  1,138) and the request fell back to the header checkpoint; which replies re-tokenize differently was not traced.

**(e) pi.**  pi 0.85.1 on the Mac, its own agent directory (`PI_CODING_AGENT_DIR`, the user's `~/.pi` untouched),
provider `openai-completions` at the engine through an SSH tunnel, Q4_K_XL no cap, `pi -p --mode json`: it wrote
`primes.py` with its `write` tool, ran it with `bash`, and reported `2 3 5 ... 71`.  Three requests, two tool calls,
80 thinking deltas and 82 text deltas seen by pi; nothing dropped by the parser.

**Tests.**  The Qwen3.6 ctest subset on the free card (`ctest -R "qwen36|vram_plan_test|pool_native_widths_test"`):
21 of 21.  `make test-frontend` and `make test-chat-parser` pass.  The engine is untouched by these two commits, so
no logits comparison was owed.

**Seen, then chased.**  The same greedy request does not always generate the same tokens on one server (check b,
thinking on, Q4 no cap: 632 tokens in one pass, 1,179 in the next).  Traced in the next section: not a fault.
Images (mmproj) were out of scope.

## The same greedy request, different tokens: which paths differ, and by how much

2026-10-13, VM 221, binary `11da5388afdfabcb`.  The engine was driven on its own stdin protocol (`unbiased-strata
--engine --serve ...` with the front end's arguments, `GEN <max_new> <ids>`), so the record is token ids.  Two
prompts from A5, rendered by the front end: `think` (26 tokens, "What is 2+2? Answer, then write three sentences
about addition.", thinking on) and `nothink` (25 tokens, the lighthouse request, thinking off), `max_new` 3,000.
Material in `/models/work/serve/nd-20261013/` (`nd.py` the driver, `cmp.py`, `trdiff.py`, `acc.py`, one `.jsonl` of
token ids and one engine log per run, 23 GB of logit dumps).

**Verdict: benign.**  No state leaks between requests.  The output is a pure function of the request and of which
of three numerically different, individually correct paths each number went through; two things move a request
between those paths from one pass to the next:

1. **Where an expert is computed.**  A resident expert and one read over PCIe (DMA) run the GPU kernel (llama.cpp's
   CUDA arithmetic: activations as Q8_1); a miss computed on the CPU runs ggml's CPU kernel (activations as Q8_K).
   The two round differently, as llama.cpp's own CPU and CUDA back ends do.  The adaptive tier (`--adapt-every`, 4
   rounds) swaps routed-but-missing experts into VRAM and keeps them across requests, so the second pass of a
   request finds experts on the card that the first computed on the CPU.  It settles: once the cache stops
   changing the output stops changing.
2. **How the prompt's state was built.**  A part of at most `--short-read` (64) tokens and everything a checkpoint
   or the live session supplies went through the verify windows; a longer part read from token 0 goes through the
   batched prompt path, which is different arithmetic (step4-runner.md: the two agree to 2.1e-2 on the fixture).
   Restoring a checkpoint is exact; what differs is a state made by decode against the same state re-read in a batch.

With misses present a third input reaches the output through (1): the lookup drafter's policy picks window sizes
from measured round times, the adaptive tier counts rounds, and the PCIe share of a layer's misses is a fraction
of the misses in that window.  With `--suffix-draft 0` a fresh engine gave the same tokens with and without the
tracing below (which slows every window); with the default it gave 557, 554 and 555 tokens for `think`.  With
every expert resident the window shape does not reach the output (the window is bitwise independent of its shape).

**Which configurations repeat.**  Ten requests of each prompt, alternating, on one engine unless noted:

| configuration | misses run on | distinct outputs, `think` / `nothink` | tokens generated |
| --- | --- | --- | --- |
| IQ3_XXS, 262,144, q4_0, no cap (10,240 resident) | - | 1 / 1 | 480 / 197 |
| the same, `think` alone 10 times (21 of 26 reused) and with `--prompt-cache 0` | - | 1, the same tokens | 480 |
| Q4_K_XL, 8,192, fp16 (10,240 resident) | - | 1 / 1 | 664 / 218 |
| Q4, 8,192, `--expert-cache 8000 --pcie-frac 1.0`, adaptive tier on or off | GPU (DMA) | 1 / 1, **the all-resident tokens** | 664 / 218 |
| Q4, 8,192, `--expert-cache 8000 --pcie-frac 0.0 --adapt-every 0` | CPU, fixed set | 1 / 1 | 632 / 191 |
| the same with the adaptive tier | CPU, changing set | 7 / 6, the same from the 7th request | 511, 557, 639, 633, 1,451, 478, 479 x 4 |
| Q4, 262,144, q4_0, no cap (9,980 slots): the A5 configuration | CPU (0.05% of lookups) | 3 / 3, the same from the 3rd | 557, 620, 553 x 8 / 221, 213, 214 x 8 |
| the same, `--prompt-cache 0` | CPU | the tokens of the row above | |
| the same, `--adapt-every 0` | CPU, fixed set | 1 / 1 | 558 / 211 |
| the same, `--suffix-draft 0` | CPU | 2 / 3 | 550, 551 x 9 |
| the same, `--mtp-max-t 1 --suffix-draft 0` (6 requests) | CPU | 4 / 2 | 561, 552, 1,014, 552 x 3 |
| the same, a new engine for every request | CPU | 1 / 1 | 557 / 196 |

Every configuration with one path is bit-repeatable; the only ones that are not have CPU misses and a cache that
is still changing.  The checkpoint cache, the drafter and the lookup drafter are not sources.  The second cause,
with every expert resident (IQ3 no cap; Q4 at 8,192), one engine against a new engine reading from token 0:

| request | state from | tokens | first difference |
| --- | --- | --- | --- |
| 258-token prompt, three times | 0 reused, then 253 reused + 5 | identical, and identical to a new engine | - |
| `think`, its 480 / 664 generated tokens and a second user turn (529 / 713) | the live session (505 / 690 reused) or the header checkpoint (524 / 708) | identical to each other | |
| the same on a new engine | the batched prompt path, all of it | IQ3 554 against 495, Q4 437 against 401 | token 191 / 140 |
| a 73-token second turn | 21 reused + 52 through the windows, or 68 reused | identical to each other | |
| the same on a new engine | 73 read from 0 | IQ3 890 against 494, Q4 465 against 438 | token 61 / 61 |

**The margins where they part, and the accuracy of every state.**  `STRATA_SERVE_LOGIT_TRACE=FILE` (new, off by
default) writes the head's five largest logits, the window and its misses per emitted token, and
`STRATA_SERVE_LOGIT_DUMP=PREFIX` the whole row (the last window's rows past the stop token included);
`tools/qwen36_teacher_llama.cpp` grew `--prompt-ids` and `--force-ids`, so llama.cpp (F16 K/V, the 4090) is
teacher-forced along each run's own tokens and every position of every run is scored, with the path that run took.  Margins are logit differences between the two tokens chosen.

| first difference | this pass | the other pass | llama.cpp |
| --- | --- | --- | --- |
| Q4 no cap, `think`, request 1 against 2, token 249 | 0.03 for A | 0.23 for B | 0.16 for B |
| request 1 against 3, token 245 | 0.10 for A | 0.78 for B | 0.09 for B |
| request 3 against 4, token 531 | 0.007 for A | 0.20 for B | 0.31 for B (its own pick is a third token, 0.12 above B) |
| IQ3 no cap, session state against a batched re-read, token 191 | 0.27 | 0.34 | 0.03 |
| Q4 8,192, the same, token 140 | 0.08 | 0.03 | 0.012 |
| IQ3 / Q4 second turn, windows against batched, token 61 | 0.09 / 0.11 | 0.37 / 0.08 | 0.32 / 0.03 |

Over the common prefix two passes differ by a mean of 0.12-0.13 logits on the reference's 20 largest (KL 1.4e-3
to 2.1e-3 between them), the size of each pass's distance from llama.cpp.  Every state against llama.cpp, the
whole run and its first 64 positions (the rule: 62 of 64, mean KL below 1e-2):

| run, path | positions | top-1 | mean KL | max KL | first 64: top-1, mean KL |
| --- | --- | --- | --- | --- | --- |
| Q4 8,192 fp16, all resident (serve) | 664 | 660 (99.4%) | 8.3e-4 | 0.077 | 64, 5.6e-4 |
| Q4 8,192, CPU misses, `--expert-cache 8000`, fixed (0.7% of lookups) | 632 | 624 (98.7%) | 8.9e-4 | 0.053 | 64, 5.5e-4 |
| Q4 8,192, CPU misses, `--expert-cache 4000`, fixed (33% of lookups) | 547 | 542 (99.1%) | 8.9e-4 | 0.046 | 64, 1.3e-3 |
| Q4 no cap 262,144 q4_0, requests 1 / 2 / 3 / 4 of `think` | 555 / 630 / 557 / 553 | 98.4 / 98.4 / 98.7 / 98.6% | 3.4e-3 / 3.3e-3 / 4.0e-3 / 3.8e-3 | 0.08 / 0.30 / 0.16 / 0.13 | 63, 1.8e-3 to 2.1e-3 |
| IQ3 no cap 262,144 q4_0, second turn: session state / batched re-read | 495 / 554 | 98.8 / 99.3% | 3.4e-3 / 4.7e-3 | 0.11 / 0.10 | 64, 1.4e-3 / 64, 1.0e-3 |
| IQ3, 73-token second turn: windows / batched | 494 / 890 | 98.2 / 98.5% | 4.1e-3 / 4.0e-3 | 0.15 / 0.14 | 62, 3.5e-3 / 63, 1.4e-3 |
| Q4 8,192, second turn: session state / batched re-read | 401 / 437 | 99.8 / 99.3% | 9.4e-4 / 8.1e-4 | 0.023 / 0.033 | 64, 2.5e-4 / 64, 2.0e-4 |
| Q4 8,192, 73-token second turn: windows / batched | 438 / 465 | 99.3 / 99.4% | 9.0e-4 / 1.5e-3 | 0.021 / 0.12 | 64, 8.3e-4 / 63, 6.2e-4 |
| `strata --teacher-force`, 400 tokens of the all-resident run: resident / DMA / CPU 8000 / CPU 4000 / CPU 4000 fixed | 400 | 393 / 393 / 396 / 396 / 395 | 9.2e-4 / 9.2e-4 / 9.6e-4 / 7.9e-4 / 8.8e-4 | 0.028 to 0.036 | 63 or 64, 3.6e-4 to 5.9e-4 |

Every top-1 disagreement with llama.cpp in these runs (113 in 9,912 positions) sits where llama.cpp's own margin is
under 0.75 logits; one position in 20 to 25 has a top-2 gap under 0.5, so a greedy run of 500 tokens meets some
twenty places where a pass can turn.  The teacher-forced resident and DMA dumps are the same bytes (`cmp`).  No
rising drift: Spearman(position, KL) is between 0.00 and +0.13 in every run but the IQ3 batched re-read (+0.20, KL
3.2e-3 in its first half and 6.1e-3 in its second).
The 262,144 rows carry the q4_0 K/V against llama.cpp's F16 (mean KL 3-5e-3 where the fp16 rows have 8e-4 to
1.5e-3); the K/V format is the likely reason (not isolated here), and it is in every pass alike.

**For a run that must repeat.**  `--pcie-frac 1.0` (every miss on the GPU: the all-resident tokens whatever the
cache holds, at the DMA path's cost; only while no layer of a window has more than 16 distinct misses, past which the
rest run on the CPU, see the REGINA-111 section at the end), or `--adapt-every 0 --suffix-draft 0` (the miss set fixed, no timed policy).
A conversation that continues on one engine still differs from the same prompt read from token 0 on another.

**Not checked.**  The 1080 Ti (VM 211, off limits); a capped Q4 or IQ3 with thousands of misses per request beyond
the `--expert-cache 4000` rows; sampled (temperature above 0) requests; requests cancelled mid-generation followed
by the same prompt; the q4_0 K/V's own share of the 262,144 rows' KL.  `--teacher-force` with `--mtp` committed
the drafter's guesses as if they were the oracle's tokens (70 of 400 top-1 on a run that scores 394 without it): it
is now refused.

## A5 on the 1080 Ti: the serving checks at the branch head

2026-10-13, VM 211 (GTX 1080 Ti 11,264 MiB, sm_61, 14 vCPU, 52 GB; the box was free), tree at 542b53b (the streaming
fix c3b41cc, `preserve_thinking` 3aa9eed, the logit diagnostics c7825b4), built by `/home/rk/build-step4.sh`:
`unbiased-strata` sha256 `4e161358c8fd20c1`, `strata` `a44d896ccb25ff98`.  The ladder on this card was measured with
7ebc9e5 (`e6c8bbf4cfb16e09`); the engine source differs from it by the diagnostics and the `--teacher-force --mtp`
refusal only (`generate.cpp`, 52 lines).  Material: `/home/rk/bench/a5-1080ti-20261013/` (`logits/`, one `.json`,
`.log`, `.summ.txt` per configuration, `speed/`), drivers `/home/rk/bench/a5-1080ti-job1a.sh` and `job1b.sh` there.

**Tests.**  `ctest -R "qwen36|vram_plan_test|pool_native_widths_test"` on the free card: 21 of 21 (668 s).

**The accuracy rule on this binary** (teacher-forced logits against llama.cpp's, `tools/qwen36_compare_logits.py`,
fixtures `/home/rk/qwen36-step4-teacher`, the command of the REGINA-108 check: context 8,192, `--kv fp16`, `--spec 4
--suffix-draft 0`, 64 positions; pass = top-1 at least 62 and mean KL below 1e-2).  Every miss forced onto one path:
`--pcie-frac 0.0` the CPU pool, `1.0` DMA with the GPU computing.  `auto` holds 8,061 of 10,240 slots for IQ3 and
4,240 for Q4 on this card; `--expert-cache 2000` holds 2,263 and 2,399.  Cells are top-1 of 64, mean KL; **bold**
misses the rule.

| fixture | auto, CPU misses | auto, DMA | 2000, CPU misses | 2000, DMA | the 4090, all resident |
| --- | --- | --- | --- | --- | --- |
| iq3-test20 | 64, 1.3e-3 | 63, 9.1e-4 | 64, 1.4e-3 | 64, 8.0e-4 | 64, 1.1e-3 |
| iq3-task-prime-add-test | 64, 4.5e-3 | **64, 1.26e-2** | 64, 4.8e-3 | **63, 1.18e-2** | 64, 6.6e-3 |
| iq3-task-psim-collide | 64, 2.3e-3 | 64, 2.1e-3 | 64, 2.0e-3 | 64, 2.0e-3 | 64, 1.9e-3 |
| q4-test20 | 62, 6.0e-3 | **61**, 8.5e-3 | **61**, 8.3e-3 | **62, 1.17e-2** | 61, 7.4e-3 |
| q4-task-prime-add-test | 64, 8.1e-4 | 64, 1.2e-3 | 64, 1.1e-3 | 64, 7.2e-4 | 64, 1.7e-3 |
| q4-task-psim-collide | 64, 1.3e-3 | 64, 7.0e-4 | 64, 1.3e-3 | 64, 7.7e-4 | 64, 4.2e-4 |

**19 of 24 runs pass; 5 do not, all on two fixtures** (REGINA-111).  Not a regression of this binary, and not broad:

- the CPU column at `auto` is the REGINA-108 post-fix table again (1.4e-3 / 4.8e-3 / 2.0e-3, q4-test20 62 at 6e-3,
  q4-prime-add 8e-4), and q4-test20 `auto` DMA has the pre-fix binary's figures to the last printed digit (61, 0.00845,
  0.13765 at position 8): neither path moved;
- each miss is one position.  q4-test20: positions 8, 9, 13, the near ties where the 4090 with every expert resident
  also reads 61 of 64 (KL 0.147 at position 8; here 0.086 to 0.323 in the three runs that miss).  iq3-prime-add: position 40, 0.152 on the 4090,
  0.416 / 0.467 on this card's GPU path, under 0.05 with CPU misses at `auto`; the median KL of those runs is 3e-5;
- this card against the 4090's all-resident dumps (`eng/*.f32`): mean KL 5.2e-3 to 1.2e-2 on these two fixtures, 5e-4
  to 2.7e-3 on the other four, the size of either engine's distance from llama.cpp;
- the four DMA runs repeated: byte-identical dumps.

Open in the ticket: on the 4090 the resident and DMA dumps were the same bytes; here `auto` DMA and `2000` DMA differ
from each other although the GPU computes every miss (the adaptive tier was on: 139 to 384 experts swapped in during the 64
tokens), and no all-resident run fits this card at fp16 to compare against.  Closed in the REGINA-111 section at the end: the
`2000, DMA` column is not a GPU-only path (its logs read `CPU experts` 0.22 to 0.73 per layer), and the five misses are
the reference's own spread.

**The checks.**  `a5.py` of the 4090 section unchanged, the measured ladder's launch (`--mtp qwen36-mtp-draft-full
--ctx 262144 --threads 8 ... -- --kv q4_0`; the 4090 table used the subset head), greedy, client and server on the VM.
All nine checks pass in all three configurations.

| check | Q4_K_XL, no cap (3,141 slots) | Q4_K_XL, `--vram-cap 6144` (397 slots) | IQ3_XXS, `--vram-cap 6144` (1,258 slots) |
| --- | --- | --- | --- |
| (a) chat, not streamed, thinking off: "capital of France" | pass, 0.61 s | pass, 0.74 s | pass, 0.77 s |
| (a) chat, not streamed, thinking on: 17 x 23 = 391 | pass, 314 tokens at 63 tok/s | pass, 321 at 43 | pass, 412 at 53 |
| (b) stream, thinking on: reasoning deltas, first, largest gap | 520, 0.345 s, 108.6 ms | 466, 0.528 s, 131.9 ms | 420, 0.491 s, 105.1 ms |
| (b) the same request: content deltas, first, largest gap | 89, 8.328 s, 82.6 ms | 89, 11.604 s, 142.8 ms | 55, 8.323 s, 108.1 ms |
| (b) the same request: tokens, decode, largest gap between any two deltas | 612 at 63 tok/s, 108.6 ms | 558 at 41, 142.8 ms | 478 at 54, 108.1 ms |
| (b) stream, thinking off (the ticket's request): content deltas, first, largest gap | 207 of 208 tokens, 0.364 s, 73.0 ms | 213 of 214, 0.515 s, 83.2 ms | 204 of 205, 0.471 s, 84.1 ms |
| (c) tool round trip, not streamed: call, result, answer | pass: `get_weather {"city":"Paris"}`, `tool_calls`, then "17°C with light rain", `stop` | pass | pass |
| (c) tool round trip, streamed | pass: reasoning 585 deltas, the call in 1 delta at 9.28 s; the answer in 14 content deltas | pass: 546 reasoning deltas, call in 1 at 13.72 s; 14 content deltas | pass: 169, call in 1 at 3.73 s; 18 content deltas |
| (c) tools declared, not called, streamed, thinking on: content deltas, first, largest gap | 73, 11.704 s, 101.4 ms | 83, 23.747 s, 135.1 ms | 80, 14.626 s, 71.4 ms |
| (c) the same, thinking off | 71, 0.102 s, 68.4 ms | 76, 0.164 s, 85.5 ms | 69, 0.140 s, 79.2 ms |
| (d) three-turn chat, tokens reused of the prompt per turn | 0 / 23, 18 / 111, 106 / 154 | 0 / 23, 18 / 143, 138 / 184 | 0 / 23, 18 / 165, 160 / 221 |

The largest gap between any two deltas in any stream is 142.8 ms, a hundredth of the 15 s keepalive; no keepalive
comment was sent in any run.  The longest silence a client sees is the wait for the first content delta with thinking
on, during which reasoning deltas arrive: 23.7 s at Q4 `--vram-cap 6144` (974 tokens at 43 tok/s).  A streamed tool
call still arrives whole, 9.3 to 13.7 s after the request for Q4, with reasoning deltas before it.  `/health` after
each run: `ram_reads` and `file_reads` 0 (the arena is pinned), VRAM hits per lookup 78% / 29% / 51%.  VRAM after
load 10,615 / 10,617 / 10,603 MiB (the cap holds the rest of the card back itself), 10,747 / 10,749 / 10,735 at the end.

(d) as on the 4090: the previous prompt less its 5-token header.  `reuse.py` on Q4 no cap: with
`preserve_thinking` and `reasoning_content` sent back, 942 of 961 and 1,267 of 1,285; without it 18 of 127 and 122 of
170; thinking off 18 of 236 and 229 of 294.

**Decode speed with this binary** (`sweep_engine_ladder.py --host vm211 --threads 8 --no-depth --no-stability`, the
ladder's method: cache dropped, fresh launch, regina twice and the short request; `speed/`):

| | regina rep 0 / rep 1, decode tok/s | short, decode tok/s | prefill 3.1K / 203 tokens, tok/s | drafts accepted (regina rep 1) | the ladder with 7ebc9e5: regina rep 1 / short |
| --- | --- | --- | --- | --- | --- |
| Q4_K_XL, full card | 66.2 / **66.4** | 66.3 | 942 / 132 | 802 / 967 | 66.3 / 65.3 |
| IQ3_XXS, full card | 83.3 / **81.2** | 77.8 | 911 / 193 | 792 / 958 | 81.0 / 77.4 |

No regression.

**Not checked.**  pi against this card (check e); images; `--vram-cap 8192`; IQ3 at the full card through `a5.py`
(its decode is in the speed row); the nondeterminism section's ten-request repeats on this card.

## REGINA-111: the five misses on the 1080 Ti

2026-10-13.  **Verdict: no fault in the engine, on either card.**  Two findings: the `2000, DMA` column above was not a
forced path, and the two fixtures that miss are ones llama.cpp fails against itself.  No engine source changed; the
binaries are the ones above (`a44d896ccb25ff98` on VM 211, `b1575740af995a78` on VM 221).  Material:
`rk@192.168.50.43:/home/rk/bench/r111/` and `rk@192.168.50.151:/models/work/r111/` (`r111-*.sh` the drivers, `pos.py`
and `dist.py` the per-position tables, `oracle_rows.py`, `SHA256.txt`).

**Why two "all GPU" configurations differed.**  They were not both all GPU.  The window's PCIe share is staged in 16
VRAM blobs (`kStagingBlobs`, `verify.hpp:358`; 8 per group when a window runs as two groups, `verify.cpp:1705`), and
the pool sends a miss to DMA only while `fetches < P.staging_cap` (`expert_source.cpp:2100`).  A layer of a window with
more than 16 distinct misses computes the rest on the CPU, whatever `--pcie-frac` says.  A 4-token window routes 32
experts; with 2,399 of 10,240 slots resident that limit is reached, with 4,240 or 8,061 it is not.

- `STRATA_POOL_TRACE=1` on q4-test20, `--expert-cache 2000 --pcie-frac 1.0`: 46 of 680 layer calls publish 16 fetches
  and then run 1 to 16 CPU jobs; no call with fewer than 16 fetches runs any.  The first is window 1 (positions 1 to
  4), layer 0: 16 by DMA, 12 on the CPU.  The first logits row that differs from the `auto` run is position 3.
- With `--spec 2` (at most 16 routed experts a window) the log reads `CPU experts 0.00`, and the `2000` and `auto`
  dumps are the same bytes, for both fixtures; for q4-test20 they are also the bytes of the `auto` run at `--spec 4`
  (`790e501683ea6af6`).  The GPU path on sm_61 does not depend on what is resident or on the window's shape.
- The 4090 does the same: `--expert-cache 2000 --pcie-frac 1.0` at `--spec 4` reads `CPU experts` 0.33 (Q4) and 0.68
  (IQ3) per layer and its dumps differ from the all-resident ones; at `--spec 2` they are the all-resident bytes
  (`e4e5fdc81b4bc604`, `de2fd0e08c2e6679`).  The earlier resident-against-DMA identity there was measured at
  `--expert-cache 8000`, which never reaches 16 misses.  Nothing here is specific to Pascal.

So in the A5 table the `auto, DMA` column is the GPU path (`CPU experts 0.00` in all six logs) and the `2000, DMA`
column is the GPU path with 0.22 to 0.73 CPU-computed experts per layer mixed in.  A run meant to force the GPU path
has to show `CPU experts 0.00`; `--spec 2` guarantees it.

**The soft positions.**  The fixtures' reference is llama.cpp on the 4090, every layer offloaded.  The same pinned
llama.cpp was teacher-forced along the same ids on VM 211, once with no layer offloaded (`--ngl 0`, "CPU" below) and once on the 1080 Ti with
what fits (28 of 41 layers for IQ3, 15 of 42 for Q4), and `ref/qwen36.py` (the numpy oracle: F32 weights as
dequantized, F32 activations, its own routing) was run over prompt and continuation in one pass, plain and with
`--act-quant q8`.  Each run against the fixture's reference, and against the F32 oracle (top-1 of 64, mean KL, the
largest KL and its position; **bold** misses the rule's thresholds):

| run | q4-test20 vs llama.cpp 4090 | q4-test20 vs F32 oracle | iq3-prime-add vs llama.cpp 4090 | iq3-prime-add vs F32 oracle |
| --- | --- | --- | --- | --- |
| llama.cpp, the 4090 (the reference) | - | **61**, 6.1e-3, 0.141 (8) | - | 64, 7.6e-3, 0.316 (40) |
| llama.cpp, CPU | **60**, 9.2e-3, 0.174 (8) | 63, 5.5e-3, 0.079 (8) | 62, **2.72e-2**, 1.400 (40) | 62, **1.48e-2**, 0.626 (40) |
| llama.cpp, the 1080 Ti, partly offloaded | **60**, 9.9e-3, 0.199 (8) | 63, 5.7e-3, 0.087 (8) | 62, **2.70e-2**, 1.368 (40) | 62, **1.31e-2**, 0.570 (40) |
| engine, 1080 Ti, GPU only (`auto`, DMA) | **61**, 8.5e-3, 0.138 (8) | 64, 4.3e-3, 0.050 (10) | 64, **1.26e-2**, 0.416 (40) | 64, 4.5e-3, 0.066 (48) |
| engine, 1080 Ti, `auto`, CPU misses | 62, 6.0e-3, 0.049 (10) | 62, 7.0e-3, 0.161 (8) | 64, 4.5e-3, 0.062 (54) | 64, 5.9e-3, 0.244 (40) |
| engine, 1080 Ti, `2000`, GPU with the CPU overflow | 62, **1.17e-2**, 0.323 (8) | 63, 6.5e-3, 0.122 (8) | 63, **1.18e-2**, 0.467 (40) | 63, 7.4e-3, 0.141 (40) |
| engine, 1080 Ti, `2000`, CPU misses | **61**, 8.3e-3, 0.086 (8) | 62, 5.7e-3, 0.083 (8) | 64, 4.8e-3, 0.073 (45) | 64, **1.22e-2**, 0.612 (40) |
| engine, 4090, all resident | **61**, 7.4e-3, 0.147 (8) | 64, 3.8e-3, 0.040 (8) | 64, 6.6e-3, 0.152 (40) | 64, 3.5e-3, 0.082 (40) |
| Q8 oracle | | 63, 7.5e-3, 0.109 (8) | | 64, 7.6e-3, 0.186 (45) |

llama.cpp on another back end misses the rule against its own 4090 run on both fixtures, by more than any engine run
does.  Against the F32 oracle the engine's GPU path is inside the thresholds on both cards (1080 Ti 64 of 64 at 4.3e-3
and 4.5e-3; 4090 64 of 64 at 3.8e-3 and 3.5e-3) and nearer the oracle than the reference is (61 at 6.1e-3; 64 at
7.6e-3).  The 1080 Ti is 0.5e-3 and 1.0e-3 farther from the oracle than the 4090, 64 of 64 on both.

The positions themselves (margins are logits, first candidate minus second; KL against the reference / the F32 oracle):

| position | llama.cpp 4090 | llama.cpp CPU | llama.cpp 1080 Ti | engine 1080 Ti, GPU only | engine 4090, resident | F32 oracle |
| --- | --- | --- | --- | --- | --- | --- |
| q4-test20, 8: " operation" / " expression" / " list"; entropy 4.2 nats | " operation" by 0.21 (p 0.21 / 0.17); - / 0.141 | " list"; 0.174 / 0.079 | " list"; 0.199 / 0.087 | " list"; 0.138 / 0.029 | " list"; 0.147 / 0.040 | " list" (p 0.15) |
| q4-test20, 9: ":" / "?" | ":" by 0.15 | "?" by 0.14 | "?" by 0.18 | "?" by 0.27 | "?" by 0.30 | "?" by 0.13 |
| q4-test20, 13: " Extract" / " Take" | " Extract" by 0.13 | " Take" by 0.08 | " Take" by 0.12 | " Take" by 0.28 | " Take" by 0.20 | " Take" by 0.07 |
| iq3-prime-add, 40: "user" / `<\|im_start\|>` | "user" by 2.70 (p 0.86); - / 0.316 | `<\|im_start\|>` by 1.57; 1.400 / 0.626 | `<\|im_start\|>` by 1.26; 1.368 / 0.570 | "user" by 0.30; 0.416 / 0.053 | "user" by 0.96; 0.152 / 0.082 | "user" by 0.77 (p 0.55) |

At q4-test20 positions 8, 9 and 13 every other source, the F32 oracle included, picks what the engine picks: the three
"misses" are the 4090 llama.cpp run standing alone at near ties.  iq3-prime-add position 40 is the fifth token after
the reply's `<|im_end|>` (position 35), in `<|im_end|>\n<|endoftext|><|im_start|><|im_start|>user`: text no request
is ever served, where llama.cpp's own back ends differ by 4.3 logits on the margin (+2.70 to -1.57) and the engine sits
between them, next to the oracle.  On that fixture positions 0 to 35 read mean KL 2.7e-3 (largest 0.025) for the 1080
Ti's GPU path and 3.4e-3 (0.071) for the 4090; positions 36 to 63 read 2.5e-2 and 1.1e-2, and llama.cpp's CPU run
5.5e-2.

**Is 0.4 at one position acceptable.**  Per-position KL against the reference, the 24 runs of the table above and
the 4090's six: the median is 1e-6 to 2.5e-3 in every run and the 90th percentile at most 2.4e-2.  17 of 30 runs
have no position above 0.05.  Positions above 0.1 exist only on these two fixtures: position 8 of q4-test20 (0.138 to
0.323 in three of five runs) and positions 40, 42 and 48 of iq3-prime-add (0.416, 0.128, 0.119 in the GPU-only run;
0.467 with the overflow; 0.152 on the 4090).  With its largest position left out no run's mean is above 7.1e-3.  A KL
of 0.4 at a position where two llama.cpp back ends are 1.4 apart measures the reference, not the engine; the same
0.4 at a position where they agree to 1e-2 would be a fault; no position of the other four fixtures is above 0.1.

**Proposed for the rule, not adopted** (the thresholds 62 of 64 and 1e-2 stand as written until decided):

1. Score positions up to and including the reference's first stop token only (36 of 64 on iq3-prime-add; the other five
   fixtures do not stop), or regenerate that fixture to hold 64 positions before the stop.
2. Keep a second llama.cpp run per fixture (`--ngl 0`, forced along the same ids; under half a minute each) and
   report, not gate, the positions where the two llama.cpp runs disagree on top-1 or differ by more than 0.05 KL, as
   the runner tests already do where the F32 and Q8 oracles disagree.  q4-test20 loses positions 6, 8, 9, 13, 15;
   iq3-prime-add loses 0, 2, 8, 40.  The thresholds become shares of what is left (96.9% and 1e-2).
3. A forced-GPU run counts only with `CPU experts 0.00` in its log.

With 1 and 2 all eight 1080 Ti runs of the two fixtures pass: q4-test20 59 of 59 in each, mean KL 5.2e-3 to 5.8e-3;
iq3-prime-add 33 of 33 in each, 8.9e-4 to 2.1e-3.

**Left as it is, for a decision.**  `--pcie-frac 1.0` does not force the GPU past 16 distinct misses per layer and
window (8 per group).  README line 232 and the paragraph "For a run that must repeat" above promise repeats from that
flag; with a small cache (the 1080 Ti at `--expert-cache 2000`, or a `--vram-cap` configuration) the overflow runs
on the CPU and the adaptive tier can then move a request between passes as in the CPU rows of that section.  Sizing the
staging to the window (8 blobs per token: 16 more at 4 tokens, under 58 MiB at Q4) would make the flag exact; it also moves
the default share's ceiling, so it changes what the ladder measured and was not done here.

**Not checked.**  A Pascal-against-Ada bisect by layer: the two cards run different kernels from the prompt on (the
BF16 GEMM through fp32 and no tensor-core MoE on sm_61), so they are not expected to agree in bytes, and each agrees
with the F32 oracle to 4e-3; the other four fixtures against the oracle or a second llama.cpp run; llama.cpp fully
offloaded on a second card (neither file fits the 1080 Ti); whether the overflow makes a served request differ between
passes (inferred from the CPU rows above, not run); the 262,144 q4_0 configuration.

## Flash-Next regression check

2026-10-13, VM 221 (RTX 4090, 62 GB RAM), card otherwise idle.  Does the branch still serve Qwen3.8-Flash-Next as the
production binary does?  Production: `/tank/nvme/strata/unbiased-strata`, sha256 `abbe990f4e3c658e` (main's lineage
with 121d23a).  Branch: `/models/work/qwen36-step4/build/unbiased-strata`, `11da5388afdfabcb` at d00d383 and
`5428c8d791f7a697` at 0513d5b.  Both started by hand with the command line of `unbiased-strata.service` on
127.0.0.1:18091 (`--ctx 196608 --mmproj ... --threads 8 -- --kv-resident 32768 --ple-io mmap --kv int8`,
`CUDA_MODULE_LOADING=EAGER`, memlock unlimited; the unit's first-prompt capture variable left out so the captured file
was not overwritten).  Scripts and logs: `/models/work/flashcheck/`.

**One fault, fixed (REGINA-112).**  The branch took f07b81e from main but not 121d23a, so its front end sized
`max_new` as context - prompt and the engine (prompt + max_new + 8 <= context) refused every request that sent no
`max_tokens`, and any with `max_tokens` at or above context - prompt - 7, with a 500.  The serving checks below did
not see it because each sets `max_tokens`.  0513d5b carries main's hunk and `tests/python/serve_max_tokens_check.py`
(`make test-serve PORT=...`): 0 of 5 on `11da5388`, 5 of 5 on production and on `5428c8d7`, the latter on Flash-Next
at 196,608 and on Qwen3.6 Q4_K_XL at 262,144.

**Serving**, `a5.py` of the A5 section and one image request, on both branch binaries:

| check | d00d383 | 0513d5b |
| --- | --- | --- |
| start, plan | 10,368 expert slots (production 10,373), 430 MiB free, KV 32,768 of 196,608 cells resident | the same |
| `/health`, `/v1/models` | ok, tiers; `input_modalities` text, image | ok |
| chat, not streamed, thinking off / on | pass / pass (391) | pass / pass |
| stream, thinking on: reasoning deltas, content deltas | 84, 41 of 128 tokens, largest gap 77 ms | 90, 50 of 143 |
| stream, thinking off: content deltas | 193 of 194 tokens | 200 of 201 |
| tool round trip, not streamed / streamed | pass / pass (the call in 1 delta, the answer in 14) | pass / pass (16) |
| tools declared, not called, streamed, thinking on / off | 79 / 90 content deltas | pass / pass |
| three turns, tokens reused of the prompt | 0 / 65, 60 / 138, 133 / 169 | 0 / 65, 60 / 145, 140 / 176 |
| image (96 x 96 PNG, red left, blue right) | pass, 1.0 s, 41 prompt tokens | pass |

**Speed**, greedy, reasoning off: 4 prompts of 400 tokens and 2 prompts of 19,593 and 20,820 tokens, five passes per
start, production, branch (d00d383), production, branch.  The host holds 62 GB against 76 GB of model files, so
14,208 of the 24,576 experts and the n-gram table are read through the file cache and the first two passes of a start
measure the cache (60 tokens/s and 700 prompt tokens/s on both binaries on the first pass).  The table is passes 3 to
5, tokens over time, per start:

| | Generation, tokens/s | Prompt, tokens/s |
| --- | --- | --- |
| Production | 89.5, 92.1 (74 to 109 per prompt) | 2,093, 2,098 |
| Branch | 89.0, 89.2 (74 to 107) | 2,092, 2,094 |
| Branch / production | 0.98 | 1.00 |

Drafts accepted 0.61 to 0.62 of those offered on both.  The 2% in generation is inside the spread between the two
production starts.

**ctest**, the branch build at d00d383 on the free card: 94 of 98 pass, `pack_load_check` skipped, and the three
host failures (`ple_parity`, `platform_memory_test`, `expert_multi_test`).

**On main.**  main at e7ae2c8 builds the same binary as 0513d5b (`5428c8d791f7a697`).  One more start of each, same
requests, passes 3 to 5: production 90.6 tokens/s and 2,097 prompt tokens/s, main 90.4 and 2,091 (1.00, 1.00).

### Both models on the one 4090, two processes

One process serves one model: the front end loads one `--model`, starts one engine child, and answers every request
with it whatever the request's `model` field says.  Two models are two processes on two ports.  Tried once, main's
binary, Flash-Next on 18091 and Qwen3.6 IQ3_XXS at 262,144 on 18092, `--threads 8` each on 16 vCPUs:

- **`--vram-cap` does not partition a card.**  It allocates a blocker of total - cap, so one capped engine holds the
  whole card.  What works: start Flash-Next first with a fixed `--expert-cache N`, then Qwen3.6 uncapped, whose plan
  takes what is free.  `--expert-cache 4000` gave Flash-Next 5,357 slots and 15.4 GB of the card, with 8.8 GB free;
  Qwen3.6 then planned a 4,331 MiB fixed set and 3,187 of 10,240 expert slots (8.1 GB), 447 MiB left.  At
  `--expert-cache 6000` (8,053 slots) 4.3 GB stayed free, below Qwen3.6's fixed set.
- **RAM binds first on this host.**  Flash-Next alone uses 47 of 62 GB (its 40 GiB expert arena is pinned).  With
  Qwen3.6 beside it 60 GB were used and 2 GB available; Qwen3.6 found too little to pin its 7,053 off-card experts
  and mapped them from the file, and took 239 s to start.  No swap, no OOM kill in this run.
- **Idle costs nothing, concurrency costs most of both.**  An idle engine shows 0% GPU.  4 prompts of 400 tokens,
  greedy, reasoning off, tokens/s:

| | Flash-Next | Qwen3.6 IQ3_XXS |
| --- | --- | --- |
| Alone on the card, service configuration (10,368 slots) | 90 | |
| Alone, `--expert-cache 4000` (5,357 slots) | 54 | |
| The other engine loaded and idle | 45 | 133 |
| Both generating | 12 | 27 |

  Requests to both at once complete without error; each engine runs at about a quarter of its speed with the other
  idle.  Two CUDA contexts time-slice the card, each engine's verify graph spins on the GPU while it waits for its
  host, and 16 pool workers plus two host threads share 16 vCPUs.  Not separated which of the three costs what.
