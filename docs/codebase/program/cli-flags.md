[Codebase docs](../README.md) › [Program](README.md) › Command-line flags

# Command-line flags

**Covers:** `struct Options` (`src/program/generate.cpp:265-567`), `usage()` (`:569-757`) and the argv loop
(`:1308-1593`). The loop is two `if` chains (the second starts at `:1417`) because one chain of 120+ `else if`
exceeded MSVC's nesting limit (C1061); order and behaviour are unchanged. An **unknown flag is an error** (exit 2 after
printing `usage()`, `:1585-1591`); a flag missing its value exits 2 (`next()`, `:1310-1313`). Numbers are parsed with
`atoi`/`atoll`/`atof` unless noted, so a malformed number silently becomes 0.

Flags that are parsed but have **no line of their own in `usage()`**: `--spec`, `--mtp`, `--serve`, `--layer-split`,
`--spec-oracle` (these are only mentioned inside other entries), and `--adapt-decay`, `--adapt-every`, `--adapt-swaps`
(not mentioned at all, nor are the following), `--dump-final-r`, `--eos-ids`, `--mtp-window`, `--no-fast-attn`,
`--no-fast-select`, `--no-fused-gdn`, `--no-prefill-borrow`, `--no-publish-kernel`, `--spec-split`, `--no-spec-split`,
`--no-split-rows`, `--pcie-frac`, `--pcie-mode`, `--prefill-until`, `--prompt-cache-root`, `--resident-budget-gib`,
`--spec-corrupt`, `--spec-min-p`, `--split-device`, `--split-skip-if-fits`, `--stop-eos`, `--vram-reserve-mib`.

The front end starts the engine with `--engine --serve --pack P --native MODEL --expert-profile P/expert-profile.bin
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp DIR --max-context CTX`, plus `--ple-gguf SHARD2
--ple-io ram` when the model has a second shard, `--kv int8` above 8K context, `--pool-workers N`, `--vram-cap`,
`--ram-cap`, `--vision`, and anything after `--` (`frontend/main.cpp:676-692`).

## Input, output length, context

| Flag | Default | Effect |
| --- | --- | --- |
| `--help`, `-h` | | Print `usage()`, exit 0 |
| `--pack DIR` | `pack/full` | The pack directory (`index.txt`, the arena, `experts.bin` or `native_experts.txt`) |
| `--tokens LIST` | (required unless `--serve`) | Prompt as comma/space-separated ids in `[0, 2^31)`; parsed by `parse_i64_list` (`:759`). Only one of `--tokens`/`--tokens-file` |
| `--tokens-file PATH` | | The same from a file (commas or whitespace; NUL bytes refused) |
| `--max-new N` | 16 | Tokens to generate. Must be `> 0`, `<= --max-context`, and prompt + N must fit (`:1938`) |
| `--max-context N` | 4096 | KV/state capacity in cells; must be below 2^31 - 8. The Qwen3.6 VRAM plan may lower it (`:2281-2285`) |
| `--stop-eos` | off | Stop at an end-of-turn id (`--eos-ids`); forced on in `--serve` with no `--tokens` |
| `--eos-ids LIST` | `248044,248046` | End-of-turn ids (`<|endoftext|>`, `<|im_end|>`); also turns `--stop-eos` on |

## Sampling (one-shot mode)

`--serve` ignores these and samples per request from `GEN` keys ([serve-mode.md](serve-mode.md)).

| Flag | Default | Effect |
| --- | --- | --- |
| `--greedy` | on | Argmax |
| `--seed S` | 0 | Philox seed; **also switches greedy off** |
| `--top-k N` | 20 | Top-k (the start-up log prints it capped to 64, `:4273`) |
| `--top-p F` | 0.95 | Nucleus; must be in `(0, 1]` |
| `--temperature F` | 1.0 | Must be finite and `>= 0`; `<= 0` is reported as greedy |

## Rope scaling (context extension)

Resolved once before `session_init` (`:2394-2487`). Precedence: explicit flag, then the model file's
`<arch>.rope.*` keys, then defaults. `none` and `1` are explicit opt-outs. With type `none` every scaling knob is reset
(only the frequency base survives) and a note is printed if any were given.

| Flag | Default | Effect |
| --- | --- | --- |
| `--rope-scaling none\|linear\|yarn` | absent (model file, else none) | Scaling type, llama.cpp's names |
| `--rope-scale F` | 0 = absent | Extension factor, must be `>= 1` when given |
| `--rope-freq-base N` | 0 = model's (1e7) | Must be `> 1` when given |
| `--rope-freq-scale F` | 0 = `1/--rope-scale` | Raw ggml knob |
| `--yarn-orig-ctx N` | 0 = model's, else 262144 | Trained context YaRN corrects toward |
| `--yarn-ext-factor F` | -1 = auto (1 for yarn, 0 otherwise) | |
| `--yarn-attn-factor F` | 1.0 | Must be `> 0` |
| `--yarn-beta-fast F` | 32.0 | Must be `> 0` |
| `--yarn-beta-slow F` | 1.0 | Must be `> 0` |

All knobs must be finite (`:1960-1965`); the resolved config is checked as a whole by `rope_scaling_invalid`.

## Model files and the native path

| Flag | Default | Effect |
| --- | --- | --- |
| `--native SHARD1` | off | The whole native arithmetic set at once (`:1844-1905`): every shard of the model (`gguf_split_paths`; a missing shard is an error), `--ple-gguf` defaulted to the shard holding `per_layer_token_embd.weight`, and sets `--stream-token`, `--gr-native-mmvf`, `--native-bf16`, `--native-bf16-extra`, `--native-moe-combine`, `--native-gdn`, `--native-router`, `--native-qsa`, `--native-qsa-indexer`, `--native-rope`, and `--native-ple-key`/`--native-ple-postops` when the model has a PLE. `--native-head-gguf` defaults to SHARD1, `--native-dense-gguf` to every shard plus the PLE shard (unless that is a PLE-only `strata-ple` file). Required by a native (IQ) pack. Note: `usage()` says it also selects the CPU q8_0 contract; the code does not (`:1903-1904`) |
| `--native-head-gguf PATH` | (SHARD1 with `--native`) | Native output head; requires `--stream-token` |
| `--native-dense-gguf PATH` | | Native GDN/QSA/shared projections; repeat per shard |
| `--embd-gguf PATH` | | Token embedding from this GGUF instead of `--native`'s (mapped host memory) |
| `--ple-gguf PATH` | (auto with `--native`) | The PLE (n-gram) table shard. Required unless `--no-ple` or the model has no PLE (Qwen3.6: giving it is an error, `:1870-1875`) |
| `--no-ple` | off | Diagnostic ablation; a different model. Exclusive with `--ple-gguf` |
| `--ple-io direct\|mmap\|ram` | `direct` | Table reads: unbuffered SSD reads (`direct`), plain mmap, or mmap locked in RAM (`ram`, not on Windows) |
| `--ple-row-cache N` | 1048576 | Bounded row cache (90 B rows); 0 disables |
| `--ple-inflight N` | 256 | Outstanding reads, 1..1024 |
| `--ple-delay-us U` | 0 | Fault injection: each row read completes no earlier than U us |
| `--ple-sync-submit` | off | Submit reads on the token thread instead of an I/O thread |
| `--no-ple-prefetch` | off | Read the PLE rows one at a time |
| `--keep-canonical` | off | Also load canonical copies of natively served tensors (more VRAM). Refused with a native pack |
| `--cpu-oracle-q8-0` | off | llama.cpp-exact CPU q8_0 contract; refused with `--expert-cache` or `--expert-profile` |
| `--stream-token` | off | Ordered token work on the session stream; skips the per-token device sync |
| `--check-logits` | off | Copy back and scan all logits even in the stream-token path |
| `--gr-fp32-activations`, `--gr-native-mmvf`, `--native-bf16`, `--native-bf16-extra`, `--native-ple-key`, `--native-moe-combine`, `--native-gdn`, `--native-qsa-indexer`, `--native-qsa`, `--native-rope`, `--native-ple-postops`, `--native-router` | off | Individual native-arithmetic switches (A/B arms of `--native`); applied at `:2616-2653` |
| `--native-flash-attn-short` | off | Diagnostic pinned attention; needs `--max-context <= 256` and the normal decode loop |

## KV cache

| Flag | Default | Effect |
| --- | --- | --- |
| `--kv fp16\|int8\|q4_0\|k8v4` | `fp16` | KV storage (`q4` is an alias of `q4_0`). Sets `qsa_set_kv_int8/q4/hybrid` (`:1818-1826`). `int8`: int8 codes + fp16 scale per 64 values; `q4_0`: 4-bit after a Hadamard rotation; `k8v4`: INT8 K + rotated Q4_0 V (816 B/cell) |
| `--kv-resident N` | 0 | KV streaming: N cells per attention layer in VRAM, the whole K/V in pinned RAM. Not with `k8v4`; refused for a model without a selected-attention indexer (`qsa_kv_streaming_applies`) |

## Expert tiers and memory

See [expert-tiers.md](expert-tiers.md) for how these interact.

| Flag | Default | Effect |
| --- | --- | --- |
| `--expert-cache N\|auto` | 0 (off) | VRAM expert slots; `auto` = -1 (sized from free VRAM). With `--expert-profile`, 0 also means auto (`:2993`) |
| `--expert-profile P` | | `profile.bin` ranking (layer, expert) pairs; fills the cache once at start and makes residency static except for adaptive swaps |
| `--expert-profile-save P` | | `--serve`: write the learned ranking on `QUIT` and periodically |
| `--expert-profile-save-every MIN` | 10 | Minutes between saves (0 = only on QUIT) |
| `--expert-cache-per-layer` | off | Each layer owns its own slot range (keeps uniform slot sizes) |
| `--expert-cache-cpu-order` | off | GPU expert reduction in the CPU's order |
| `--no-hit-poke` | off | A/B: do not poke the driver after the hit launch |
| `--vram-reserve-mib N` | 700 | VRAM left free after the auto cache; may be lowered automatically to 300 on small cards unless given (`:3583-3600`) |
| `--vram-cap MIB` | 0 | Emulate a smaller card: free figures clamped, a blocker of `total - cap` allocated (`:2130-2172`) |
| `--ram-cap GIB` | 0 | Bound the RAM tier and the pinned arena (`set_host_ram_cap`) |
| `--vram-elastic` | off | `--serve`, one NVIDIA GPU: cache in physical segments so the `VRAM` command can resize it |
| `--vram-segment-mib N` | 512 | Segment size; below 64 turns `--vram-elastic` off |
| `--mmap-experts` | off | Read experts through `FileExpertSource` (mmap of `experts.bin`, or the GGUF shards in place) instead of the resident arena |
| `--shared-expert-arena FILE` | | Linux: back the resident arena with a `MAP_SHARED` file (e.g. on `/dev/shm`); not with `--mmap-experts` |
| `--resident-cpu-experts` | off | With `--mmap-experts` and a profile: copy the experts the GPU does not hold into RAM |
| `--resident-experts` | off | Low-RAM resident mode: `--mmap-experts --resident-cpu-experts`, page-locked when possible, 4 GiB headroom, soft fallback to plain mmap. Env `STRATA_RESIDENT_PIN=0`, `STRATA_RESIDENT_HEADROOM_GIB` adjust |
| `--resident-budget-gib N` | 0 | Resident mode with a RAM budget: the N GiB of hottest non-VRAM experts in RAM, the rest read from files (implies `--mmap-experts --resident-cpu-experts`, pinned) |
| `--pcie-frac F` | <0 = auto | Share (0..1) of each layer's missed experts the GPU reads over PCIe in verify windows. Auto: 0.2 for the canonical pack, 0.55 for native packs scaled by a measured H2D bandwidth below 20 GB/s (`:2366-2380`) |
| `--pcie-mode auto\|dma\|kernel\|direct` | `auto` | How those reads happen; passed as 0 = dma, 1 = direct, 2 = otherwise (`:5516`, `:8284`) |
| `--adapt-every N` | 4 | Adaptive VRAM tier: swap every N verify rounds; 0 = static residency |
| `--adapt-swaps N` | 96 | Max swaps per adaptation |
| `--adapt-decay F` | 0.7 | Usage counts multiplied by F after each adaptation |
| `--expert-cache-device1/2/3 N\|auto` | 0 | Helper expert caches on CUDA1..3 (must be enabled in order) |
| `--expert-cache-remote-placement stripe\|layer` | `stripe` | Stripe ranks across helpers or give each layer one helper |
| `--remote-expert-opt` | off | Remote-expert decode optimizations; requires `--serve` |
| `--peer-device N` | -1 | Second GPU as an adaptive expert tier; not with `--layer-split` or helper caches; needs a profile and cache |
| `--peer-reserve-mib N` | 600 | VRAM the peer leaves free |
| `--peer-slots N` | 0 = what fits | Peer slot cap |
| `--peer-adapt-swaps N` | -1 = `--adapt-swaps` | Peer swaps per adaptation |
| `--peer-prefill-rows N` | -1 = half of chunk x top-k | Prompt rows per layer the peer computes; 0 keeps the prompt on the primary |

## CPU expert pool

| Flag | Default | Effect |
| --- | --- | --- |
| `--pool-workers N` | 0 = all physical cores minus the host's | Worker count |
| `--pool-affinity all\|auto\|p-cores` | `all` | Worker placement (`pcores` accepted); other values exit 2 |
| `--no-host-worker` | off | The host thread does not join the drain |
| `--no-split-rows` | off | One whole expert per pool thread |
| `--no-pool` | off | Skip the CPU pool: the GPU-only floor (wrong tokens by design) |

## Prompt path (batched prefill)

| Flag | Default | Effect |
| --- | --- | --- |
| `--prefill CHUNK\|auto\|auto:N` | 0 = token path | Batched prompt processing in chunks. `auto` = the largest chunk up to 8192 the cache can lend; `auto:16384`/`auto:32768` (or `STRATA_PREFILL_AUTO_MAX` with bare `auto`) raise the ceiling, rounded down to 8192/16384/32768 (`:1440-1449`). Without a profile or with `--no-prefill-borrow`, `auto` becomes a fixed 2048 (`:1785-1788`) |
| `--no-prefill-borrow` | off | Reserve the prompt buffers for the whole session instead of borrowing cache slots |
| `--prefill-until P` | 0 | Validation: batch only positions `[0, P)`, run the rest through the token path |
| `--short-read N` | 64 | `--serve`: a prompt part of at most N fresh text tokens goes through verify windows instead of the batched path; 0 = always batched |

## Speculative decoding

See [speculative-decoding.md](speculative-decoding.md).

| Flag | Default | Effect |
| --- | --- | --- |
| `--spec T` | 0 = plain decode | Verify-window size (last token + T-1 drafts), max `kVerifyMaxT` = 8. With `--suffix-draft > 0`, `T >= 2` and no `--mtp-max-t`, the engine sets `--mtp-max-t T` and `--spec min(T+2, 8)` (`:1839-1842`) |
| `--mtp DIR` | | MTP draft-layer runtime directory; ignored (cleared) without `--spec >= 2` |
| `--mtp-window N` | 32768 | Draft layer attends to the last N cells (0 = all) |
| `--mtp-max-t M` | 0 = `--spec` | Cap on the MTP's own windows; longer ones come only from suffix drafts |
| `--suffix-draft N` | 3 | Prompt-lookup drafts from an earlier repeat of the last N+ tokens; 0 = MTP only |
| `--spec-min-p F` | 0.0 | Shrink the window to drafts whose draft probability stays `>= F` |
| `--coupled-draft` / `--no-coupled-draft` | `STRATA_SPEC_COUPLED` | Sampled (coupled) MTP drafts under sampling vs argmax drafts |
| `--spec-split` / `--no-spec-split` | off | Opt-in split verify window (exact, ~7% slower per the comment) |
| `--spec-oracle FILE` | | Drafts from a token file (exactness test) |
| `--spec-corrupt N` | 0 | Replace every Nth draft with a wrong token |
| `--teacher-force` | off | With `--spec-oracle`: commit every row and continue from the oracle's token; not with `--serve` or `--mtp` |

## Serve mode

See [serve-mode.md](serve-mode.md) and [prompt-cache.md](prompt-cache.md).

| Flag | Default | Effect |
| --- | --- | --- |
| `--serve` | off | Stay resident, take requests on stdin. Requires `--spec >= 2`, `--mtp`, `--prefill` |
| `--vision` | off | Keep an M-RoPE (t, h, w) table so `GENI` image requests work |
| `--prompt-cache N` | 6 | Conversation checkpoints kept (~118 MB of RAM each per the comment); 0 = re-read every prompt |
| `--prompt-cache-every N` | 16384 | Also checkpoint every N fresh prompt tokens; 0 = off |
| `--prompt-cache-root N` | 2048 | Checkpoint the first turn boundary (end of system prompt) of a prompt read from 0 when it is at least N tokens; 0 = never |
| `--turn-token ID` | 248045 | The `<|im_start|>` id that marks turn boundaries |
| `--tail-role-token ID` | -1 | A short turn of this role before the last turn stays out of the checkpoint |
| `--conversation-cache-mib N` | 0 | RAM budget for parked conversations (strictly parsed non-negative integer) |
| `--conversation-cache-slots N` | 4 | Max parked conversations |
| `--conversation-cache-min-free-mib N` | 2560 | Physical RAM floor when parking |
| `--batch N`, `--slots N` | 0 | Up to N requests decode together in batch slots (2..8); needs `--serve` |
| `--batch-groups G` | 1 | With a layer split: slot groups pipelined through the stages; must divide `--batch` |
| `--expert-profile-save*` | | See above |

## Multi-GPU layer split

See [layer-split.md](layer-split.md).

| Flag | Default | Effect |
| --- | --- | --- |
| `--layer-split K[,K2..]\|auto` | none | `--serve` only. Each K is the first layer of a later GPU; `auto` places them by a cost model. Needs `--expert-profile` across GPUs |
| `--split-device D1[,D2..]` | the next visible GPUs | Devices for the later stages; `0` with one K = both stages on this GPU (bit-exact A/B) |
| `--split-skip-if-fits` | off | With `auto`: run on CUDA0 alone if it holds every profiled pair plus session and reserve |
| `--trim-stage-weights` | off | Explicit split only: each GPU loads only its layers' dense weights (same as `STRATA_STAGE_TRIM=1`) |

## Control vectors

| Flag | Default | Effect |
| --- | --- | --- |
| `--control-vector FILE` | | A llama.cpp control-vector GGUF at scale 1 (repeatable) |
| `--control-vector-scaled FILE:SCALE[,...]` | | With scales; the last colon splits (Windows drive letters survive) |
| `--control-vector-layer-range A B` | 1 .. last | Inclusive layer range (two values) |
| `--cvec-mode add\|project` | `add` | `h += s v` or `h -= s (h.v) v` |
| `--cvec-dir per-layer\|single:L` | per-layer | Project mode: use layer L's direction everywhere |

Loading is `load_control_vectors` (`:1130-1199`); `--serve` requests switch it with `cvec=0|1`.

## Diagnostics and A/B arms

See [diagnostics.md](diagnostics.md).

| Flag | Default | Effect |
| --- | --- | --- |
| `--dump-logits PATH` | | Header `{n_vocab, n_rows}` + one f32 row per position |
| `--logits-stride N` | 1 | Store every Nth row plus the final one; N > 1 needs `--max-new 1` and `--dump-logits`; only once; strict positive decimal |
| `--dump-residual PATH` | | Final residual `R` (`hc x n_embd`) with a 2-int header |
| `--dump-mixed PATH` | | The head input `bb.mixed` (`n_embd` floats) |
| `--dump-layers PATH` | | Residual after every layer per position (needs the captured path) |
| `--dump-halves PATH` | | Per-layer attention/MoE halves (armed before capture) |
| `--dump-routing PATH` | | Routed ids and weights per layer per position (needs the pool) |
| `--dump-final-r PATH` | | Token path: `{pos, tok}` + final residual per position |
| `--stats` | off | Per-stage breakdown in the final report |
| `--stage-timing` | off | Per-stage event shares; needs `--no-capture` |
| `--graph-only` | off | Replay the `pre` graphs only, print, exit |
| `--gpu-only-full` | off | Replay pre + post + head (the GPU floor), print, exit |
| `--gpu-stages` | off | Three graphs per layer timed from outside the capture, print, exit |
| `--no-capture` | off | Run layers directly (`session_token`); requires `--no-pool` |
| `--sync-every-layer` | off | Debug sync after every layer (with `--no-capture`) |
| `--shared-late` | off | Shared expert after the CPU pool (old order) |
| `--no-token-graph` | off | Two graphs per layer instead of one per token |
| `--no-fused-gr`, `--no-fast-attn`, `--no-publish-kernel`, `--no-fused-gdn`, `--no-fast-select` | off | Plan v0.3 kernel A/B arms (`:2620-2625`) |

## Cross-flag rules checked at start

Beyond the per-flag checks above (`:1596-2041`): `--resident-cpu-experts` needs `--mmap-experts` and a profile and
does not work with layer splits or helper caches (setup's soft `--resident-experts` under a split is downgraded to
plain mmap with a warning, `:1672-1682`); `--teacher-force` needs `--spec-oracle`, not `--serve`, not `--mtp`;
`--native-ple-key` needs PLE and `--native-dense-gguf`; `--peer-device` excludes `--layer-split` and helper caches;
`--gpu-only-full`/`--gpu-stages` are refused across a multi-GPU split; `--no-capture` is refused unless `--no-pool`
and with `--dump-layers`/`--dump-halves`; `--dump-routing` is refused with `--no-pool`; `--stage-timing` without
`--no-capture` is refused. `--batch` problems are warnings that turn batching off rather than errors (`:3274-3340`).

## See also

- [startup.md](startup.md), [env-vars.md](env-vars.md)
- [../frontend/README.md](../frontend/README.md) for the flags the front end passes
