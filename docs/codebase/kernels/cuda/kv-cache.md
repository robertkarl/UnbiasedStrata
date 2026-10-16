[Codebase docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › KV cache formats and streaming

# KV cache: INT8, rotated Q4_0, hybrid K8V4, and streaming

## Purpose

The attention layers' K/V storage. The FP16 pool and its append/gather live in `qsa.cu`
([attention-qsa.md](attention-qsa.md)). This page covers the compact formats and KV streaming:

- INT8 codes with an fp16 scale per 64 values (`--kv int8`);
- a 256-point Walsh-Hadamard rotation followed by ggml's Q4_0 (`--kv q4_0`);
- the hybrid K8V4: INT8 K (unrotated, so the scores are exact up to the INT8 rounding) and rotated Q4_0 V (`--kv k8v4`);
- KV streaming (`--kv-resident N`): the authoritative K/V in pinned, device-mapped host memory, and a VRAM pool of
  page slots made resident on demand inside the captured graphs.

Only storage changes. The attention kernels read every format directly (`qsa_decode_attn`, `qsa_prompt_attn`), or
through a dequantizing gather into the same FP16 scratch.

Files: `src/kernels/cuda/kv_q8.cu`, `kv_q4.cu`, `kv_stream.cu`; headers `include/strata/kernels/kv_q8.hpp`,
`kv_q4.hpp`, `kv_stream.hpp`. The pool pointer bundles are `QsaAttnPools` (`qsa_decode_attn.hpp:34-44`) and
`KvHostPools` (`kv_stream.hpp:32-42`).

## Formats

All pools use the FP16 pool's layout: `[page][kv_head][page_size][head_dim]`, with the page found through the
page table (`qsa.hpp:63-72`). Sizes per cell at Flash-Next / Qwen3.6 geometry (2 KV heads x 256):

| Format | Per value | Bytes per cell (K + V) | Kernels |
| --- | --- | ---: | --- |
| FP16 | `f16_from_f32` | 2,048 | `kv_append_step`, `kv_gather_step` (`qsa.cu`) |
| INT8 (`KV_Q8_GROUP` = 64) | `scale = f16(max|x| / 127)`, `code = clamp(rint(x / scale_stored), -127, 127)`; codes `[..][head_dim]`, scales `[..][head_dim / 64]` | 1,056 | `kv_append_q8_step`, `kv_gather_q8_step` |
| Q4_0 + FWHT | `block_q4_0` = fp16 `d` + 16 B (element j in the low nibble of `qs[j]`, j + 16 in the high one), 18 B per 32; after rotation by H/16 | 576 | `fwht256_cuda`, `kv_append_q4_step`, `kv_append_q4`, `kv_gather_q4_step` |
| K8V4 | INT8 K + rotated Q4_0 V | 816 | the INT8 append for K and the Q4_0 append for V |

`kv_q8_bytes_per_cell`, `kv_q4_bytes_per_cell` and `kv_block_bytes` compute these. The comment in `kv_q8.hpp:3-8`
puts FP16 over Flash-Next's 12 QSA layers at 24,576 B per token against 12,672 B for INT8 (6.4 GB vs 3.3 GB at 262K).

### Why the rotation works

`H` is the orthonormal 256-point Hadamard matrix (scale 1/16), which is its own inverse. K and V are stored as Hk
and Hv. The query is rotated too, so `<Hq, Hk> = <q, k>` and the scores are unchanged. The attention output (a
mix of rotated values) is rotated back with H. The rotation spreads outlier channels over all 256 dims, which is what
makes 4 bits usable. What is lossy is the 4-bit rounding itself (`kv_q4.hpp:3-9`). In K8V4 only V is rotated, so q
is not rotated and only the output is rotated back (`src/core/layer.cpp:616`, `:1048`). `STRATA_KV_ROT=1` also
rotates INT8 KV (read in `src/program/generate.cpp:1822`).

## Main entry points

| Function | Launch | Notes |
| --- | --- | --- |
| `kv_append_q8_step` (`kv_q8.cu:103`) | grid `(kv_heads, head_dim / 64, 2)` (K and V), 64 threads | max-abs over 64 values with two warps combined in a fixed order; quantizes against the stored fp16 scale. Position from `step[kStepPos]` (capturable) |
| `kv_gather_q8_step` (`kv_q8.cu:114`) | 256 threads, one thread per 4 values | selected cells to FP16 scratch `[id][kv_head][head_dim]`; the grid is sized by `max_ids`, the count read from `step` |
| `fwht256_cuda` (`kv_q4.cu:184`) | one warp per row (`dim3(32, rows_per_block)`), 8 values per lane | butterflies: 5 levels across lanes with `__shfl_xor_sync`, 3 within a lane's registers |
| `kv_append_q4_step` (`kv_q4.cu:193`) | one warp per 32-value group, 32 threads | `q4_group` (`kv_q4.cu:75`): ggml q4_0, `d = (value of largest |x|) / -8`, `q = clamp(trunc(x / d + 8.5), 0, 15)`. Ties in `|x|` resolve to the larger value in every lane, so all lanes agree on d |
| `kv_append_q4` (`kv_q4.cu:204`) | batch kernel | the prompt path: T consecutive cells from `pos0`; also into a one-layer staging pool when given |
| `kv_gather_q4_step` (`kv_q4.cu:217`) | warp per row | to FP16 scratch, still rotated |

Every append writes the host copy (identity layout) when `KvHostPools` is present, and writes the VRAM page only
if its block is resident (page table >= 0).

## KV streaming (`kv_stream.hpp`, `kv_stream.cu`)

A streamed layer keeps K/V in pinned host memory laid out as a fully resident pool with the identity page table.
VRAM holds `n_slots` pages, where one page = `page_size` = 4 cells = one indexer block (all KV heads, K and V). The
page table becomes a residency map: `page_table[block]` = slot, or -1.

`KvStreamMap` (`kv_stream.hpp:49-59`) holds the device arrays at fixed addresses (graphs bake them in):
`page_table`, `slot_block`, `slot_stamp`, `slot_ref` (CLOCK bit), `ctl` (16 ints: epoch, hand, misses, overflow, and
u64 counters for misses / lookups / calls), `miss_block`, `miss_slot`.

| Function | What |
| --- | --- |
| `kv_stream_resolve` (`kv_stream.cu:204`) | Capturable. `resolve_kernel` (one block of 1,024 threads): (1) walk the selected ids (one lookup per block, since ids are ascending); a resident block takes this call's epoch and its reference bit, and a missing one is claimed once (-1 to -2). (2) A CLOCK sweep from the hand picks one victim per miss; a slot used in this call (stamp == epoch) is never taken, and a referenced one loses its bit and is taken on the next pass. The table is re-pointed. (3) `copy_kernel` (96 x 128 threads) copies the missed blocks from the host copy over PCIe, 16 B per thread. Requires `n_slots >= 1024`; an overflow sets `ctl[3]` and the attention masks that block |
| `kv_stream_reset` (`:199`) | evict everything |
| `kv_ring_table` (`:226`), `kv_ring_restore` (`:231`) | the MTP drafter's ring: a static table `block -> block % n_slots`, refilled from the host copy on a resume (`cudaMemcpyAsync`, not capturable) |
| `kv_stage_from_host` (`:244`), `kv_unstage_to_host` (`:254`) | the prompt path's one-layer staging pool, by DMA (not capturable); unstage is the HIP A/B `STRATA_KV_HOST_DMA=1` path |
| `kv_stream_counters` (`:264`) | synchronous read of misses / lookups / calls / overflow |

Writers always write the host copy and write a slot only if it is resident. As a result slots never go stale, and
nothing needs invalidating after a rejected draft or a conversation-cache resume (`kv_stream.hpp:16-17`).

## Control and data flow

```
k, v (F32, normed + roped) --(rotated formats: fwht256)--> kv_append_{_, q8, q4}_step  --> VRAM page (if resident) + host copy
selection ids --kv_stream_resolve--> page_table, slots filled
attention: qsa_decode_attn_step / _batch / qsa_prompt_attn_batch read the pools through page_table
           (rotated formats: q rotated before, output rotated back after)
```

Callers: `src/core/layer.cpp:957-971` (append), `:777` (`kv_stream_resolve`, via `qsa_kv_resolve`), `:1007-1048`
(rotation); `src/core/verify.cpp:858-866`; `src/core/mtp.cpp:668-671`, `:905-908`; `src/prefill/prefill.cpp`
(batch append and staging).

## Configuration

- CLI: `--kv fp16|int8|q4_0|k8v4` (default fp16), `--kv-resident N` (stream with N cells resident per layer,
  minimum 20,480; not with `k8v4`) (`src/program/generate.cpp:586-594`).
- Environment: `STRATA_KV_ROT=1` (rotated INT8, read by `src/program/generate.cpp:1822`); `STRATA_KV_HOST_DMA=1`
  (read by `src/prefill/prefill.cpp:2309`; the HIP A/B unstaging path).

## Model-specific behaviour

Both models use 2 KV heads x 256, so every format applies unchanged. Streaming saves no VRAM for Qwen3.6's dense
attention, because every block is selected every token and the resolve degenerates into copying the whole KV each
token (`docs/QWEN36-35B-A3B.md` section 3.3 item 6). Flash-Next's sparse selection reads at most 2,051 cells per
token.

## Tests

`kv_q8_parity` (INT8 vs a host reference and vs FP16), `kv_q4_parity` (rotated Q4_0 vs a host reference),
`kv_hybrid_parity` (K8V4 append, decode and prompt attention vs host references), `kv_stream_parity` (streamed vs
resident attention, bitwise), and `qsa_dense_parity` (every format under dense attention). All are synthetic and
need no model.

## Gotchas

- The q4_0 nibble order is split-half (j and j + 16 in one byte), not interleaved.
- The resolve block assumes `n_slots >= 1024`; fewer would let two threads take the same slot (`kv_stream.cu:212-219`).
  The engine never streams with fewer than 5,120 slots.
- `kv_block_bytes` requires a block's scale run to be a multiple of 16 bytes, which `kv_stream_resolve` checks.
- The measurements cited in these headers (`bench/results/2026-09-27-kv-q4`, `-p7-int-audit`) and
  `docs/kv-streaming-design.md` are not in this repository.

## See also

- [attention-qsa.md](attention-qsa.md) (the readers, `QsaStep`), [bits-helpers.md](bits-helpers.md) (`f16_from_f32`)
- `docs/upstream/DETAILS.md`, "How it works" (KV from 64K streaming from RAM)
