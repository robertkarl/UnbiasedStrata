[Codebase docs](../../README.md) › [Core](../README.md) › Experts

# Expert management

This part of the core engine decides where each routed MoE expert's bytes live and which processor computes them. The
host loop and the verify window (`src/core/session.cpp`, `src/core/verify.cpp`, driven from `src/program/generate.cpp`)
hand it, per layer, the normed activation, the router's top-k expert ids and an output buffer. It answers with one row
per routed entry. To do that it looks up the VRAM expert cache (`ExpertCache`), plans which experts the GPU computes
(VRAM hits and a PCIe share of the misses), and runs the rest on the CPU expert pool (`strata::kernels::cpu::ExpertPool`).
The pool reads its blobs through an `ExpertSource`: a resident pinned RAM arena by default, or mapped files with an
optional RAM copy. Other parts of this area keep the VRAM tier following the conversation (adaptive swaps), add tiers
on more GPUs, read and write the expert profile, and report hit counters to the server's `/health`.

## Contents

| Page | What it covers |
|---|---|
| [expert-sources.md](expert-sources.md) | `ExpertSource` and its two implementations: the pinned RAM arena (`ArenaExpertSource`) and the mapped or GGUF-in-place file tier (`FileExpertSource`), plus routing-aware prefetch, unbuffered reads, retrying reads (`robust_read.hpp`) and the host-RAM probe |
| [vram-cache.md](vram-cache.md) | `ExpertCache`: VRAM slots, the residency table, admission policies, sized and segmented (`--vram-elastic`) arenas, fill and verify. Also the expert profile reader and writer and `--expert-cache auto` sizing |
| [dispatch.md](dispatch.md) | `ExpertDispatch` and the three dispatch entry points: the single-token pool, the verify-window pool with its GPU plan (VRAM, PCIe, peer), and the R4.2c hit hook (`hit_hook.hpp`). Also the PCIe fraction, the hit counters, `/health` STATS and the watchdog's `progress.hpp` |
| [resident-ram-and-adaptive.md](resident-ram-and-adaptive.md) | The resident RAM mode (`pin_cache_complement`: compact copy, lend region, RAM budget, partial pin), adaptive swapping between VRAM and RAM, and `--expert-profile-save` |
| [multi-gpu-tiers.md](multi-gpu-tiers.md) | Extra expert tiers on other GPUs: `PeerExperts` (`--peer-device`), `RemoteExperts` (`--expert-cache-device1..3`) and `RemoteExpertOpt` (`--remote-expert-opt`) |
| [pack-format.md](pack-format.md) | The pack files the expert code reads: `experts.bin`, `native_experts.txt` (v2 to v4), `index.txt`, `conversions.json`, `expert-profile.bin`. Also the pack and GGUF tests in scope |

## Source map

| File | Lines | Role |
|---|---:|---|
| [include/strata/core/expert_source.hpp](../../../../include/strata/core/expert_source.hpp) | 727 | `ExpertSource` interface, `RouterLookahead`, `GpuPlanSink`, `ExpertDispatch`, dispatch entry points, `FileExpertSource`, `ArenaExpertSource`, GGUF expert checks and loader, `detail::` RAM and complement helpers |
| [src/core/expert_source.cpp](../../../../src/core/expert_source.cpp) | 3046 | Implementation of everything above: host-RAM probe, file tier, GGUF-in-place staging, resident RAM copy, CPU dispatch and GPU plan, hit path, arena load |
| [include/strata/core/expert_cache.hpp](../../../../include/strata/core/expert_cache.hpp) | 227 | `ExpertCache` (VRAM slots plus residency table) and the profile read, rank and write functions |
| [src/core/expert_cache.cpp](../../../../src/core/expert_cache.cpp) | 696 | Slot allocation (one `cudaMalloc`, sized, or CUDA VMM segments), admission, fills, read-back verification, profile file I/O |
| [include/strata/core/hit_hook.hpp](../../../../include/strata/core/hit_hook.hpp) | 41 | `HitPhase` / `HitFn`: the seam between the host loop and the VRAM hit path |
| [include/strata/core/progress.hpp](../../../../include/strata/core/progress.hpp) | 59 | Process-wide `Progress` heartbeat and stage for the `--serve` watchdog; diagnostic hooks the pool registers |
| [include/strata/core/peer_experts.hpp](../../../../include/strata/core/peer_experts.hpp) | 106 | `PeerExperts`: an adaptive expert tier on a second GPU |
| [src/core/peer_experts.cpp](../../../../src/core/peer_experts.cpp) | 325 | Peer tier open, launch (native grouped kernel plus optional scatter), finish, adapt |
| [include/strata/core/remote_experts.hpp](../../../../include/strata/core/remote_experts.hpp) | 89 | `RemoteExperts`: a static, profile-filled expert tier on CUDA1..3 |
| [src/core/remote_experts.cpp](../../../../src/core/remote_experts.cpp) | 341 | Remote tier preflight, open, begin (grouped launch), finish |
| [include/strata/core/remote_expert_opt.hpp](../../../../include/strata/core/remote_expert_opt.hpp) | 53 | `RemoteExpertOpt`: optional decode optimisation over the remote tiers |
| [src/core/remote_expert_opt.cu](../../../../src/core/remote_expert_opt.cu) | 194 | Weighted reduction on the helper GPU, CPU-row copy that skips remote rows, adaptive swaps for remote caches |
| [include/strata/platform/robust_read.hpp](../../../../include/strata/platform/robust_read.hpp) | 181 | `ReadGate`, `read_at`, `read_range`: retrying positional reads (POSIX only). Used here by `load_experts_gguf` and also by `src/kernels/ngram.cpp` |
| [tests/core/expert_cache_per_layer_test.cpp](../../../../tests/core/expert_cache_per_layer_test.cpp) | 44 | Per-layer admission on uniform and sized caches (CUDA) |
| [tests/core/expert_cache_segmented_test.cpp](../../../../tests/core/expert_cache_segmented_test.cpp) | 99 | Segmented cache shrink and grow keep slot addresses and bytes (CUDA VMM) |
| [tests/core/file_expert_source_test.cpp](../../../../tests/core/file_expert_source_test.cpp) | 360 | File tier size checks, canonical and native layouts, complement plan, lend region, exchanges, cgroup and host-RAM probe |
| [tests/core/expert_profile_save_test.cpp](../../../../tests/core/expert_profile_save_test.cpp) | 92 | `rank_learned_profile` order and the profile file format round trip (CPU only) |
| [tests/core/expert_layout_test.cpp](../../../../tests/core/expert_layout_test.cpp) | 412 | `native_experts.txt` v2 to v4 parsing, plus `check_experts_gguf` and `load_experts_gguf` on synthetic shards |
| [tests/core/pack_load_check.cpp](../../../../tests/core/pack_load_check.cpp) | 226 | Loads a real native pack as `generate --native --pack` would, up to the first token (skips without a model) |
| [tests/core/pool_native_widths_test.cpp](../../../../tests/core/pool_native_widths_test.cpp) | 125 | The CPU pool's native path uses each layer's own widths (Qwen3.6 2048 x 512) |
| [tests/core/gguf_split_test.cpp](../../../../tests/core/gguf_split_test.cpp) | 220 | Split-GGUF shard discovery and joint view on synthetic files |
| [tests/core/native_dense_ple_key_test.cpp](../../../../tests/core/native_dense_ple_key_test.cpp) | 169 | A native pack's BF16 PLE key row stays in the dense arena (#326) |
| [tests/core/gguf_fixture.hpp](../../../../tests/core/gguf_fixture.hpp) | 113 | Writes synthetic GGUF v3 files for the loader tests |

Related files outside this area: [include/strata/kernels/cpu/expert_layout.hpp](../../../../include/strata/kernels/cpu/expert_layout.hpp)
(87) and [src/kernels/cpu/expert_layout.cpp](../../../../src/kernels/cpu/expert_layout.cpp) (349) parse `native_experts.txt`
into the process-wide `ExpertLayout` that every class here asks for blob sizes and offsets.
[tools/make_profile.py](../../../../tools/make_profile.py) and [tools/iq_pack.py](../../../../tools/iq_pack.py) write the
profile and the pack.

## Key concepts

**Blob.** One expert of one layer, stored as `[gate rows | up rows | down rows]`. A canonical Q2_0 pack (Flash-Next
only) uses one size for every layer, `BLOB = 1,382,400` bytes. A native pack (`tools/iq_pack.py`; every Qwen3.6 pack)
keeps the raw GGUF slices, so the blob size and quant types change per layer. `ExpertLayout::blob_bytes(layer)` and
`blob_offset(layer, expert)` (`include/strata/kernels/cpu/expert_layout.hpp:35-41`) are the single source of truth.

**Index.** A `(layer, expert)` pair is indexed as `layer * n_expert + expert` everywhere: the residency table, the
complement offsets, the usage counts and the profile's rank table. Flash-Next has 48 x 512 = 24,576 pairs. Qwen3.6-35B-A3B
has 40 x 256 trunk pairs (`tools/make_profile.py` docstring).

**The tiers.** Which tiers are in use depends on the start mode:

```
                     routed (layer, expert)
                              |
          +-------------------+--------------------------------------------+
          | ExpertCache: VRAM slots, residency table host_res (slot or -1) |  hit  -> GPU grouped kernel
          +-------------------+--------------------------------------------+
                              | miss
          +-------------------v--------------------------------------------+
          | other GPU tiers (optional): PeerExperts / RemoteExperts        |  held -> that GPU
          +-------------------+--------------------------------------------+
                              | miss
          +-------------------v--------------------------------------------+
          | host RAM                                                       |
          |  default:  ArenaExpertSource - every expert, pinned/locked RAM |  -> CPU pool, or the GPU reads
          |  --mmap-experts + --resident-*: FileExpertSource RAM copy      |     a PCIe share (pinned only)
          |            (the experts no GPU slot holds, or the hottest of   |
          |            them under --resident-budget-gib)                   |
          +-------------------+--------------------------------------------+
                              | not in RAM (mmap modes only)
          +-------------------v--------------------------------------------+
          | files: experts.bin mapped, or the GGUF shards in place;        |  -> page faults / staged
          |  OS page cache, warm hints, unbuffered reads (Windows)          |     reads, then the CPU pool
          +----------------------------------------------------------------+
```

- **VRAM.** `ExpertCache` holds `slots()` blobs. In the verify window and the token graph, residency is a static
  host table `host_res` (`n_layers x n_expert`, slot or -1) that is mirrored on the device. It is filled once from the
  expert profile at start-up, then changed only by adaptive swaps between verify rounds. See [vram-cache.md](vram-cache.md).
- **Pinned RAM.** By default (`ArenaExpertSource`) all experts are read into one `PinnedArena` at start-up. The CPU
  computes misses in place, and when the arena is registered the GPU can read misses over PCIe. With `--mmap-experts`
  plus `--resident-experts` / `--resident-cpu-experts`, `FileExpertSource::pin_cache_complement` builds a compact RAM copy
  of only the experts that no GPU slot holds. See [expert-sources.md](expert-sources.md) and [resident-ram-and-adaptive.md](resident-ram-and-adaptive.md).
- **Files.** Only in the mmap modes. A miss that is in neither VRAM nor the RAM copy is read from the mapped
  `experts.bin`, or assembled from the three GGUF role slices into a staging buffer (`staged_blob`). It is warmed ahead
  by `RouterLookahead`, and on Windows it can be read unbuffered.

**Who computes a miss.** In a verify window, `expert_pool_dispatch_multi` gives each routed entry a kind:
`0` VRAM hit, `1` PCIe (the GPU reads a pinned host blob), `2` another GPU, `-1` CPU. The last `pcie_num/256` of a layer's
distinct missed experts (`--pcie-frac`) are given to the GPU over PCIe when their blobs are pinned. See [dispatch.md](dispatch.md).

**Profile.** `expert-profile.bin` (`STRP` format) ranks `(layer, expert)` pairs by routing frequency. The VRAM tier is
filled in rank order (`generate.cpp:3802-3843`). The RAM budget mode keeps the next-hottest pairs in RAM. The extra GPU
tiers take the next ranks that the primary does not hold. `--expert-profile-save` writes what the adaptive tier learned
back in the same format.

**Adaptive swapping.** Between verify rounds, every `--adapt-every` rounds (default 4), the serve loop swaps up to
`--adapt-swaps` (default 96) of the most-routed non-resident experts into the slots of the least-routed resident ones of
the same layer. It uses decayed routing counts (`ExpertDispatch::usage`, decay `--adapt-decay` 0.7). See
[resident-ram-and-adaptive.md](resident-ram-and-adaptive.md).

## Data flow of one verify-window layer

```
verify graph publishes ids  -->  expert_pool_dispatch_multi (pool thread, src/core/expert_source.cpp:2011)
  1. usage[layer][e] += 1 per routed entry                       (adaptive tier counts)
  2. distinct experts -> kind 0 (host_res slot) / 2 (peer) / 1 (last pcie_num/256 of misses, if pinned) / -1
  3. GpuPlanSink filled (VRAM groups, then PCIe groups); publish(); fetch() starts DMA of PCIe blobs
  4. peer->launch(), remote[r]->begin()                           (other GPUs start)
  5. CPU: quantize activations, src->prefetch(misses), one job per distinct missed expert, pool->run_split_multi*
  6. remote[r]->finish(), peer->finish()                          (rows land in the same mapped output)
  7. counters: cache_hits, cache_refused, offload_entries, peer_entries, multi_misses ...
```

## Tests at a glance

| Test (CTest name) | Needs | Covers |
|---|---|---|
| `file_expert_source_test` | CPU (links `strata_engine`) | [expert-sources.md](expert-sources.md), [resident-ram-and-adaptive.md](resident-ram-and-adaptive.md) |
| `expert_profile_save_test` | CPU | [vram-cache.md](vram-cache.md) |
| `expert_cache_per_layer_test`, `expert_cache_segmented_test` | CUDA device (exit 77 = skipped) | [vram-cache.md](vram-cache.md) |
| `expert_layout_test`, `pool_native_widths_test`, `native_dense_ple_key_test`, `pack_load_check` | `STRATA_BUILD_TESTS` and `STRATA_NATIVE_EXPERTS`; some need CUDA or a model | [pack-format.md](pack-format.md) |
| `gguf_split_test` | `STRATA_BUILD_TESTS` | [pack-format.md](pack-format.md) |

Run them with `ctest -R <name>` from the build directory. CMake registers them in `CMakeLists.txt:456-503` and `:231-233`.

## See also

- [../conversation/README.md](../conversation/README.md): the prompt and conversation cache, which shares the serve loop
  with the adaptive tier.
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md), "How it works": upstream's summary of the memory tiers.
- [docs/qwen36-port/step3-experts.md](../../../qwen36-port/step3-experts.md): the native GPU expert kernels' format coverage.
- [docs/qwen36-port/step4-pack.md](../../../qwen36-port/step4-pack.md): building a Qwen3.6 native pack.
- [docs/qwen36-port/geometry-inventory.md](../../../qwen36-port/geometry-inventory.md): rows 108-118 cover the expert layout and cache.
