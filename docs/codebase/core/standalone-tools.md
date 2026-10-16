[Codebase docs](../README.md) › [Core](README.md) › Standalone tools

# Standalone core tools

## Purpose

Four small executables built from `src/core/*_main.cpp` with the core library. They predate the serving engine
(Phase 2 bring-up) and answer one question each about the machine: is the GPU usable and does the memory plan fit,
how fast does the expert arena load, do the CPU and the PCIe stream compete for DRAM, and does streaming an expert
overlap computing on one. None of them loads a model; three need an `experts.bin` file.

Files covered: [src/core/device_main.cpp](../../../src/core/device_main.cpp),
[src/core/load_main.cpp](../../../src/core/load_main.cpp), [src/core/concurrent_main.cpp](../../../src/core/concurrent_main.cpp),
[src/core/overlap_main.cpp](../../../src/core/overlap_main.cpp).

Built only with `STRATA_ENABLE_CUDA` or `STRATA_ENABLE_HIP` (`CMakeLists.txt`, after `strata_core`). `strata-device`
links `strata_core`; the other three also link `strata_kernels`.

## `strata-device` (`device_main.cpp`)

```
strata-device [--selftest] [--list-devices]
```

* Default: `device_info(0)` (throws if the card cannot run this build), then name, compute capability or HIP arch,
  SM count, total/free VRAM, driver/runtime versions (and on Windows HIP the loaded `amdhip64_7.dll` path). Then it
  prints the **Phase-1 planner** `strata::plan::make_plan(20480, Geometry{}, Costs{})` from
  `include/strata/plan/plan.hpp` against the free VRAM, with `FITS` / `*** DOES NOT FIT ***`. (This is the older
  Flash-Next plan, not [vram-plan.md](vram-plan.md).)
* `--list-devices`: every device the runtime enumerates, numbered as `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`
  number them, with `cannot run: <why>` for cards the binary has no code for. The comment documents the format that
  `setup.py` parses (#325).
* `--selftest`: a 64 MiB poisoned `DeviceArena`, two aligned allocations, and a check that over-allocation throws.

## `strata-load` (`load_main.cpp`)

```
strata-load --file experts.bin [--layers N] [--threads N] [--chunk-mb N] [--no-pin] [--stream]
```

Loads `layers x 512 x 1,382,400 B` (Flash-Next's canonical Q2_0 blob; default 2 layers) into a `PinnedArena` (or
`malloc` with `--no-pin`) with `load_experts` (default 8 threads, 16 MiB chunks), prints GiB/s and per-layer FNV-1a
checksums. `--stream` adds `stream_bandwidth` at 1 blob, 16 blobs, 1 layer and the whole arena, and a per-token
projection at the design's 663.6 MB of experts per token. The layer count is a flag so a full 34 GB load can be
projected from a smaller one without evicting the page cache.

## `strata-concurrent` (`concurrent_main.cpp`)

```
strata-concurrent --file experts.bin [--experts N] [--cpu-threads N] [--gpu-roles N]
```

Measures whether the CPU expert path and the GPU stream path are independent resources. It fills a pinned arena
with real Q2_0 code planes, then runs three modes in one process - CPU alone (a multi-threaded **read loop** over the
arena, deliberately not the real kernel, to measure DRAM demand), GPU alone (double-buffered H2D copies plus
`s_gemv_split_async` per role), and both - and prints each side's throughput loss when the other runs.

## `strata-overlap` (`overlap_main.cpp`)

```
strata-overlap --file experts.bin [--experts N] [--tpr N]
```

Measures at one expert's granularity whether copying an expert (H2D on one stream) overlaps computing a GEMV on the
previous one (another stream): a serial baseline and a double-buffered loop in the same process; prints the ratio
(1.00 = no overlap).

## Configuration

Only their own flags; the `PinnedArena` they use honours `STRATA_NO_LARGEPAGES`, `STRATA_ARENA_PIN_GIB` and
`STRATA_ARENA_LOCK` ([device-and-memory.md](device-and-memory.md)).

## Model-specific behaviour

The constants (1,382,400-byte blobs, 512 per layer, 2560 x 640 roles) are Flash-Next's canonical Q2_0 layout; these
tools have not been adapted to Qwen3.6's native blobs.

## Tests

Not registered with CTest; `strata-device --selftest` is a self-check.

## Gotchas

* Their numbers in the comments (bandwidths, ledger entries L3-L9) are historical measurements on the upstream
  author's machine.

## See also

* [device-and-memory.md](device-and-memory.md), [vram-plan.md](vram-plan.md).
