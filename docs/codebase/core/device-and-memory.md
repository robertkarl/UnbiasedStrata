[Codebase docs](../README.md) › [Core](README.md) › Device and memory helpers

# Device and memory helpers

## Purpose

Low-level runtime pieces the rest of the core and the expert side build on: device discovery and the run-time
architecture check, a bump-allocated VRAM arena, the page-locked host arena the expert weights live in (with its
loaders and a bus benchmark), a scoped device switch for multi-GPU objects, and an emulation switch for testing
another GPU generation's code paths.

Files covered:

* [include/strata/core/device.hpp](../../../include/strata/core/device.hpp), [src/core/device.cu](../../../src/core/device.cu)
* [include/strata/core/pinned.hpp](../../../include/strata/core/pinned.hpp), [src/core/pinned.cu](../../../src/core/pinned.cu), [src/core/pinned_shared_test.cpp](../../../src/core/pinned_shared_test.cpp)
* [include/strata/core/on_device.hpp](../../../include/strata/core/on_device.hpp)
* [include/strata/core/emulate.hpp](../../../include/strata/core/emulate.hpp)

## Device (`device.hpp` / `device.cu`)

| Item | Line | What it does |
| --- | --- | --- |
| `struct DeviceInfo` | `device.hpp:21` | ordinal, name, compute capability, total/free bytes, driver/runtime versions, SM count, HIP arch name |
| `device_info(ordinal)` | `device.hpp:54` | Fills `DeviceInfo` and **throws `CudaError`** on a card this build cannot run: CUDA needs compute capability >= 7.5, or >= 6.0 in the `STRATA_EXPERIMENTAL_SM60` build (Pascal/Volta, #236); HIP builds check the card's `gcnArchName` against the compiled `STRATA_HIP_ARCHS` and wave32; the gfx906 compat build records the arch |
| `device_count()`, `device_summary()` | `device.hpp:44`, `:49` | Enumerate and describe every card without the arch check (for `strata-device --list-devices`) |
| `gpu_arch_problem(ordinal)`, `compiled_gpu_archs()` | `device.hpp:36`, `:39` | HIP: why a card cannot run this binary / which archs it was built for; empty on CUDA |
| `device_code_error()` | `device.hpp:58` | Empty when the build has device code for the current device (CUDA); HIP returns "" because the arch check ran earlier |
| `class CudaError` | `device.hpp:60` | `std::runtime_error` carrying the CUDA error code |
| `class DeviceArena` | `device.hpp:72` | One `cudaMalloc` with bump sub-allocation (power-of-two alignment, default 256) and no frees; `poison` fills new memory with quiet NaNs so uninitialised reads are visible; over-allocation throws |

`DeviceArena` is used only by `strata-device --selftest` in this checkout; the engine's weight arena and session arena
are plain `cudaMalloc` regions carved by their own code ([weights-and-layout.md](weights-and-layout.md),
[session.md](session.md)).

Windows HIP only (`device.cu`, `strata::hip_compat::mem_get_info`): `hipMemGetInfo` reports the card minus this
process's own allocations, so the free figure is lowered by what the Windows per-process video-memory budget (DXGI
`QueryVideoMemoryInfo`) withholds for other programs; `STRATA_WDDM_BUDGET=0` turns that off.

## Pinned host arena (`pinned.hpp` / `pinned.cu`)

`struct PinnedArena` (`pinned.hpp:30`) is the host memory the expert weights live in. Construction
(`pinned.cu:331`):

1. **Reserve** (`reserve`, `pinned.cu:58`): on Linux try `MAP_HUGETLB | MAP_HUGE_2MB` (all-or-nothing; on failure the
   note names the 2 MiB pages needed against `vm.nr_hugepages`), else a 2 MiB-aligned anonymous mapping with
   `MADV_HUGEPAGE` (transparent huge pages); on Windows `VirtualAlloc(MEM_LARGE_PAGES)` needs SeLockMemoryPrivilege
   and normally falls back to 4 KB pages. `STRATA_NO_LARGEPAGES=1` skips large pages. With `shared_file` (Linux), a
   `MAP_SHARED` file (intended for tmpfs, e.g. `/dev/shm`) whose first 4 KiB is a header (`STRATA-ARENA-V1`, arena
   size, pack hash) that must match, so several processes can share one populated arena.
2. **Register with CUDA before touching pages**: `cudaHostRegister(Portable | Mapped)` of the whole arena. If refused
   (or capped), register slice by slice from the start (`bounds`: uniform slices or one per layer for native packs);
   `registered_bytes` / `slice_starts` record the pinned prefix. The cap comes from the caller
   (`max_pinned_bytes`) or `STRATA_ARENA_PIN_GIB=N` (`auto` on Windows = below the GPU's shared-memory budget, #243).
3. Whatever is not registered is kept resident through the working-set lock (`platform::lock_resident`) unless
   `STRATA_ARENA_LOCK=0`. A failed registration's sticky CUDA error is consumed so it does not surface later in an
   unrelated kernel check.

`backing` (`LargePages`, `NormalPages`, `PinnedByCuda`) and `note` say what was obtained, for the start-up print.

Loaders and helpers:

| Function | Line | What it does |
| --- | --- | --- |
| `load_experts(path, dst, blob, blobs_per_layer, layers, threads, chunk)` | `pinned.hpp:101` | Parallel reader, one `FILE*` per thread (`fread`, not `ifstream`, because MSVC split reads into 4 KiB calls), per-layer FNV-1a checksums; `LoadStats.ok` is false on a short read |
| `load_experts_ranges(...)` | `pinned.hpp:104` | Same with one byte range per layer (native packs) |
| `load_experts_direct(...)` | `pinned.hpp:110` | Windows unbuffered reads straight into the arena (4 KiB alignment required; otherwise returns ok = false with an empty error and the caller falls back) |
| `experts_unbuffered(files, arena_bytes, why, ...)` | `pinned.hpp:123` | Decides unbuffered vs cached loading from a timed random-read probe and the RAM left beside the arena; `STRATA_UNBUFFERED_LOAD=1/0` forces it |
| `fnv1a64` | `pinned.hpp:128` | Per-layer checksum |
| `stream_bandwidth(src, bytes, chunk, iters)` | `pinned.hpp:150` | H2D bandwidth at a given transfer granularity (used by `strata-load --stream`) |
| `arena_pin_cap_gib()` | `pinned.hpp:28` | Parses `STRATA_ARENA_PIN_GIB` (-1 unset, 0 no cap, N GiB, -2 auto) |

Callers: the expert side (`src/core/expert_source.cpp`, around line 2874, creates the production arena; see
[experts/](experts/README.md)) and the standalone tools ([standalone-tools.md](standalone-tools.md)).

## `OnDevice` (`on_device.hpp:10`)

RAII: switch to `device` for a scope and restore the previous device on exit; a negative or already-current device
does nothing. `Verifier` and `MtpDrafter` record the device they were initialised on and wrap every public call in it,
so a layer split's stage on another GPU can be driven from any thread.

## `STRATA_EMULATE_CC` (`emulate.hpp`)

For tests only: with a build whose device code is PTX for an older architecture (e.g.
`CMAKE_CUDA_ARCHITECTURES=86-virtual`) running on a newer card, `STRATA_EMULATE_CC=75|80|86|89` makes host-side
queries answer as that card would: `emulated_cc()`, `cc_major_of`, `cc_minor_of`, and `smem_optin_of` (64 KB on
Turing, 163 KB on A100, at most 99 KB otherwise). Results are meaningful, speed is not.

## Configuration summary

| Variable / option | Effect |
| --- | --- |
| `STRATA_NO_LARGEPAGES=1` | skip hugetlb / large pages / THP request |
| `STRATA_ARENA_PIN_GIB=N|auto` | cap the arena's CUDA registration |
| `STRATA_ARENA_LOCK=0` | do not lock the unregistered part resident |
| `STRATA_UNBUFFERED_LOAD=1|0` | force unbuffered expert loading on/off |
| `STRATA_WDDM_BUDGET=0` | Windows HIP: do not lower the free VRAM figure by the DXGI budget |
| `STRATA_EMULATE_CC` | emulate another compute capability (tests) |
| CMake `STRATA_EXPERIMENTAL_SM60`, `STRATA_HIP_GFX906`, `STRATA_ENABLE_HIP` | change the run-time device checks |

## Tests

* `pinned_shared_test` (`src/core/pinned_shared_test.cpp`, Linux only, CTest): maps a shared-file arena twice with
  the same pack hash and checks that a byte written through the first is seen through the second, then checks that a
  different pack hash and a different arena size are refused.
* `strata-device --selftest` exercises `DeviceArena` (alignment, poison, refusal of over-allocation).
* `platform_memory_test` (`src/platform/memory_test.cpp`) covers the working-set lock (platform area).

## Gotchas

* `device_info` throws; callers that only want to list cards use `device_count` / `device_summary`.
* A partly registered arena is still correct, but copies from outside the pinned prefix are slower; the note says how
  much was pinned.
* The shared-file backing does not coordinate population between processes; that is the caller's job (header comment).

## See also

* [standalone-tools.md](standalone-tools.md), [experts/](experts/README.md), [vram-plan.md](vram-plan.md).
* [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) "Sharing the GPU with other programs" and "Windows" for user-facing settings.
