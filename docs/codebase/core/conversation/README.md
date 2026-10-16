[Codebase docs](../../README.md) › [Core](../README.md) › Conversation state

# Conversation state and the prompt cache

This part of the core engine lets a `--serve` request skip re-reading the part of its prompt the engine has already
seen. It has two layers:

1. **Checkpoints within the session.** Each checkpoint is a copy of the model's *running* state after a prompt
   prefix: the gated DeltaNet (GDN) recurrence and convolution state, the PLE history, and the QSA indexer's partial
   rows. The KV cache is positional and stays in place, so a checkpoint is valid only while the cells below it are
   untouched.
2. **Parked conversations** (opt-in, `--conversation-cache-mib`). A whole session image, including every attention
   layer's K/V and the draft layer's ring, held in host RAM. Several independent conversations can then alternate
   without re-reading each other's prompts.

The serve loop in `src/program/generate.cpp` owns the policy: when to checkpoint, which source to resume from, when to
park. This area provides the data types, the prefix matching, the RAM budget and admission, and the
save / validate / restore primitives. Those primitives copy between `SessionState` / `QsaState` device buffers (from
`src/core/session.cpp` and the QSA kernels) and host memory.

## Contents

| Page | What it covers |
|---|---|
| [running-state.md](running-state.md) | `ConversationCheckpoint` and `conversation_state.cpp`: what a checkpoint holds (GDN state and conv state, PLE history, indexer tail / dead row / block position), its sizes, save / validate / restore, whole-session snapshot save, validate and restore, the geometry key. Also how the serve loop builds and prunes the checkpoint chain |
| [kv-images.md](kv-images.md) | `conversation_snapshot.cpp`: K/V capture per attention layer in every KV format (F16, INT8, Q4_0, hybrid K8V4, rotated), streaming and draft-ring residency, incremental re-capture from retained pages, read-back verification |
| [conversation-cache.md](conversation-cache.md) | `ConversationCache` (parked images, prefix matching, eviction, superseded copies, retained K/V), `ConversationBuffer` (segmented host storage), `conversation_memory` (physical-RAM admission), layer-split split/merge, and the in-session retention policy in `conv_cache.hpp` |

## Source map

| File | Lines | Role |
|---|---:|---|
| [include/strata/core/conversation_cache.hpp](../../../../include/strata/core/conversation_cache.hpp) | 283 | Data types (`ConversationCheckpoint`, `ConversationKv`, `ConversationKvReuse`, `SavedConversation`), `conversation_prefix`, layer-split split and merge, `ConversationCache` |
| [include/strata/core/conversation_snapshot.hpp](../../../../include/strata/core/conversation_snapshot.hpp) | 96 | Save / validate / restore API for checkpoints, K/V images and whole snapshots |
| [include/strata/core/conversation_buffer.hpp](../../../../include/strata/core/conversation_buffer.hpp) | 154 | `ConversationBuffer`: segmented (16 MiB) host byte storage with allocation-peak prediction |
| [include/strata/core/conversation_memory.hpp](../../../../include/strata/core/conversation_memory.hpp) | 19 | Physical RAM telemetry and the admission predicate |
| [src/core/conversation_state.cpp](../../../../src/core/conversation_state.cpp) | 340 | Running-state sizes, checkpoint save / validate / restore, snapshot bytes / save / validate / restore |
| [src/core/conversation_snapshot.cpp](../../../../src/core/conversation_snapshot.cpp) | 269 | Per-layer K/V layout, capture, validation, restore, verify |
| [src/core/conversation_memory.cpp](../../../../src/core/conversation_memory.cpp) | 51 | `MemAvailable` parsing (Linux), `GlobalMemoryStatusEx` (Windows) |
| [src/core/conversation_checked.hpp](../../../../src/core/conversation_checked.hpp) | 24 | Overflow-checked `add` / `product` used by every size computation |
| [src/core/conversation_cache_test.cpp](../../../../src/core/conversation_cache_test.cpp) | 305 | Buffer segmentation, budget accounting, matching, eviction, retained K/V, images, split/merge (CPU) |
| [src/core/conversation_memory_test.cpp](../../../../src/core/conversation_memory_test.cpp) | 66 | `MemAvailable` parsing and admission arithmetic (CPU) |
| [src/core/conversation_snapshot_test.cpp](../../../../src/core/conversation_snapshot_test.cpp) | 324 | Full capture / restore round trips on a real device in every KV format and mode (CUDA) |
| [src/core/conversation_validation_test.cpp](../../../../src/core/conversation_validation_test.cpp) | 239 | Invalid images rejected before any CUDA call. With `CONVERSATION_TEST_TRANSFERS`, fault injection into the restore (host only) |

Related, outside this directory: [include/strata/program/conv_cache.hpp](../../../../include/strata/program/conv_cache.hpp)
(43, the checkpoint chain's eviction victim) and its test [src/program/conv_cache_test.cpp](../../../../src/program/conv_cache_test.cpp) (67).

## Key concepts

```
 request tokens:  [system prompt ......|turn 1 .......|reply 1 .....|turn 2 ...........|<|im_start|>assistant]
                                       ^ root ckpt     ^ turn ckpt                      ^ last token: always read
 sources tried (generate.cpp:6812-6846), longest prefix wins:
   live session (what the session holds after the last request)  -> nothing to restore
   in-session checkpoints `checks` (running state only)          -> conversation_checkpoint_restore
   an idle --batch slot's session or checkpoint                  -> copy_from_slot
   a parked conversation (full image incl. K/V)                  -> conversation_snapshot_restore
```

- **What is snapshotted.**
  - A checkpoint holds the token ids, image keys, `gdn` (per GDN layer: the recurrent state
    `ssm_state_size x ssm_v_heads x ssm_state_size` floats **plus** the conv state `ssm_conv_channels x (ssm_d_conv - 1)`
    floats), `ple` (the n-gram/PLE history, `NG_HIST x NG_HC_DIM` floats, Flash-Next only), and per QSA layer the
    indexer's `tails`, `dead` row and `block_pos`.
  - A parked image adds a `ConversationKv` per owned attention layer plus the draft layer: K, V, their scales, and the
    indexer's pooled rows.
- **Prefix matching** (`conversation_prefix`, `conversation_cache.hpp:144-157`). A checkpoint of `n` tokens matches a
  prompt only if:
  - `n < prompt.size()`, because the last prompt token always starts the next verify window;
  - its ids equal the prompt's first `n`;
  - the images starting below `n` equal the checkpoint's, in order, by `(start, hash)`.

  Parked entries must also match the control-vector ("steering") mode.
- **Memory budget.**
  - In-session checkpoints: at most `--prompt-cache N` (default 6). By the code's own estimate each is about 118 MB on
    Flash-Next, which matches `36 GDN layers x (128*48*128 + 10240*3) floats x 4 B`. The same formula with
    Qwen3.6's values from the comments in `include/strata/core/layout.hpp:50-52` (32 v-heads, 8192 conv channels, 30
    GDN layers) gives about 66 MB; that figure is derived here, not measured.
  - Parked conversations: `--conversation-cache-mib` bytes (default 0 = off) and `--conversation-cache-slots` entries
    (default 4). Before and after each capture, `conversation_memory_admit` also requires physical RAM to stay above
    `--conversation-cache-min-free-mib` (default 2560). Unknown telemetry fails closed.
- **Validation.**
  - Every restore validates the whole image first: an 18-value geometry key, the session's layer carve, the payload
    sizes, the checkpoints-are-prefixes rule, image key order and the K/V layout. All of this happens **before** any
    CUDA call or destination write.
  - An invalid image returns `ConversationRestore::invalid` and the session is untouched. The caller discards the image
    and falls back to a shorter source.
  - A copy failure after that returns `transfer_failed`. The caller must not continue from that session, and the serve
    loop answers `ERR` and exits.

## Data flow of one serve request (simplified)

```
match sources -> (park the outgoing conversation if switching) -> restore chosen source
             -> prune `checks` to prefixes of the new prompt (cells past `resume` will be rewritten)
             -> read prompt [resume, n-1) in parts, checkpoint at root_at, turn_at, every --prompt-cache-every
             -> verify windows / decode -> live = consumed tokens, live_ok
```

## Tests and how to run them

| CTest name | Built when | Needs |
|---|---|---|
| `conversation_cache_test`, `conversation_memory_test` | `-DSTRATA_BUILD_CONVERSATION_TESTS=ON` (`CMakeLists.txt:93-104`) | CPU only, no engine library |
| `conversation_snapshot_test` | the same option, inside the engine block (`:542-545`) | CUDA device (exit 77 = skipped) |
| `conversation_validation_test` | the same | Runs with `CUDA_VISIBLE_DEVICES=-1`: must never initialise CUDA |
| `conversation_transfer_test` | the same, on GNU/Clang Linux without HIP (`:550-557`) | `-Wl,--wrap=cudaMemcpy,cudaDeviceSynchronize,cudaGetLastError` fault injection |
| `conv_cache_test` | `STRATA_BUILD_TESTS` or the conversation option (`:1064-1069`) | CPU |

Then run `ctest -R conversation` or `ctest -R conv_cache` from the build directory. Upstream also had Python harnesses
(`tools/test_conversation_cache_*.py`, named in [docs/qwen36-port/upstream-tooling.md](../../../qwen36-port/upstream-tooling.md));
they are not in this tree's `tools/`.

## See also

- [../experts/README.md](../experts/README.md): expert management. The serve loop interleaves the adaptive expert tier
  with these requests.
- [docs/qwen36-port/step5-serve.md](../../../qwen36-port/step5-serve.md): the Qwen3.6 serve port, including the
  checkpoint geometry fix (row 2) and why reuse between chat turns is short (section d).
- [docs/qwen36-port/step3-attn.md](../../../qwen36-port/step3-attn.md): the attention notes, which mention the session
  save and restore.
- `README.md` (repository root), "Known problems": Qwen3.6's template re-renders earlier replies, so reuse stops at the
  previous prompt.
