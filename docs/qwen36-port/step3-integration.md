# Qwen3.6 port, Phase 2 step 3: the integration branch `qwen36-step3`

Branched from the step-2 head (0e40d4e) on 2026-10-11. Five areas were developed in parallel from 8b7525e, each on its
own branch with its own notes, and cherry-picked here in this order with no conflicts: block (`step3-block.md`, 5
commits), gdn (`step3-gdn.md`, 8), experts (`step3-experts.md`, 7), attn (`step3-attn.md`, 9), prefill
(`step3-prefill.md`, 12 of 17: its five cherry-picks of attn and gdn commits were dropped here because the originals
are present). One commit of the integrator's own, 7cfb7dc: `generate.cpp` calls `qsa_kv_streaming_applies` right after
the geometry is read (the attention agent could not edit that file).

## Both cards, the whole branch

Configure for both: `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON
-DSTRATA_MMQ_KQUANTS=ON` with gcc-13 as the host compiler; the 1080 Ti adds `-DSTRATA_EXPERIMENTAL_SM60=ON
-DCMAKE_CUDA_ARCHITECTURES=61`, the 4090 `-DCMAKE_CUDA_ARCHITECTURES=89`. `STRATA_MMQ_KQUANTS` is needed for the
Q4_K_XL file's prompt path (Q6_K down on layers 34, 38, 39) and defaults OFF; step 5 turns it on in the Makefile.

| Card | Build | ctest | Failures |
| --- | --- | --- | --- |
| RTX 4090 (VM 221, sm_89) | 0 errors, 420 s for every target | 89 of 92 pass | ple_parity (no Flash-Next shard on the host), platform_memory_test (mlock limit), expert_multi_test (no AVX-512) |
| GTX 1080 Ti (VM 211, sm_61) | 0 errors, 302 s | 87 of 92 pass, 2 skipped (the sm_80 fused prompt MoE tests) | the same three |

Every step-3 test (qwen36_block_test, qwen36_seam_parity, gdn_parity, router_top10_parity, shared_expert_parity,
qsa_dense_parity, the native_expert_parity pairs, prefill_mmq_kquant_test, gdn_rec_parity, prefill_qwen36_test) passes
on both cards; the per-seam numbers are in each area's notes and are identical on both cards to the printed digits.

## The review (`step3-review`, read-only, on the 41-commit range) and what it changed

1. HIGH, attn x block seam: `shapes_of` in verify.cpp and mtp.cpp left `idx_top_k = 2048` on a no-indexer geometry, a
   "half set up" shape that `qsa.cu`'s validate refuses, so the verify window would have exited on the first Qwen3.6
   attention layer. Not covered by any test (no test drives `Verifier` on qwen35moe; that is step 4's runner test).
   Fix: one exported shape builder for all four sites and one dense predicate.
2. MEDIUM, Flash-Next: the MTP drafter's attention (cap = the 32,768 draft window) took the new bounded split-K kernel,
   a different summation order; drafts are verified, so draft quality, not correctness. Fix: sparse-model shapes keep
   the previous grid.
3. LOW: the prefill fixture failed instead of skipping without numpy (fixed, c4f599f); native_expert_parity's 3e-2
   bound tightened to the policy's 2e-2 with separate CPU rows (77983af); `qsa_attn_bench` registered as a test
   (removed from the suite).

All fixed on the attention branch and cherry-picked here (one builder `strata::core::qsa_shapes(g)` in
`include/strata/core/qsa_shapes.hpp` used by layer, verify, mtp, session and prefill, the last by delegation from
prefill's `shapes_of`; `qsa_dense()` and the kernels' guard test the same three-zero predicate; the split-K bound applies
to dense shapes only, so Flash-Next's drafter keeps its 512-block grid and 12.09 MiB scratch, A/B equal within 1 us on
the 1080 Ti). After the fixes: 4090 88 of 91 (the same three host failures); 1080 Ti incremental build 81 s, the 33
attention / prefill / expert / verify tests 33 of 33.
4. NOTE: the GDN head pairing is tiled (`h % 16`), as llama.cpp, the oracle and the engine all have it; the reviewer's
   brief had said interleaved, which is the oracle's negative control.

Not checked end to end until step 4's pack and runner exist: the verify window on qwen35moe, the decode mixers through
`block_layer_pre` stage 1 on a real pack, `native_rope_apply` against the Qwen3.6 oracle, dense attention above 20,000
cells against a reference, the drafter on qwen35moe.

## Findings that feed step 4 and 5

- The real files' F32 routers, `ffn_gate_inp_shexp`, `ssm_alpha`, `ssm_beta` and `ssm_conv1d` are BF16-exact
  (low 16 bits zero in every element of both files), so the engine's BF16 reading is lossless and `--compat-bf16` is
  not needed; the measured top-8 flip rate on the real file is 0 of 320 slots (`step3-gdn.md`).
- The native expert kernels accept every layer of both files; on the 1080 Ti the down projection per expert is
  Q5_K 4.3 us, Q6_K 5.3, IQ3_XXS 6.8, IQ4_XS 6.6 against Q5_1's 4.4 (`step3-experts.md`).
- Dense attention per query row at 262,144 cells: 1.0 ms on the 4090, 2.4 ms on the 1080 Ti (fp16 KV); the split-K
  scratch is 2.02 MiB per row (`step3-attn.md`).
- `moe_fused_iq.cu` (sm_80+) still lacks the new down types; those layers take MMQ in the prompt path.
- VM hygiene: sync with `rsync -rlpgoD --checksum` (`-a` keeps Mac mtimes and can leave a stale object linked);
  `pgrep -f "[s]weep_ladder_1080ti.py"` (the unbracketed pattern matches its own shell).
