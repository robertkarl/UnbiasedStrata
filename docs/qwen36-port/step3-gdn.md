# Qwen3.6 port, Phase 2 step 3, area `gdn`: the GDN output gate, the router and the shared expert

Working notes, kept as the step was done (2026-10-11, branch `qwen36-step3-gdn`, from `qwen36-step2` at 8b7525e).
The design is `docs/QWEN36-35B-A3B.md` sections 3.2 and 3.4 and step 3's first and third bullets in section 6; the
plan is `docs/PLAN-qwen36-35b-a3b.md` section 5 step 3.  Every item says what was decided, where it lives, and how
it is checked, with the numbers from both cards.

## 1. What changed

| Decision | Where | Why |
| --- | --- | --- |
| `kernels::GdnGate { Sigmoid, Silu }` names the GDN output gate; every decode output kernel takes it as a required argument | `include/strata/kernels/gdn.hpp` | the one numerical difference between the two models (study 3.2, inventory row 39): `rms_norm(o) * ssm_norm * act(z)`, sigmoid for Flash-Next and SiLU for Qwen3.6.  A required parameter, not a default, so a call site rewritten later cannot silently keep sigmoid |
| `ModelGeometry::gdn_gate()` returns SiLU for `qwen35moe`, sigmoid for `qwen4exp` | `include/strata/core/layout.hpp` (`[shared]`) | from the architecture string, like `hc` and the indexer; no key says it.  No global anywhere: one process can serve both models |
| The four kernels: `gdn_out_norm` (legacy, `gdn.cu`), `fused_gdn_step_norm` (the default native decode, `fused_gdn.cu`), `native_gdn_out_norm` (pinned unfused, `native_gdn_preprocess.cu`), `gdn_step_norm_multi` (the verify window, `verify_kernels.cu`) | `src/kernels/cuda/` | SiLU is ggml's `op_silu` form `z / (1 + exp(-z))`; the sigmoid expression is untouched, so Flash-Next's arithmetic is bitwise what it was.  `verify_kernels.cu` was not in the brief's list of three, but native packs decode only through the verify window (inventory row 114), so the gate had to reach it; it stays bitwise the fused kernel (checked) |
| The seven call sites pass `g.gdn_gate()`: `layer.cpp` (3), `verify.cpp` (4) | `src/core/layer.cpp`, `src/core/verify.cpp` (`[shared]`, one commit, nothing else touched) | the smallest change the block agent's files needed |
| No decode-path literal assumed 16/48 | (none) | `grep` of `src/kernels/cuda`, `src/core`, `src/program`, `include` for 48 / 6144 / 10240 / VPK finds only `ModelGeometry`'s defaults (step 2 reads them from the file), the PLE post-ops (Flash-Next only) and comments (`session.hpp:34`, `layer.hpp:10-16`, `gdn.hpp:11`).  The kernels' only rule is `h_v % h_k == 0`; `fused_gdn_conv_l2` wants `channels % 128 == 0` (8192 is), `native_gdn_*` wants counts up to 65,535 (8192 is), `fused_gdn_ab` wants `n_embd % 8 == 0` (2048 is).  The VPK 2 shape is run through every kernel in `gdn_parity` (16/32) and `qwen36_seam_parity` (2/4) |
| The router and the shared expert: parameterisation and tests only | `src/kernels/router_top10_parity.cpp`, `src/kernels/shared_expert_parity.cpp` | generic already (study 3.1): `router_top10` takes `n_expert <= 512, k <= 32` at runtime; the native 512 x 10 router is guarded by `n_expert == 512 && k == 10` in `layer.cpp:370` and `verify.cpp:915`, so 256/8 takes the generic kernel.  No literal of 10 or 512 in the decode router path needed changing; `verify.cpp`'s `HIP_GFX906` branch already carries a 256-expert window path |
| The fixture tool grows `--gdn K,V,S`, `--experts E,K`, `--shexp-ff N`, `--quant-shexp q8_0`, `--bf16-exact` | `tools/qwen36_synth_gguf.py` (`[shared]`) | the CUDA GDN kernels are pinned to state 128 and the toy fixture has 16; the default fixture is byte-identical to before |

## 2. The harnesses

### `gdn_parity` (`src/kernels/gdn_parity.cpp`, `ctest -R gdn_parity`)

Both geometries, 16/48 (VPK 3, sigmoid) and 16/32 (VPK 2, SiLU), through all four implementations; the existing
four traps (modulo versus interleaved pairing, decay order, conv direction, l2 epsilon) kept; the fifth trap is the
gate, and it is run THROUGH the kernels: each kernel with the other model's gate must differ from the reference by
more than 5%.  The verify kernel is held bitwise to the fused one on the verify, commit and split-commit paths.

Note for a later reader: the first version of this rewrite reported the native and fused kernels 7.5x off at the
32-head geometry and exact at 48.  It was the harness: fixtures reset with pageable `cudaMemcpy` on a non-blocking
stream race the DMA.  The stream is blocking now, and the comment at its creation says why.

### `qwen36_seam_parity` (`src/kernels/qwen36_seam_parity.cpp`, `cmake/qwen36_oracle_dumps.cmake`)

`ctest -R 'qwen36_oracle_dumps|qwen36_seam_parity'`.  The fixture `k.gguf`: hidden 64, GDN 2 k-heads / 4 v-heads of
128 (VPK 2), conv 4, 256 experts top-8 of 32, shared expert 512 in Q8_0, the BF16-read weights BF16-exact; the oracle
run twice on the 11-token "hello world" sequence (F32, and `--act-quant q8`).  The engine's kernels then hold, by
llama.cpp's `cb()` names, token by token with the state carried:

| seam | through | tolerance | basis |
| --- | --- | --- | --- |
| `conv_output_silu-0`, `q_conv_predelta-0`, `k_conv_predelta-0` | legacy, native, fused (`fused_gdn_conv_l2`), verify window (`gdn_conv_l2_multi`) | 1e-6 | one 4-tap F32 dot per channel |
| `gate-0`, `beta_sigmoid-0` | fused (`fused_gdn_ab`), verify window (`gdn_ab_multi`) | 1e-6 | a 64-term dot against a BF16-exact weight |
| `final_output-0` (every token) | all four | 8e-5 | oracle-crosscheck.md's `gdn_out` class |
| `new_state-0` (after the 11th token; the verify window after two commits of 8 + 3) | all four | 2e-5 | the recurrence in F32 both sides |
| `ffn_moe_topk-0` | `router_top10` at 256/8 from the oracle's `ffn_moe_logits-0` | identical | exact top-k on the F32 fixture only |
| `ffn_moe_weights_norm-0` | same | 1e-5 | |
| `ffn_shexp_gated-0` | `shared_expert`, native Q8_0 projections, the 1-D BF16 scalar gate | 1e-5 against the Q8 oracle | the engine rounds activations to Q8_1 as ggml does; the F32 oracle is 1.2e-2 away, which is the rounding itself and is printed, not gated |
| `shared_expert_multi` | 8 + 3 tokens | bitwise the per-token call | verify.cpp's contract |

Negative cases through the kernels: the sigmoid gate at `final_output-0` (the oracle's control says 1.07), the
un-normalised router weights at `ffn_moe_weights_norm-0`, the shared gate skipped at `ffn_shexp_gated-0`.

### `router_top10_parity` and `shared_expert_parity`

The router's five fixtures at both 512/10 and 256/8 (random, all-equal, a 12-way tie wider than k, a tie of exactly
k scattered through the range, a dominant expert) and the clamp proved unreachable at both (8/256 is 512x the
2^-14 clamp).  The shared expert gains Qwen3.6's shape, 2048 x 512 with Q8_0 projections through `native_mmvq` and
Q8_1 activations and the native BF16 scalar gate, against a host reference written from the block layouts; the
silu-on-up reading and the skipped gate as negatives; `shared_expert_multi` bitwise the per-token call; and the
output moved by rounding the F32 gate weight to BF16 reported (2.5e-4 on a random gate), not asserted.

## 3. The measured-not-asserted parity: the BF16 router (study 3.4)

`tools/qwen36_router_bf16_flips.py REAL.gguf DUMP` on the oracle's `--act-quant q8` dump of the first 8 tokens of
`"Hello, world! How are you?"` (ids 9419, 11, 1814, 0, 2500, 513, 488, 30 by `llama-tokenize` on the IQ3_XXS file),
VM 221, 2026-10-11 (`/models/work/logs/qwen36-step3-gdn-flips.log`):

| contract | top-8 slots differing (of 320) | experts in one set and not the other | tokens affected | rate |
| --- | ---: | ---: | ---: | ---: |
| F32 x F32 (llama.cpp, the oracle) | 0 | 0 | 0 | 0 |
| F32 x BF16(W) (the engine's native MMVF, `bf16_gemv_fp32_mmvf`) | **0** | **0** | **0** | **0** |
| BF16(x) x BF16(W) (the legacy `bf16_gemv`, both rounded) | 34 | 6 | 6 | 0.23% (layers 4, 9, 12, 31, 32, 37; one slot each) |

**Why the native contract flips nothing: the F32 routers are already BF16.**  A census of both shipped files
(IQ3_XXS and Q4_K_XL): every element of `ffn_gate_inp`, `ffn_gate_inp_shexp`, `ssm_alpha`, `ssm_beta` and
`ssm_conv1d` has its low 16 bits zero (20,971,520 of 20,971,520 router values per file); the checkpoint is BF16 and
the converter stored these as F32.  So `BF16(W) == W`, the engine's BF16 reading is lossless, and the pack's own rule
("round exactly when the low 16 bits are zero", `upstream-tooling.md` section 4) already covers all four tensors
`moe_shared`, `fused_gdn_ab` and the router read as BF16: **`--compat-bf16` is not needed for Qwen3.6**.  The only
rounding left is the activation's, in the legacy both-operands contract, which `native_bf16_projections` (step 2's
default for native packs) does not use.  `attn_norm` is not BF16-exact (17%) and is read as F32, which is fine.
The study's "the router's flip rate is measured on the real file" (3.4, risk 6) stands, with the answer zero.

## 4. Results

### VM 221, the RTX 4090 (sm_89), 2026-10-11

    rsync -rlpgoD --checksum --delete --exclude .git --exclude '/build/' --exclude '/build-mac/' --exclude '/logs/' \
          --exclude '/third_party/llama.cpp/' ~/Code/UnbiasedStrata-step3-gdn/ rk@192.168.50.151:/models/work/qwen36-step3-gdn/
    /models/work/build-step3-gdn.sh configure && /models/work/build-step3-gdn.sh all       # 178 s clean, rc 0
    (cd build && ctest --output-on-failure -R 'qwen36_oracle_dumps|qwen36_seam_parity|gdn_parity|router_top10_parity|shared_expert_parity')

(`rsync -a` preserves the Mac's mtimes, and an object built seconds after an edit then looks newer than the edit:
the first full build linked against a stale `verify.cpp.o`.  `-rlpgoD --checksum` is `-a` without `-t`.)

5 of 5 pass.  The numbers (relative L1 unless stated):

| harness | check | measured | tolerance |
| --- | --- | ---: | ---: |
| `gdn_parity` 16/48 | legacy / native / fused state vs reference | 4.0e-8 / 6.0e-8 / 6.0e-8 | 1e-5 |
| | legacy / native / fused y (sigmoid) | 2.3e-7 / 1.4e-7 / 1.7e-7 | 1e-5 |
| | the wrong gate (SiLU) through each kernel | 87% / 87% / 90% / 91% apart | > 5% |
| `gdn_parity` 16/32 | legacy / native / fused state vs reference | 4.4e-8 / 6.0e-8 / 6.0e-8 | 1e-5 |
| | legacy / native / fused y (SiLU) | 2.8e-7 / 1.4e-7 / 1.7e-7 | 1e-5 |
| | the wrong gate (sigmoid) through each kernel | 110% / 125% / 116% / 110% apart | > 5% |
| | verify kernel bitwise fused (y; commit; split commit) | yes / yes / yes | bitwise |
| | pairing, decay, conv layout, l2 eps traps | 128% / 83% / 128% / 1031% apart | > 5% |
| `qwen36_seam_parity` | `conv_output_silu-0`, worst path | 5.96e-8 | 1e-6 |
| | `q/k_conv_predelta-0` (fused, verify) | 9.4e-8 | 1e-6 |
| | `gate-0` / `beta_sigmoid-0` (fused_gdn_ab, gdn_ab_multi) | 2.3e-7 / 1.1e-7 | 1e-6 |
| | `final_output-0`, worst token, legacy / native / fused / verify | 2.28e-7 / 1.91e-7 / 2.15e-7 / 2.15e-7 | 8e-5 |
| | `new_state-0`, legacy / native / fused / verify (two commits) | 1.31e-7 / 1.49e-7 / 1.59e-7 / 1.59e-7 | 2e-5 |
| | sigmoid gate at `final_output-0` (fused; verify window) | 125.8% / 125.8% apart | > 5% |
| | `ffn_moe_topk-0` | 0 of 88 differ | identical |
| | `ffn_moe_weights_norm-0`; the un-normalised reading | 6.1e-8; 23.5% apart | 1e-5; > 5% |
| | `ffn_shexp_gated-0` vs the Q8 oracle; vs the F32 oracle | 3.5e-7; 1.2e-2 (reported) | 1e-5 |
| | the shared gate skipped; `shared_expert_multi` vs per-token | 51.5% apart; bitwise | > 5%; bitwise |
| `router_top10_parity` 512/10 and 256/8 | ids on 5 fixtures x 64 tokens; weights | exact; worst 1e-6-class | exact; 1e-5 |
| `shared_expert_parity` 2048 x 512 | native Q8_0 + BF16 gate vs reference | 9.6e-8 | 1e-4 |
| | silu-on-up; gate skipped; F32 vs BF16 gate weight | 83% / 272% apart; 2.5e-4 (reported) | > 5% |

### VM 211, the GTX 1080 Ti (sm_61), 2026-10-11

`/home/rk/qwen36-step3-gdn-run.sh` (kept on the VM), under the card lock for 5 minutes: the rsync above to
`/home/rk/qwen36-step3-gdn/`, `set -o noclobber; echo qwen36-step3-gdn > /home/rk/VM-BUSY` after checking
`pgrep -x llama-server` and `pgrep -f "[s]weep_ladder_1080ti.py"` are empty (the bracket keeps `pgrep -f` from
matching its own shell, which it did on every earlier poll), then

    cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON \
          -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
          -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13                                   # 4 s
    cmake --build build -j 8 -- -k                                                                 # 287 s, rc 0, everything
    (cd build && ctest --output-on-failure -R 'qwen36_oracle_dumps|qwen36_seam_parity|gdn_parity|router_top10_parity|shared_expert_parity')
    rm /home/rk/VM-BUSY            # the script's EXIT trap

5 of 5 pass (`/home/rk/logs/qwen36-step3-gdn-1080ti.log`, the four binaries' printouts beside it).  Every seam number
of `qwen36_seam_parity` is the same digits as on the 4090 (the kernels are F32 and the summation orders do not
depend on the card): `final_output-0` 2.28e-7 / 1.91e-7 / 2.15e-7 / 2.15e-7, `new_state-0` 1.31e-7 / 1.49e-7 /
1.59e-7 / 1.59e-7, `conv_output_silu-0` 5.96e-8 worst, `gate-0` 2.27e-7, `ffn_moe_topk-0` identical,
`ffn_moe_weights_norm-0` 6.1e-8, `ffn_shexp_gated-0` 3.5e-7 against the Q8 oracle, `shared_expert_multi` bitwise,
the three negatives 125.8% / 23.5% / 51.5%.  `gdn_parity`: states 5.3e-8 to 6.0e-8, y 1.4e-7 to 2.8e-7, the verify
kernel bitwise the fused one on all three paths, the wrong gate 87% apart at 16/48 and 110% at 16/32 through every
kernel.  `shared_expert_parity` at 2048 x 512: 9.6e-8; `router_top10_parity`: ids exact at both shapes, weights
worst 1.5e-7.

Nothing in the changed kernels is architecture-conditional: the SiLU is an `expf` and a divide in kernels that
already ran on sm_61 in step 2's build, so no new guard was needed; the pass on the card is the proof.

## 5. The interface other agents consume

    namespace strata::kernels {
    enum class GdnGate : int { Sigmoid = 0, Silu = 1 };                       // strata/kernels/gdn.hpp
    inline const char* gdn_gate_name(GdnGate g);
    void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S,
                      float eps, GdnGate act, void* stream);                  // gdn.hpp (legacy)
    void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                             const float* beta, const float* z, const float* gamma, float eps, GdnGate act,
                             float* y, int h_k, int h_v, void* stream);       // fused_gdn.hpp
    void native_gdn_out_norm(const float* output, const float* z, const float* gamma, float* destination,
                             int64_t heads, int64_t cols, float epsilon, GdnGate act, void* stream);  // native_gdn_preprocess.hpp
    void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                             const float* z, const float* gamma, float eps, GdnGate act, float* y, int h_k, int h_v,
                             int n_tok, const int32_t* n_keep, void* stream, int t_out_begin = 0);  // verify_kernels.hpp
    }
    kernels::GdnGate strata::core::ModelGeometry::gdn_gate() const;          // strata/core/layout.hpp

Everything else is unchanged: `GdnShapes{S, h_k, h_v}` at runtime, the state `(S, h_v, S)` with j fastest, the conv
history `[channel][3]`, `router_top10(logits, n_tokens, n_expert, k, ids, weights, stream)` at 256/8, `shared_expert`
and `shared_expert_multi` with `NativeSharedWeights` (type ids per `native_mmvq.hpp`) at `n_ff` 512.  The prompt-path
GDN kernels (`src/prefill/kernels.cu:324, 612`) are the prefill agent's and still gate with sigmoid; they will want the
same `GdnGate` argument.  `layer.hpp:16`'s comment still says "SIGMOID for qwen4exp; qwen3.5 used SiLU" (the block
agent's file; the code beneath it is right).
