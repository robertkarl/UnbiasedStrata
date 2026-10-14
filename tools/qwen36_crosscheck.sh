#!/usr/bin/env bash
# tools/qwen36_crosscheck.sh - run the Qwen3.6 numpy oracle against the pinned llama.cpp graph on synthetic GGUFs.
#
#   tools/qwen36_crosscheck.sh WORKDIR [REAL.gguf-with-nextn-for---like-real]
#
# Builds (CPU only, outside the repo, in WORKDIR/llbuild) the submodule's libllama and llama-eval-callback, builds
# tools/qwen36_dump_llama.cpp against it, writes the synthetic fixtures, and for each case compares
#   (a) every cb()-named tensor, in full  : ref/qwen36.py vs qwen36-dump      (tools/qwen36_compare_dumps.py)
#   (b) the printed stats of llama-eval-callback vs the oracle                (tools/qwen36_check_eval_callback.py)
# Exit status 1 if any seam fails its tolerance (1e-4 of max(1, max|x|)).  STRATA_GGUF_PY is honoured.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mkdir -p "${1:?usage: $0 WORKDIR [REAL.gguf]}" && cd "$1" && pwd)"
REAL="${2:-}"
LL="${LLAMA_CPP_DIR:-$REPO/third_party/llama.cpp}"
B="$W/llbuild"
PY="${PYTHON:-python3}"

if [ ! -x "$B/bin/llama-eval-callback" ]; then
  cmake -S "$LL" -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF > "$W/cmake.log" 2>&1
  cmake --build "$B" --target llama-eval-callback -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" > "$W/build.log" 2>&1
fi
c++ -std=c++17 -O1 "$REPO/tools/qwen36_dump_llama.cpp" -o "$B/qwen36-dump" \
    -I"$LL/include" -I"$LL/ggml/include" -L"$B/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin"

SYN="$REPO/tools/qwen36_synth_gguf.py"
mkdir -p "$W/fx" && cd "$W/fx"
$PY "$SYN" base.gguf > /dev/null
$PY "$SYN" seed7.gguf --seed 7 > /dev/null
$PY "$SYN" fused_gate_up.gguf --fused-gate-up --seed 3 > /dev/null
$PY "$SYN" fused_qkv.gguf --fused-qkv --no-nextn --seed 4 > /dev/null
$PY "$SYN" gqa.gguf --heads 4,2 --seed 11 > /dev/null          # n_head_kv > 1: the h // (n_head/n_head_kv) mapping is exercised
$PY "$SYN" stray_scale.gguf --expert-weights-scale 2.5 --seed 5 > /dev/null
[ -n "$REAL" ] && $PY "$SYN" like_real.gguf --like-real "$REAL" | head -4

T11=104,101,108,108,111,32,119,111,114,108,100                                     # "hello world"
T40=104,101,108,108,111,32,119,111,114,108,100,33,10,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,48
fail=0
# compare DIR_A DIR_B: print failures and the summary lines, set fail=1 on a failed seam
compare() {
  local out
  out=$($PY "$REPO/tools/qwen36_compare_dumps.py" "$1" "$2" --quiet) || fail=1
  echo "$out" | grep -E "^(FAIL|SHAPE|PAD)|^compared|^llama only" | sed 's/^/   /'
}
# case: NAME MODEL TOKENS CHUNK(0 = one batch) WITH_MTP(0/1)
run_case() {
  local name=$1 model=$2 toks=$3 chunk=$4 mtp=$5
  rm -rf "d_$name" "o_$name" "d_${name}_mtp" "o_${name}_mtp"
  local extra=() oextra=()
  [ "$mtp" = 1 ] && extra=(mtp) && oextra=(--mtp)
  [ "$chunk" != 0 ] && oextra+=(--split "$chunk")
  "$B/qwen36-dump" "$model" "d_$name" "$toks" "$chunk" ${extra[@]+"${extra[@]}"} > "dump_$name.log" 2>&1
  $PY "$REPO/ref/qwen36.py" "$model" --tokens "$toks" --dump "o_$name" ${oextra[@]+"${oextra[@]}"} > /dev/null
  echo "== $name  ($model, $(echo "$toks" | tr ',' '\n' | wc -l | tr -d ' ') tokens, chunk $chunk, mtp $mtp)"
  compare "o_$name" "d_$name"
  if [ "$mtp" = 1 ]; then
    echo "   nextn block (graph_mtp):"
    compare "o_${name}_mtp" "d_${name}_mtp"
  fi
}
run_case base_11   base.gguf $T11 0 1
run_case base_40   base.gguf $T40 0 0
run_case base_ar   base.gguf $T40 1 0          # 40 tokens one at a time: the autoregressive GDN op, carried state and KV
run_case base_c7   base.gguf $T40 7 0          # chunks of 7, the last call (5 tokens) continues from 35
run_case seed7     seed7.gguf $T11 0 1
run_case fused_gu  fused_gate_up.gguf $T11 0 1
run_case fused_qkv fused_qkv.gguf $T11 0 0
run_case gqa4_2    gqa.gguf $T40 0 1
run_case stray_scl stray_scale.gguf $T11 0 1
[ -n "$REAL" ] && run_case like_real like_real.gguf $T40 0 1

# (b) llama-eval-callback's own printout vs the oracle, for the 11-token prompt "hello world"
# llama-eval-callback's async logger drops its queue at exit, so the tail of the printout is sometimes cut off; retry until complete
for try in 1 2 3 4 5 6 7 8; do
  "$B/bin/llama-eval-callback" -m base.gguf -p "hello world" -ctk f32 -ctv f32 -fa off -c 64 -t 1 -ngl 0 > eval_callback.txt 2>&1
  $PY "$REPO/tools/qwen36_check_eval_callback.py" o_base_11 eval_callback.txt --quiet --require-tail > /dev/null 2>&1 && break
done
grep -m1 "number of input tokens" eval_callback.txt
echo "== llama-eval-callback printed stats vs o_base_11"
$PY "$REPO/tools/qwen36_check_eval_callback.py" o_base_11 eval_callback.txt --quiet --require-tail || fail=1
exit $fail
