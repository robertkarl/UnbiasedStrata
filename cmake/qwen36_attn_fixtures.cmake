# cmake/qwen36_attn_fixtures.cmake - the synthetic qwen35moe GGUFs at the engine's head_dim (256) and the oracle's
# dumps of their attention layer, for qsa_dense_parity (Qwen3.6 step 3, docs/qwen36-port/step3-attn.md).
#
#   cmake -DPYTHON=<python3 with numpy> -DSOURCE_DIR=<repo> -DOUT=<dir> -P cmake/qwen36_attn_fixtures.cmake
#
# Three fixtures, one per query-heads-per-KV-head the kernels are built for: g8 (16 / 2, Qwen3.6's ratio), g12
# (24 / 2, Flash-Next's), g2 (4 / 2, the oracle's own GQA case).  Each: <OUT>/<name>.gguf and <OUT>/<name>-dump/
# with every seam of ref/qwen36.py on 40 tokens.  Writes <OUT>/SKIP (and exits 0) when numpy or gguf-py is
# missing, so the test that needs the files skips (77) instead of passing on nothing.
file(MAKE_DIRECTORY "${OUT}")
set(_synth "${SOURCE_DIR}/tools/qwen36_synth_gguf.py")
set(_oracle "${SOURCE_DIR}/ref/qwen36.py")
set(_text "dense attention over sixteen heads, G=8!")   # 40 ASCII tokens (the fixture's vocabulary is the byte set)
foreach(_case "g8;16,2;3" "g12;24,2;5" "g2;4,2;11")
  list(GET _case 0 _name)
  list(GET _case 1 _heads)
  list(GET _case 2 _seed)
  execute_process(COMMAND ${PYTHON} -I ${_synth} "${OUT}/${_name}.gguf" --heads ${_heads} --head-dim 256 --no-nextn
                          --seed ${_seed}
                  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message("${_out}${_err}")
    if(_err MATCHES "ModuleNotFoundError" OR _out MATCHES "gguf-py" OR _err MATCHES "gguf-py")
      message("qwen36_attn_fixtures: numpy or gguf-py is missing - skipping")
      file(WRITE "${OUT}/SKIP" "${_out}${_err}")
      return()
    endif()
    message(FATAL_ERROR "qwen36_attn_fixtures: ${_synth} ${_name} failed (${_rc})")
  endif()
  file(REMOVE_RECURSE "${OUT}/${_name}-dump")
  execute_process(COMMAND ${PYTHON} -I ${_oracle} "${OUT}/${_name}.gguf" --text "${_text}" --dump "${OUT}/${_name}-dump"
                  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message("${_out}${_err}")
    message(FATAL_ERROR "qwen36_attn_fixtures: ${_oracle} on ${_name} failed (${_rc})")
  endif()
endforeach()
file(REMOVE "${OUT}/SKIP")
message("qwen36_attn_fixtures: 3 fixtures and their dumps in ${OUT}")
