# cmake/qwen36_oracle_dumps.cmake - the Qwen3.6 kernel fixture and the oracle's seam dumps for the GPU parity tests.
#
#   cmake -DPYTHON=<python3 with numpy> -DSOURCE_DIR=<repo> -DOUT=<dir> -P cmake/qwen36_oracle_dumps.cmake
#
# Writes OUT/k.gguf with tools/qwen36_synth_gguf.py at the geometry the engine's CUDA GDN kernels are pinned to
# (state 128, 2 k-heads / 4 v-heads: VPK 2 like the real 16 / 32), the real router shape (256 experts, top-8), the
# shared expert at its real width (512) in Q8_0, and the weights the engine reads as BF16 already BF16-representable;
# then runs ref/qwen36.py on the 11-token "hello world" sequence twice: OUT/o_f32 (F32 activations) and OUT/o_q8
# (--act-quant q8: activations rounded to Q8 before every quantized matmul, as the engine's native MMVQ path does).
# Every tensor llama.cpp names with cb() lands in those directories as <name>.f32 (docs/qwen36-port/oracle-crosscheck.md).
#
# Exit 3 (and OUT/SKIP written) when numpy or the vendored gguf-py is missing, so the tests that need the files skip.
file(MAKE_DIRECTORY "${OUT}")
set(_synth "${SOURCE_DIR}/tools/qwen36_synth_gguf.py")
set(_oracle "${SOURCE_DIR}/ref/qwen36.py")
set(_tokens "104,101,108,108,111,32,119,111,114,108,100")
set(_step1 ${PYTHON} -I ${_synth} "${OUT}/k.gguf" --gdn 2,4,128 --experts 256,8 --shexp-ff 512 --quant-shexp q8_0 --bf16-exact --no-nextn)
set(_step2 ${PYTHON} -I ${_oracle} "${OUT}/k.gguf" --tokens ${_tokens} --dump "${OUT}/o_f32")
set(_step3 ${PYTHON} -I ${_oracle} "${OUT}/k.gguf" --tokens ${_tokens} --dump "${OUT}/o_q8" --act-quant q8)
foreach(_name _step1 _step2 _step3)
  set(_cmd ${${_name}})
  execute_process(COMMAND ${_cmd} RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message("${_out}${_err}")
    if(_err MATCHES "ModuleNotFoundError" OR _err MATCHES "gguf-py was not found")
      message("qwen36_oracle_dumps: numpy or gguf-py is missing - skipping")
      file(WRITE "${OUT}/SKIP" "${_err}")
      return()
    endif()
    message(FATAL_ERROR "qwen36_oracle_dumps: `${_cmd}` failed (${_rc})")
  endif()
endforeach()
file(REMOVE "${OUT}/SKIP")
file(WRITE "${OUT}/tokens.txt" "${_tokens}\n")
message("qwen36_oracle_dumps: k.gguf, o_f32/, o_q8/ in ${OUT}")
