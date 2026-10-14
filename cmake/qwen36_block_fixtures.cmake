# cmake/qwen36_block_fixtures.cmake - the oracle's dumps of the synthetic qwen35moe file on a verify window's worth of
# tokens, for tests/cuda/qwen36_block_test.cpp (the plain block held to the dumps seam by seam).
#
#   cmake -DPYTHON=<python3 with numpy> -DSOURCE_DIR=<repo> -DSYNTH=<dir with toy.gguf> -DOUT=<dir> -P cmake/qwen36_block_fixtures.cmake
#
# Writes <OUT>/o8/<seam>.f32 and manifest.txt (ref/qwen36.py --dump) for 8 tokens ("hello wo": the fixture's vocabulary is
# the ASCII bytes).  Exit 3 when numpy or the vendored gguf-py is missing (the SKIP marker, as qwen36_synth_fixtures.cmake).
file(MAKE_DIRECTORY "${OUT}")
if(EXISTS "${SYNTH}/SKIP" OR NOT EXISTS "${SYNTH}/toy.gguf")
  file(WRITE "${OUT}/SKIP" "no synthetic fixture in ${SYNTH}")
  message("qwen36_block_fixtures: no toy.gguf in ${SYNTH} - skipping")
  return()
endif()
file(REMOVE_RECURSE "${OUT}/o8")
execute_process(COMMAND ${PYTHON} -I "${SOURCE_DIR}/ref/qwen36.py" "${SYNTH}/toy.gguf"
                        --tokens 104,101,108,108,111,32,119,111 --dump "${OUT}/o8"
                RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  message("${_out}${_err}")
  if(_err MATCHES "ModuleNotFoundError")
    message("qwen36_block_fixtures: numpy or gguf-py is missing - skipping")
    file(WRITE "${OUT}/SKIP" "${_err}")
    return()
  endif()
  message(FATAL_ERROR "qwen36_block_fixtures: ref/qwen36.py failed (${_rc})")
endif()
file(REMOVE "${OUT}/SKIP")
message("qwen36_block_fixtures: dumps in ${OUT}/o8")
