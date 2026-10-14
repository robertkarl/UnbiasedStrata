# cmake/qwen36_synth_fixtures.cmake - writes the synthetic qwen35moe GGUFs the Qwen3.6 geometry tests read.
#
#   cmake -DPYTHON=<python3 with numpy> -DSOURCE_DIR=<repo> -DOUT=<dir> -P cmake/qwen36_synth_fixtures.cmake
#
# Exit 3 when numpy or the vendored gguf-py is missing (a trimmed third_party): the tests that need the files skip.
cmake_policy(SET CMP0007 NEW)   # keep the empty flag of "toy.gguf;" (cmake < 4 drops it and list(GET 1) fails)
file(MAKE_DIRECTORY "${OUT}")
set(_tool "${SOURCE_DIR}/tools/qwen36_synth_gguf.py")
foreach(_case "toy.gguf;" "toy-nonextn.gguf;--no-nextn" "toy-fused-gate-up.gguf;--fused-gate-up"
              "toy-fused-qkv.gguf;--fused-qkv")
  list(GET _case 0 _name)
  list(GET _case 1 _flag)
  execute_process(COMMAND ${PYTHON} -I ${_tool} "${OUT}/${_name}" ${_flag}
                  RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message("${_out}${_err}")
    if(_err MATCHES "ModuleNotFoundError")
      message("qwen36_synth_fixtures: numpy or gguf-py is missing - skipping")
      # cmake -P cannot set the exit code directly: a FATAL_ERROR exits 1, so write the marker the test reads
      file(WRITE "${OUT}/SKIP" "${_err}")
      return()
    endif()
    message(FATAL_ERROR "qwen36_synth_fixtures: ${_tool} ${_flag} failed (${_rc})")
  endif()
endforeach()
file(REMOVE "${OUT}/SKIP")
message("qwen36_synth_fixtures: 4 files in ${OUT}")
