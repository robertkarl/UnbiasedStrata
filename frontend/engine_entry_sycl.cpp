// frontend/engine_entry_sycl.cpp - the Intel build's engine: upstream's SYCL port of the engine program
// (sycl/src/program/generate.cpp), compiled into this binary under another name, as engine_entry.cpp does for CUDA.
#define main strata_engine_main
#include "../sycl/src/program/generate.cpp"
