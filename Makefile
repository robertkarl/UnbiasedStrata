# UnbiasedStrata 2 build. Needs CMake 3.24+, C/C++20 compilers and a CUDA or ROCm toolkit.
# Nothing is downloaded: ggml, llama, common and mtmd come from the third_party/llama.cpp submodule.
#
#   git clone <repo> && cd <repo>
#   git submodule update --init
#   make build                      # build/bin/{strata,strata-poly,libstrata_kq.so,unbiased-strata}
#   make test                       # the front end against a fake engine; no GPU needed
#
BUILD_DIR  ?= build
BACKEND    ?= cuda
PORTABLE_FLAG := $(if $(PORTABLE),-DSTRATA_PORTABLE=ON,)

ifeq ($(BACKEND),hip)
KQ_TARGET :=
HIP_ARCHS ?= gfx1100
ROCM_PATH ?= /opt/rocm
HIP_CXX ?= $(wildcard $(ROCM_PATH)/llvm/bin/clang++)
GPU_FLAGS := -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON \
             "-DCMAKE_HIP_ARCHITECTURES=$(HIP_ARCHS)" "-DCMAKE_PREFIX_PATH=$(ROCM_PATH)" \
             $(if $(HIP_CXX),-DCMAKE_HIP_COMPILER=$(HIP_CXX),)
else ifeq ($(BACKEND),cuda)
KQ_TARGET := strata_kq
# CUDA archs: GTX 10 (61), RTX 20 / GTX 16 (75), RTX 30 (86), RTX 40 (89).  Any arch below 75 turns on upstream's
# STRATA_EXPERIMENTAL_SM60, which needs a CUDA 12.x toolkit (13 dropped sm_61).  `make build CUDA_ARCHS=86` builds for
# one card and is several times faster.  An arch the installed nvcc cannot compile is dropped with a note.
CUDA_ARCHS ?= 61;75;86;89
# nvcc: on PATH, else the toolkit's usual place (RunPod images and distro packages differ).  NVCC=/path overrides.
NVCC ?= $(or $(shell command -v nvcc 2>/dev/null),$(wildcard /usr/local/cuda/bin/nvcc))
ifeq ($(NVCC),)
  ifneq ($(filter build configure,$(MAKECMDGOALS)),)
    $(error nvcc not found on PATH or in /usr/local/cuda/bin; install a CUDA 12.x toolkit or set NVCC=/path/to/nvcc)
  endif
endif
# Everything below that runs nvcc is skipped when there is none (a laptop running `make frontend` or `make test`).
ifneq ($(NVCC),)
NVCC_ARCHS := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | sed 's/compute_//' | tr '\n' ' ')
ifneq ($(NVCC_ARCHS),)
  CUDA_ARCHS_DROPPED := $(strip $(foreach a,$(subst ;, ,$(CUDA_ARCHS)),$(if $(filter $(firstword $(subst -, ,$(a))),$(NVCC_ARCHS)),,$(a))))
  ifneq ($(CUDA_ARCHS_DROPPED),)
    $(info make: nvcc cannot compile CUDA arch(s) $(CUDA_ARCHS_DROPPED); building without them)
    CUDA_ARCHS := $(subst $(eval) ,;,$(strip $(foreach a,$(subst ;, ,$(CUDA_ARCHS)),$(if $(filter $(firstword $(subst -, ,$(a))),$(NVCC_ARCHS)),$(a),))))
  endif
endif
SM60 := $(shell echo "$(CUDA_ARCHS)" | tr ';' '\n' | sed 's/-.*//' | grep -E '^[0-9]+$$' | awk '$$1 < 75 { print "ON"; exit }')
SM60_FLAG := $(if $(SM60),-DSTRATA_EXPERIMENTAL_SM60=ON,)
# One compiler for the C, C++ and CUDA host sides.  nvcc accepts GCC only up to a version (CUDA 12.4: 13); when the
# C++ files are compiled by a newer g++ and nvcc falls back to an older one, two libstdc++ generations meet at the link
# and it fails on __cxa_call_terminate (upstream #1645).  The probe tries the versioned compilers newest first and hands
# CMake absolute paths, so `g++` resolving to a version nvcc rejects cannot slip through.  HOSTCXX=/usr/bin/g++-13
# overrides it.
HOSTCXX ?= $(shell for c in g++-15 g++-14 g++-13 g++-12 g++-11 g++; do p=$$(command -v $$c 2>/dev/null) || continue; \
             t=$$(mktemp /tmp/strata-hc-XXXXXX.cu) && printf 'int main(){return 0;}\n' > $$t && \
             if $(NVCC) -ccbin $$p -c $$t -o /dev/null >/dev/null 2>&1; then rm -f $$t; echo $$p; break; fi; rm -f $$t; done)
HOSTCC   := $(subst g++,gcc,$(HOSTCXX))
endif
COMPILER_FLAGS := $(if $(HOSTCXX),-DCMAKE_C_COMPILER=$(HOSTCC) -DCMAKE_CXX_COMPILER=$(HOSTCXX) -DCMAKE_CUDA_HOST_COMPILER=$(HOSTCXX),)
GPU_FLAGS := -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
             "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)" $(SM60_FLAG) $(COMPILER_FLAGS) -DCMAKE_CUDA_COMPILER=$(NVCC)
else
$(error BACKEND must be cuda or hip)
endif
# capped: inside a container nproc reports the host's CPUs, not the container's quota
JOBS       ?= $(shell n=$$(nproc 2>/dev/null || echo 4); [ $$n -gt 16 ] && n=16; echo $$n)
# what `make build` produces; upstream's `all` also builds about 70 parity and bench programs
TARGETS    := strata strata-poly $(KQ_TARGET) unbiased-strata

.PHONY: build configure clean test

configure:
	@test -f third_party/llama.cpp/ggml/CMakeLists.txt || \
	  { echo "third_party/llama.cpp is empty: run 'git submodule update --init' first"; exit 1; }
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release $(GPU_FLAGS) -DSTRATA_NATIVE_EXPERTS=ON \
	      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
	      $(PORTABLE_FLAG) \
	      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(abspath $(BUILD_DIR))/bin \
	      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(abspath $(BUILD_DIR))/bin $(CMAKE_EXTRA)

build: configure
	cmake --build $(BUILD_DIR) -j $(JOBS) --target $(TARGETS)
	@echo "built:"; ls -1 $(BUILD_DIR)/bin/strata $(BUILD_DIR)/bin/strata-poly $(if $(KQ_TARGET),$(BUILD_DIR)/bin/libstrata_kq.so,) $(BUILD_DIR)/bin/unbiased-strata

# The front end alone, without CUDA: for a laptop or a CI box. Needs only a C++20 compiler and CMake.
.PHONY: frontend
frontend:
	cmake -S . -B $(BUILD_DIR)-frontend -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
	      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(abspath $(BUILD_DIR))-frontend/bin $(CMAKE_EXTRA)
	cmake --build $(BUILD_DIR)-frontend -j $(JOBS) --target unbiased-strata

# The front end's tests, against a fake engine (frontend/tests/fake-engine.py): the adapter's cases (cancel, drain,
# backpressure, child exit, ERR) and the HTTP surface a coding agent touches (frontend/tests/e2e.sh).  Needs python3
# and curl; no GPU and no model.  Runs in the CUDA-free build directory, so it works on the build box and on a laptop.
test: frontend
	cmake --build $(BUILD_DIR)-frontend -j $(JOBS) --target engine_test literals_test picture_cache_test
	$(BUILD_DIR)-frontend/bin/engine_test frontend/tests/fake-engine.py
	$(BUILD_DIR)-frontend/bin/literals_test
	$(BUILD_DIR)-frontend/bin/picture_cache_test
	frontend/tests/e2e.sh $(BUILD_DIR)-frontend/bin/unbiased-strata

clean:
	rm -rf $(BUILD_DIR)
