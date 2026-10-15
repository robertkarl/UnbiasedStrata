# UnbiasedStrata build.  Needs CMake 3.24+, a C++20 compiler and the CUDA toolkit.  Nothing is downloaded.
BUILD_DIR  ?= build
# One binary for every single NVIDIA GPU from Pascal up: GTX 10 (61), Volta (70), RTX 20 / GTX 16 (75), RTX 30 (86),
# RTX 40 (89), RTX 50 (120).  Pascal and Volta are the experimental build (STRATA_EXPERIMENTAL_SM60; needs a CUDA 12.x
# toolkit, 13 dropped them): any arch below 75 in the list switches it on.  sm_120 needs CUDA 12.8+.  An arch the
# installed nvcc cannot compile is dropped from the list with a note (CUDA 12.4 has no 120; CUDA 13 has no 61 / 70),
# so the default builds on any toolkit; `make build CUDA_ARCHS=61` keeps a single-card build short
# (docs/qwen36-port/pascal-build.md).
CUDA_ARCHS ?= 61;70;75;86;89;120
NVCC_ARCHS := $(shell nvcc --list-gpu-arch 2>/dev/null | sed 's/compute_//' | tr '\n' ' ')
ifneq ($(NVCC_ARCHS),)
  CUDA_ARCHS_DROPPED := $(strip $(foreach a,$(subst ;, ,$(CUDA_ARCHS)),$(if $(filter $(firstword $(subst -, ,$(a))),$(NVCC_ARCHS)),,$(a))))
  ifneq ($(CUDA_ARCHS_DROPPED),)
    $(info make: nvcc cannot compile CUDA arch(s) $(CUDA_ARCHS_DROPPED); building without them)
    CUDA_ARCHS := $(subst $(eval) ,;,$(strip $(foreach a,$(subst ;, ,$(CUDA_ARCHS)),$(if $(filter $(firstword $(subst -, ,$(a))),$(NVCC_ARCHS)),$(a),))))
  endif
endif
SM60 := $(shell echo "$(CUDA_ARCHS)" | tr ';' '\n' | sed 's/-.*//' | grep -E '^[0-9]+$$' | awk '$$1 < 75 { print "ON"; exit }')
SM60_FLAG := $(if $(SM60),-DSTRATA_EXPERIMENTAL_SM60=ON,)
# capped: inside a container nproc reports the host's CPUs, not the container's quota
JOBS       ?= $(shell n=$$(nproc 2>/dev/null || echo 4); [ $$n -gt 16 ] && n=16; echo $$n)

.PHONY: build clean

# Muse is an isolated native engine build. Its newer tokenizer/reference dependency
# does not replace the Qwen build's pinned llama.cpp.
.PHONY: build-muse
build-muse:
	@test -f third_party/llama-muse/src/models/muse-spark.cpp || \
	  { echo "run 'git submodule update --init third_party/llama-muse' first"; exit 1; }
	cmake -S muse -B $(BUILD_DIR)-muse -DCMAKE_BUILD_TYPE=Release \
	  "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)" $(CMAKE_EXTRA)
	cmake --build $(BUILD_DIR)-muse -j $(JOBS) --target unbiased-strata muse-check muse-kernel-test muse-chat-test

build:
	@test -f third_party/llama.cpp/ggml/CMakeLists.txt || \
	  { echo "third_party/llama.cpp is empty: run 'git submodule update --init' first"; exit 1; }
	@git -C third_party/llama.cpp apply --check --reverse ../../patches/llama.cpp-qwen3-coder-any-argument-order.patch 2>/dev/null || \
	  git -C third_party/llama.cpp apply ../../patches/llama.cpp-qwen3-coder-any-argument-order.patch
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
	      -DSTRATA_PORTABLE=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)" $(SM60_FLAG) $(CMAKE_EXTRA)
	cmake --build $(BUILD_DIR) -j $(JOBS) --target unbiased-strata
	@echo "built: $(BUILD_DIR)/unbiased-strata"

clean:
	rm -rf $(BUILD_DIR)

# Protocol and child supervision regressions; no CUDA/model required.
.PHONY: test-frontend
test-frontend:
	mkdir -p $(BUILD_DIR)/frontend-tests
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/child_watch_test.cpp -o $(BUILD_DIR)/frontend-tests/child-watch-test
	$(BUILD_DIR)/frontend-tests/child-watch-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/engine_test.cpp -o $(BUILD_DIR)/frontend-tests/engine-test
	$(BUILD_DIR)/frontend-tests/engine-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror -pthread frontend/engine_backpressure_test.cpp -o $(BUILD_DIR)/frontend-tests/engine-backpressure-test
	$(BUILD_DIR)/frontend-tests/engine-backpressure-test

# Model-template compatibility; CPU only, after the regular build produced llama libraries.
.PHONY: test-chat-parser
test-chat-parser:
	@test -n "$(MODEL)" || { echo 'set MODEL to the first GGUF shard'; exit 2; }
	mkdir -p $(BUILD_DIR)/frontend-tests
	$(CXX) -std=c++20 -O0 -Ifrontend -Ithird_party/llama.cpp/include -Ithird_party/llama.cpp/common \
	  -Ithird_party/llama.cpp/ggml/include -Ithird_party/llama.cpp/vendor -Ithird_party/llama.cpp/tools/mtmd \
	  frontend/chat_parser_test.cpp \
	  -o $(BUILD_DIR)/frontend-tests/chat-parser-test $(BUILD_DIR)/llama.cpp/tools/mtmd/libmtmd.a \
	  $(BUILD_DIR)/llama.cpp/vendor/hash/libvendor-hash.a $(BUILD_DIR)/llama.cpp/common/libllama-common.a \
	  $(BUILD_DIR)/llama.cpp/src/libllama.a $(BUILD_DIR)/llama.cpp/vendor/cpp-httplib/libcpp-httplib.a \
	  $(BUILD_DIR)/ggml/src/libggml.a $(BUILD_DIR)/ggml/src/libggml-cpu.a $(BUILD_DIR)/ggml/src/libggml-base.a \
	  $(BUILD_DIR)/llama.cpp/common/libllama-common-base.a -lpthread -ldl -lm
	$(BUILD_DIR)/frontend-tests/chat-parser-test "$(MODEL)"

# Against a running server: a request may ask for more tokens than the context has left (REGINA-112).
.PHONY: test-serve
test-serve:
	@test -n "$(PORT)" || { echo 'set PORT to a running server ([HOST:]PORT)'; exit 2; }
	python3 tests/python/serve_max_tokens_check.py $(PORT)
