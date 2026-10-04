# UnbiasedStrata build.  Needs CMake 3.24+, a C++20 compiler and the CUDA toolkit.  Nothing is downloaded.
BUILD_DIR  ?= build
# RTX 30 (86), RTX 40 (89) and RTX 50 (120) in one binary.  sm_120 needs CUDA 12.8+, and upstream recommends 13.0+.
CUDA_ARCHS ?= 86;89;120
JOBS       ?= $(shell nproc 2>/dev/null || echo 4)

.PHONY: build clean

build:
	@test -f third_party/llama.cpp/ggml/CMakeLists.txt || \
	  { echo "third_party/llama.cpp is empty: run 'git submodule update --init' first"; exit 1; }
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
	      -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)"
	cmake --build $(BUILD_DIR) -j $(JOBS) --target strata
	@echo "built: $(BUILD_DIR)/strata"

clean:
	rm -rf $(BUILD_DIR)
