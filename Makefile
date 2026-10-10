# UnbiasedStrata build.  Needs CMake 3.24+, a C++20 compiler and the CUDA toolkit.  Nothing is downloaded.
BUILD_DIR  ?= build
# RTX 30 (86), RTX 40 (89) and RTX 50 (120) in one binary.  sm_120 needs CUDA 12.8+, and upstream recommends 13.0+.
CUDA_ARCHS ?= 86;89;120
# capped: inside a container nproc reports the host's CPUs, not the container's quota
JOBS       ?= $(shell n=$$(nproc 2>/dev/null || echo 4); [ $$n -gt 16 ] && n=16; echo $$n)

# Intel Arc (experimental, `make build-sycl`): upstream's SYCL port, built with oneAPI's icpx and oneMKL.  SYCL_AOT
# compiles the GPU code ahead of time for one device (bmg-g31: Arc Pro B65 and B70; `ocloc ids` lists the others);
# empty = SPIR-V, compiled for the card at first run.
SYCL_BUILD_DIR ?= build-sycl
SYCL_AOT       ?= bmg-g31

.PHONY: build build-sycl clean

build:
	@test -f third_party/llama.cpp/ggml/CMakeLists.txt || \
	  { echo "third_party/llama.cpp is empty: run 'git submodule update --init' first"; exit 1; }
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
	      -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)"
	cmake --build $(BUILD_DIR) -j $(JOBS) --target unbiased-strata
	@echo "built: $(BUILD_DIR)/unbiased-strata"

build-sycl:
	@test -f third_party/llama.cpp/ggml/CMakeLists.txt || \
	  { echo "third_party/llama.cpp is empty: run 'git submodule update --init' first"; exit 1; }
	@command -v icpx >/dev/null || { echo "icpx not found: source oneAPI's setvars.sh first"; exit 1; }
	cmake -S . -B $(SYCL_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_SYCL=ON -DCMAKE_C_COMPILER=icx \
	      -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_PARITY=OFF "-DSTRATA_SYCL_AOT=$(SYCL_AOT)"
	cmake --build $(SYCL_BUILD_DIR) -j $(JOBS) --target unbiased-strata
	@echo "built: $(SYCL_BUILD_DIR)/unbiased-strata"

clean:
	rm -rf $(BUILD_DIR) $(SYCL_BUILD_DIR)
