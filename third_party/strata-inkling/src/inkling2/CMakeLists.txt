# engine/src/inkling2/CMakeLists.txt - Standalone & integrated CMake build for `strata-inkling` (component `inkling2`)
# MIT License - Copyright (c) 2026 Strata contributors
enable_language(C)
set(_ink_engine_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored ZSTD decompressor (zero external libzstd dependency on Win/Linux/macOS)
set(_zstd_dir "${_ink_engine_root}/third_party/zstd-dec")
if(NOT TARGET strata_zstd_dec)
  add_library(strata_zstd_dec STATIC
    ${_zstd_dir}/common/debug.c
    ${_zstd_dir}/common/entropy_common.c
    ${_zstd_dir}/common/error_private.c
    ${_zstd_dir}/common/fse_decompress.c
    ${_zstd_dir}/common/xxhash.c
    ${_zstd_dir}/common/zstd_common.c
    ${_zstd_dir}/decompress/huf_decompress.c
    ${_zstd_dir}/decompress/zstd_ddict.c
    ${_zstd_dir}/decompress/zstd_decompress.c
    ${_zstd_dir}/decompress/zstd_decompress_block.c
  )
  target_include_directories(strata_zstd_dec PUBLIC ${_zstd_dir} ${_zstd_dir}/common ${_zstd_dir}/decompress)
  target_compile_definitions(strata_zstd_dec PRIVATE ZSTD_DISABLE_ASM=1)
endif()

# Core Inkling-2-Mini + .strata v1.1 + Official HF safetensors + Community GGUF static library
add_library(strata_inkling_core STATIC
  ${PROJECT_SOURCE_DIR}/src/artifact/strata_reader.cpp
  ${_ink_engine_root}/src/quant/sk_codecs.cpp
  ${_ink_engine_root}/src/quant/dequant.cpp
  ${_ink_engine_root}/src/quant/types.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2_loop.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ink_cpu_backend.cpp
)
target_include_directories(strata_inkling_core PUBLIC
  ${_ink_engine_root}/include
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${_ink_engine_root}/third_party/stb
)
target_link_libraries(strata_inkling_core PUBLIC strata_zstd_dec)
if(WIN32)
  target_compile_definitions(strata_inkling_core PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

find_package(Threads REQUIRED)
target_link_libraries(strata_inkling_core PUBLIC Threads::Threads)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(strata_inkling_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Standalone `strata-inkling` executable
add_executable(strata-inkling
  ${CMAKE_CURRENT_SOURCE_DIR}/ink_main.cpp
)
target_link_libraries(strata-inkling PRIVATE strata_inkling_core)
if(COMMAND strata_utf8_manifest)
  strata_utf8_manifest(strata-inkling)
endif()

add_subdirectory(metal)

install(TARGETS strata-inkling RUNTIME DESTINATION bin COMPONENT inkling2)
