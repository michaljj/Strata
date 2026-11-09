# Strata Engine - top-level build.
#
# One project builds the engine for every backend: CUDA (STRATA_ENABLE_CUDA), HIP (STRATA_ENABLE_HIP; the layer-split
# and DeepSeek engines are the STRATA_BUILD_SPLIT / STRATA_BUILD_DSV41 options), the CPU-only build, and the test
# programs.  The Intel Arc (SYCL) engine is a separate project in sycl/ (STRATA_ENABLE_SYCL), and the Metal engine
# one in metal/.  CMakePresets.json has the CPU check preset (`cmake --preset cpu-check`) and the compile-only
# backend presets used by tools/check and the CI workflows; the release scripts in packaging/ pass their own flags.
#
# The CUDA kernels need sm_80 or newer (tf32 mma, bf16 math) and the guard below refuses anything older: silently
# building for a pre-Ampere arch would produce a binary that runs and is wrong about its own performance, which is
# the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.59 LANGUAGES CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_geometry.cmake)   # -DSTRATA_GEOMETRY=<name>: the model shape the kernels are compiled for (default flash_next, no flag added)
# The build is cut by backend (cleanup 5.4): each file below only DEFINES macros (strata_<part>()), which run here, in this scope, at the place their
# block used to be - so this file reads in the order the old single CMakeLists.txt evaluated.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_ggml.cmake)          # which ggml: the vendored third_party/strata-ggml, or -DSTRATA_GGML_DIR
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/cpu.cmake)         # strata_kernels_cpu, the native (ggml-cpu) expert rows, the CPU tests
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/cuda.cmake)        # NVIDIA: toolchain, llama.cpp's MMQ kernels (strata_mmq, strata_kq)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends/hip.cmake)         # AMD: gfx906 compat, MMQ on HIP
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_gpu_engine.cmake)    # the GPU engine both of them build
# EXPERIMENTAL: the Intel Arc engine, the SYCL port in sycl/ (PR #423; icpx + oneMKL, docs/INTEL_ARC.md). It is its
# own project that takes the unported sources from src/; with this on, nothing below is configured.
option(STRATA_ENABLE_SYCL "EXPERIMENTAL: build the Intel Arc (SYCL) engine from sycl/ - docs/INTEL_ARC.md" OFF)
if(STRATA_ENABLE_SYCL)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_ENABLE_SYCL is a separate backend; leave STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP off")
  endif()
  message(WARNING "STRATA_ENABLE_SYCL: the Intel Arc engine is experimental (community port, docs/INTEL_ARC.md)")
  add_subdirectory(sycl)
  return()
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
# EXPERIMENTAL, opt-in: an engine for CPUs without AVX2 (#394, #595, #623), compiled on that PC.  ggml-cpu, which then
# computes every CPU expert row, is built for this floor instead of AVX2 or the build host: "avx" (Sandy/Ivy Bridge:
# AVX, no FMA/F16C/AVX2) or "none" (SSE4.2 + POPCNT: Nehalem/Westmere).  Strata's own AVX2/AVX-512 kernels stay in the
# binary behind their CPU checks; the router lookahead gets an AVX1 kernel.  Empty (the default) changes nothing.
set(STRATA_ISA_FLOOR "" CACHE STRING "experimental older-CPU build: empty (default, AVX2 floor), avx, or none (SSE4.2)")
set_property(CACHE STRATA_ISA_FLOOR PROPERTY STRINGS "" avx none)
if(NOT STRATA_ISA_FLOOR STREQUAL "" AND NOT STRATA_ISA_FLOOR STREQUAL "avx" AND NOT STRATA_ISA_FLOOR STREQUAL "none")
  message(FATAL_ERROR "STRATA_ISA_FLOOR must be empty, avx or none (got '${STRATA_ISA_FLOOR}')")
endif()
if(STRATA_ISA_FLOOR AND STRATA_PORTABLE)
  message(FATAL_ERROR "STRATA_ISA_FLOOR (an older-CPU build) and STRATA_PORTABLE (the AVX2 release build) exclude each other")
endif()
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# #3202: Windows hands a program its command line in the system's ANSI code page (cp1251 / cp866 on a Russian Windows, cp1252 on an
# English one), but the engine reads every path as UTF-8 (the run configs and the server write UTF-8). A model folder with
# Cyrillic letters arrived as other bytes, or as "?" where the page has no such letter, and the model did not open. This manifest
# makes the process's ANSI code page UTF-8 (Windows 10 1903 and newer; older Windows ignores it and nothing changes), so the
# command line, fopen and std::ifstream all take UTF-8.
function(strata_utf8_manifest target)
  if(MSVC)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/strata-utf8.manifest)
  endif()
endfunction()

# The AVX kernels exist for x86 only.  On aarch64 (DGX Spark / GB10) portable.cpp stands in for them and
# native experts run on ggml-cpu's NEON dot products.
if(NOT CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X64|i[3-6]86|x86|X86)$")   # an unset processor stays the x86 build, as before
  set(STRATA_CPU_X86 ON)
else()
  set(STRATA_CPU_X86 OFF)
endif()
if(STRATA_ISA_FLOOR AND NOT STRATA_CPU_X86)
  message(WARNING "STRATA_ISA_FLOOR is an x86 older-CPU build and does not apply to ${CMAKE_SYSTEM_PROCESSOR}; ignored")
  set(STRATA_ISA_FLOOR "" CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
# AMD gfx906 (MI50, wave64): the same .cu sources compiled as HIP through the gfx906 compat layer.  The CUDA runtime and
# cuBLAS names are mapped by the headers in include/strata/platform/hip_compat/, which go first on the include path,
# and CUDA::cudart / CUDA::cublas become aliases of HIP's targets, so every CUDA target below stays as it is.  This is
# separate from STRATA_ENABLE_HIP (the wave32 RDNA backend): with it on, STRATA_ENABLE_CUDA is on and
# STRATA_ENABLE_HIP stays off, so every "CUDA" branch of the tree is the one taken.
option(STRATA_HIP_GFX906 "Build the CUDA targets for AMD gfx906 through HIP (wave64 compat layer)" OFF)
if(STRATA_HIP_GFX906)
  if(STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_HIP_GFX906 and STRATA_ENABLE_HIP are separate backends; enable only one")
  endif()
  set(STRATA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)
option(STRATA_PREFILL_MMQ "Build the optional GGML MMQ prefill path for HIP" OFF)
# The community-made AMD changes (include/strata/core/amd_community.hpp): on by default.  This option is the build half of the one
# switch STRATA_AMD_COMMUNITY_TUNING: OFF compiles the portable signed-byte dot for gfx1010 again (the SDWA sequence of #4855).
# The rocBLAS tables are switched at run time (STRATA_AMD_COMMUNITY_TUNING=0).
option(STRATA_AMD_COMMUNITY_TUNING "Compile the community-made gfx1010 SDWA dot product (the runtime switch of the same name controls the gfx1030 rocBLAS tables)" ON)
if(NOT STRATA_AMD_COMMUNITY_TUNING)
  add_compile_definitions(STRATA_AMD_COMMUNITY_DOT_OFF=1)
endif()
# The rows one verify / batch window holds (include/strata/kernels/window_limits.hpp): 8, the 0.2.7 window, or 16 (four pipelined
# groups of 16 slots = 64 slots on a four-card split, docs/BATCHING.md).  Only the CUDA and HIP engines are built from
# these sources with a wide window; the SYCL and Metal builds never read this variable and keep 8.  A window of up to
# 8 rows runs the kernels it always ran, whichever is chosen (the wide instantiations are extra code).  "auto" = 16 for
# the HIP engines (a HIP code object loads on first use, so the extra kernels cost nothing until a wide window forms)
# and 8 for CUDA, which loads every kernel at start (CUDA_MODULE_LOADING=EAGER: the wide kernels would take VRAM from
# the expert cache of every card); -DSTRATA_WINDOW_MAX_T=16 builds the wide CUDA engine.
set(STRATA_WINDOW_MAX_T "auto" CACHE STRING "rows of a verify / batch window: 8, 12, 16 or auto (HIP 16, CUDA 8)")
set_property(CACHE STRATA_WINDOW_MAX_T PROPERTY STRINGS auto 8 12 16)
if(NOT STRATA_WINDOW_MAX_T STREQUAL "auto" AND NOT STRATA_WINDOW_MAX_T STREQUAL "8" AND NOT STRATA_WINDOW_MAX_T STREQUAL "12" AND NOT STRATA_WINDOW_MAX_T STREQUAL "16")
  message(FATAL_ERROR "STRATA_WINDOW_MAX_T must be auto, 8, 12 or 16 (got '${STRATA_WINDOW_MAX_T}')")
endif()
if(STRATA_ENABLE_HIP OR STRATA_ENABLE_CUDA)
  if(STRATA_WINDOW_MAX_T STREQUAL "auto")
    if(STRATA_ENABLE_HIP)
      set(_strata_window_max_t 16)
    else()
      set(_strata_window_max_t 8)
    endif()
  else()
    set(_strata_window_max_t ${STRATA_WINDOW_MAX_T})
  endif()
  add_compile_definitions(STRATA_VERIFY_MAX_T=${_strata_window_max_t})
  message(STATUS "Strata: verify / batch windows hold up to ${_strata_window_max_t} rows")
endif()
# MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K (Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF
# projections, STRATA_DENSE_MMQ).  On by default since MODEL-UPDATE Phase 1: on CUDA they are their own library
# (strata_kq, loaded only for a model with these formats: the engine loads every kernel of its own binary at start,
# CUDA_MODULE_LOADING=EAGER, and these took ~3.8 MB of VRAM - 3 IQ3_XXS expert slots on a P100 - from every model); on
# HIP a code object loads on first use, so they cost other models nothing there (R9700: the same slots).  Used at run
# time when STRATA_MMQ_KQ=1, which the engine sets by itself on NVIDIA cards of compute capability 8.0 and up (include/strata/core/nv_defaults.hpp) and on
# Linux gfx12 (STRATA_MMQ_KQ=0 turns it off; setup also writes it for UD-Q5_K_XL / UD-Q6_K_XL); other cards keep the FP16 path.  This CMake option only
# decides whether the kernels are BUILT: on in every release build.
option(STRATA_MMQ_TQ_OVERLAY "Gyro trellis MMQ prompt kernels: the patched ggml-cuda headers of third_party/strata-ggml/ggml/src/ggml-cuda-tq (with -DSTRATA_GGML_DIR: an overlay of that llama.cpp's headers written at configure time, cmake/strata_ggml_tq_overlay.cmake)" ON)
option(STRATA_MMQ_KQUANTS "Build the MMQ prompt kernels for Q4_K / Q5_K / Q5_1 / Q6_K: the experts of Unsloth's K-quant files, and on HIP the dense GGUF projections (STRATA_DENSE_MMQ)" ON)
# Q6_K gate/up in the grouped expert kernels (Unsloth's UD-Q6_K_XL, and layer 2 of UD-Q5_K_XL; docs/QUANTS.md).  On by
# default since MODEL-UPDATE Phase 1: one more instance per grouped kernel in iq_kernels.cu, +19 KB of sm_60 code,
# measured to leave every expert-cache slot count as it was (P100, R9700).  -DSTRATA_Q6K_EXPERTS=OFF drops it.
option(STRATA_Q6K_EXPERTS "Build the Q6_K gate/up grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)" ON)
if(STRATA_Q6K_EXPERTS)
  add_compile_definitions(STRATA_Q6K_EXPERTS=1)
endif()
option(STRATA_ORCA_Q4KS_MMQ "Build the GGML Q5_0 MMQ instance for OrcaRouter Q4_K_S" OFF)
# CUDA: the pointer-table MMQ behind STRATA_MMQ_DIRECT=1 (the prompt path's MMQ reads resident experts in their cache
# slots through a Strata-side copy of llama.cpp's MMQ scheduler, src/prefill/moe_mmq_tab.cuh).  Off by default, like
# the K-quant options above: the engine loads every kernel when it starts (CUDA_MODULE_LOADING=EAGER), and these nine
# instances (+15 MB of binary) take 36-38 MiB of VRAM per GPU on sm_86 - expert slots every model would lose, with
# STRATA_MMQ_DIRECT unset too.  They also add ~70% of the upstream MMQ instances' compile time per CUDA arch (175 s
# of CPU for the nine at sm_86 alone).  Off, the build's device code is byte for byte the one without this feature.  CUDA
# only (not HIP, not gfx906); switched off at configure time when the llama.cpp tree's mmq.cuh is not one the copy
# was verified against.
option(STRATA_MMQ_TAB "Build the pointer-table prompt MMQ for STRATA_MMQ_DIRECT=1 (CUDA; ~37 MiB VRAM per GPU)" OFF)
if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
endif()
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
# Tests are on by default when the build can run them.  What they need: their sources (src/**/*_test.cpp, tests/core,
# bench/micro; a source drop without them has nothing to build) and the x86 CPU kernels, which most of the CPU suite drives
# (arm64 stays off by default: -DSTRATA_BUILD_TESTS=ON still builds what builds there).  GPU tests are registered only
# when STRATA_ENABLE_CUDA / STRATA_ENABLE_HIP is on, and the configure summary below says which kinds were registered, so
# a missing GPU is a stated skip and never a silent pass.  Release builds say -DSTRATA_BUILD_TESTS=OFF themselves
# (packaging/, setup.py, Dockerfile, spark/).
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/core AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro AND STRATA_CPU_X86)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
# A test that needs a model file (a real expert blob from a pack, which is not in the source tree): registered as the real test
# when the file is there, and otherwise as a stated skip that ctest lists as "Skipped" with this reason, never as a pass of nothing
# and not as a failure on a machine that has no pack.
function(strata_add_pack_test name needs)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${needs}")
    add_test(NAME ${name} COMMAND ${ARGN} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E echo "SKIPPED: ${name} needs ${needs} (a Q2_0 expert pack, a model file), which is not in this tree")
    set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: ")
  endif()
endfunction()
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
# #2027: defined here, before the GPU section reads it to register the native-expert tests (it was defined at the
# native-experts section far below, so a first configure registered four tests fewer than the second)
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

if(STRATA_BUILD_CONVERSATION_TESTS)
  enable_testing()
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_split_failure_test src/core/conversation_split_failure_test.cpp)
  target_include_directories(conversation_split_failure_test PRIVATE include)
  add_test(NAME conversation_split_failure_test COMMAND conversation_split_failure_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  add_executable(conversation_file_test src/core/conversation_file_test.cpp src/core/conversation_file.cpp)
  target_include_directories(conversation_file_test PRIVATE include)
  add_test(NAME conversation_file_test COMMAND conversation_file_test)
  add_executable(session_save_reclaim_test src/core/session_save_reclaim_test.cpp src/core/conversation_file.cpp)
  target_include_directories(session_save_reclaim_test PRIVATE include)
  add_test(NAME session_save_reclaim_test COMMAND session_save_reclaim_test)
endif()

if(STRATA_HIP_GFX906)
  strata_hip_gfx906_toolchain()
elseif(STRATA_ENABLE_CUDA)
  # #585: CMake links CUDA code with the STATIC runtime by default, and the targets below also link CUDA::cudart (the
  # DLL's import library).  MSVC with a CUDA 12.x toolkit - the experimental Pascal/Volta build - then fails with
  # LNK1169 (every cudart symbol defined twice); GNU ld lets it pass.  That build links the shared runtime alone.  The
  # ready-made engine (CUDA 13, no STRATA_EXPERIMENTAL_SM60) is unchanged; -DCMAKE_CUDA_RUNTIME_LIBRARY overrides.
  if(MSVC AND STRATA_EXPERIMENTAL_SM60 AND NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()
  # #4023: the default below has to be set BEFORE enable_language(CUDA): since CMake 3.18 that call fills an unset
  # CMAKE_CUDA_ARCHITECTURES with nvcc's own default (sm_75 for CUDA 13), so a test after it never saw "undefined" and
  # a build with no option was an sm_75 build.  -DCMAKE_CUDA_ARCHITECTURES and the CUDAARCHS environment variable win.
  set(_strata_cuda_arch_defaulted OFF)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    if(STRATA_BUILD_DSV41)
      # #3935: the DeepSeek engine's own release shape (sm_75, sm_86, sm_89, sm_120 + PTX 120, as packaging/build_dsv41_cuda.sh), so a hand
      # build of strata-dsv41 with no architecture runs on an RTX 20, 30, 40 or 50 card; the Qwen engine's default below is unchanged
      set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120" CACHE STRING "CUDA architectures")
    else()
      set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures")
    endif()
    set(_strata_cuda_arch_defaulted ON)
  endif()
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  option(STRATA_EXPERIMENTAL_SM60 "community build for GPUs below compute capability 7.5: Pascal sm_60, Volta sm_70 (#236)" OFF)
  if(_strata_cuda_arch_defaulted)
    # a recommendation, not a refusal: the release engine is built for 75;86;89;120, a hand build names its card
    message(WARNING "Strata: CMAKE_CUDA_ARCHITECTURES was not set, so this build is for ${CMAKE_CUDA_ARCHITECTURES} only (sm_120 = RTX 50). For another "
                    "card pass it: -DCMAKE_CUDA_ARCHITECTURES=native (the GPUs in this PC), 75 (RTX 20), 86 (RTX 30), "
                    "89 (RTX 40), or a list like \"86;120\".")
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Turing port: developed and measured on sm_120 (RTX 50); the tf32 mma in the QSA scorer has a
    # portable fp32-FMA fallback, so RTX 20 (75), 30 (86), RTX 40 (89) and RTX 50 (120) build too.
    # Pre-sm_75 is refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    # STRATA_EXPERIMENTAL_SM60: a Pascal (sm_60, GP100-class) or Volta (sm_70, V100 / Titan V, #236) build.
    # These sit below every architecture the kernels were written for, so two intrinsics are replaced below
    # their arch - `__dp4a` (6.1+) and `__nanosleep` (7.0+) - in include/strata/kernels/dp4a.hpp, which is
    # bit-exact for the first and merely a shorter pause for the second (sm_70 has both).  Community-tested,
    # not in the ready-made engine; RTX 20/30/40/50 cards must use the release engine instead.  It needs a CUDA
    # 12.x toolkit: CUDA 13 dropped sm_60 and sm_70 (nvcc fails CMake's compiler test with "compute_70").
    if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM60)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.  (A Pascal sm_60 or Volta sm_70 build is "
        "-DSTRATA_EXPERIMENTAL_SM60=ON; upstream does not support it.)")
    endif()
    if(_base LESS 52)
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES '${CMAKE_CUDA_ARCHITECTURES}': even the experimental build "
                          "needs compute capability 5.2 (sm_52, Maxwell) or newer")
    endif()
    # Maxwell (sm_52, #3218): one reporter built and ran this on two Quadro M6000 (CUDA 12.x, Qwen3.8-Flash-Next
    # IQ1, about 22 t/s).  It has no native half-precision arithmetic, so it is slower per byte than Pascal; only
    # this opt-in build gets the floor.  Warn, do not refuse.
    if(_base LESS 60)
      message(WARNING "Building for sm_${_base} (Maxwell): community-tested on one reporter's Quadro M6000 only, "
                      "no native half-precision arithmetic, CUDA 12.x toolkit required (docs/OLDER_GPUS.md).")
    endif()
    # #220 / #224: RTX 50 engines built with CUDA 12.8 crashed in the prompt path on Linux (13.0 and newer did not)
    if(_base GREATER_EQUAL 120 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(WARNING "Building for sm_${_base} (RTX 50) with CUDA ${CMAKE_CUDA_COMPILER_VERSION}: engines built with "
                      "CUDA 12.8 crashed on long prompts on these cards (issues #220, #224). Use CUDA 13.0 or newer.")
    endif()
  endforeach()
  if(STRATA_EXPERIMENTAL_SM60)
    # every target sees it: the Pascal / Volta code paths (the Volta prompt attention, PR #600) are compiled into this
    # build only, so the ready-made engine (sm_75+) keeps exactly the code it had
    add_compile_definitions(STRATA_EXPERIMENTAL_SM60=1)
  endif()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(STRATA_ENABLE_HIP)
  set(_strata_gpu_runtime_target strata_hip_runtime)
  set(_strata_gpu_blas_target ${STRATA_HIP_BLAS_TARGETS})
  set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  add_executable(batch_telemetry_test src/core/batch_telemetry_test.cpp)
  target_include_directories(batch_telemetry_test PRIVATE include)
  add_test(NAME batch_telemetry_test COMMAND batch_telemetry_test)
  add_executable(logit_bias_test src/core/logit_bias_test.cpp)
  target_include_directories(logit_bias_test PRIVATE include)
  add_test(NAME logit_bias_test COMMAND logit_bias_test)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # a split model's shards and their joint view: missing shard, split keys, duplicates, truncation (synthetic)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)
  # the model families (qwen4exp, qwen35moe): the guards, the geometry from metadata, the layout tables (synthetic)
  add_executable(model_family_test tests/core/model_family_test.cpp)
  target_link_libraries(model_family_test PRIVATE strata_artifact)
  add_test(NAME model_family_test COMMAND model_family_test)
  # how --batch cuts a prompt read into pieces (a layer split with no slot decoding must still reach a BYIELD)
  add_executable(batch_read_test tests/program/batch_read_test.cpp)
  target_include_directories(batch_read_test PRIVATE include)
  add_test(NAME batch_read_test COMMAND batch_read_test)
  # the model's geometry from its metadata, and the check against the kernels' compiled shape (synthetic)
  add_executable(gguf_geometry_test tests/core/gguf_geometry_test.cpp)
  target_link_libraries(gguf_geometry_test PRIVATE strata_artifact)
  add_test(NAME gguf_geometry_test COMMAND gguf_geometry_test)

  # the hipBLASLt tuning header's architecture gate: RDNA4 siblings share solution IDs, RDNA3 siblings share
  # theirs, cross-family pairs and version mismatches stay rejected (synthetic tables: no model, no GPU)
  add_executable(tuning_header_arch_family_test tests/prefill/tuning_header_arch_family_test.cpp)
  target_include_directories(tuning_header_arch_family_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
  add_test(NAME tuning_header_arch_family_test COMMAND tuning_header_arch_family_test)

  # the auto layer split's minimum stage size (#1616): header-only rule, no GPU
  add_executable(split_rules_test tests/program/split_rules_test.cpp)
  target_include_directories(split_rules_test PRIVATE include)
  add_test(NAME split_rules_test COMMAND split_rules_test)

  # #3797: the out-of-memory report at start (per card free/total, the largest other programs): the text is a pure
  # function; the driver query (NVML, loaded at run time) is only checked to be safe to call. No GPU needed.
  add_executable(gpu_report_test tests/program/gpu_report_test.cpp)
  target_include_directories(gpu_report_test PRIVATE include)
  target_link_libraries(gpu_report_test PRIVATE ${CMAKE_DL_LIBS})
  add_test(NAME gpu_report_test COMMAND gpu_report_test)

  # #3702: the later-stage reserve of the layer split search (header-only arithmetic, pinned against the old expression)
  add_executable(stage_reserve_test tests/program/stage_reserve_test.cpp)
  target_include_directories(stage_reserve_test PRIVATE include)
  add_test(NAME stage_reserve_test COMMAND stage_reserve_test)

  # the layer split's spill walk (header-only arithmetic: who is short, which step comes next)
  add_executable(split_walk_test tests/program/split_walk_test.cpp)
  target_include_directories(split_walk_test PRIVATE include)
  add_test(NAME split_walk_test COMMAND split_walk_test)

  # the prompt planner's host-memory model (#2824): header-only, no GPU
  add_executable(prefill_plan_test tests/program/prefill_plan_test.cpp)
  target_include_directories(prefill_plan_test PRIVATE include)
  add_test(NAME prefill_plan_test COMMAND prefill_plan_test)

  # the default CPU prefill share is not armed when every expert is resident (#1595): header-only rule, no GPU
  add_executable(share_rules_test tests/prefill/share_rules_test.cpp)
  target_include_directories(share_rules_test PRIVATE include)
  add_test(NAME share_rules_test COMMAND share_rules_test)

  # STRATA_SELECT_TWO_PASS: the bound between the single-pass and 3xTF32 QSA block scores and the cut-off rule that
  # keeps the selection (header-only, no GPU)
  add_executable(qsa_two_pass_test tests/prefill/qsa_two_pass_test.cpp)
  target_include_directories(qsa_two_pass_test PRIVATE include)
  add_test(NAME qsa_two_pass_test COMMAND qsa_two_pass_test)
  # STRATA_SELECT_TWO_PASS: the request's parse and the line it prints once where the scorer cannot run (no GPU)
  add_executable(qsa_two_pass_status_test tests/prefill/qsa_two_pass_status_test.cpp)
  target_include_directories(qsa_two_pass_status_test PRIVATE include)
  add_test(NAME qsa_two_pass_status_test COMMAND qsa_two_pass_status_test
           "${CMAKE_CURRENT_BINARY_DIR}/qsa_two_pass_status_test.log")
           
  # STRATA_PREFILL_ROUTED_STREAM: the prompt stream's issuer walk and route gate, on a simulated link and consumer
  # (header-only, no GPU)
  add_executable(routed_stream_test tests/prefill/routed_stream_test.cpp)
  target_include_directories(routed_stream_test PRIVATE include)
  target_link_libraries(routed_stream_test PRIVATE Threads::Threads)
  add_test(NAME routed_stream_test COMMAND routed_stream_test)

  # PR #4006: STRATA_PREFILL_TIMING does not record a mark in the phase already running (header-only, no GPU)
  add_executable(timer_marks_test tests/prefill/timer_marks_test.cpp)
  target_include_directories(timer_marks_test PRIVATE include)
  add_test(NAME timer_marks_test COMMAND timer_marks_test)

  # PR #5108: the MMQ row table's over-read margin follows the backend tile bound; every fixup read stays inside (header-only, no GPU)
  add_executable(mmq_ids_margin_test tests/prefill/mmq_ids_margin_test.cpp)
  target_include_directories(mmq_ids_margin_test PRIVATE include)
  add_test(NAME mmq_ids_margin_test COMMAND mmq_ids_margin_test)

  # PR #4844: the adaptive tier's copy-in range across serial, pipelined and drained rounds (header-only, no GPU)
  add_executable(adaptive_range_test tests/program/adaptive_range_test.cpp)
  target_include_directories(adaptive_range_test PRIVATE include)
  add_test(NAME adaptive_range_test COMMAND adaptive_range_test)

  # P0.1 (lite): STRATA_MANIFEST, the machine-readable record of the selected paths, tensor formats, engine and budgets (header-only, no GPU)
  add_executable(run_manifest_test tests/core/run_manifest_test.cpp)
  target_include_directories(run_manifest_test PRIVATE include)
  add_test(NAME run_manifest_test COMMAND run_manifest_test)

  # the Windows HIP auto prefill chunk cap (#1630) and the commit-limit verdict (#1607): header-only, no GPU
  add_executable(host_limits_test tests/platform/host_limits_test.cpp)
  target_include_directories(host_limits_test PRIVATE include)
  add_test(NAME host_limits_test COMMAND host_limits_test)
  # #4733: an engine started at Low I/O priority goes back to Normal (Windows; else skips)
  add_executable(io_priority_test tests/platform/io_priority_test.cpp src/platform/io_priority.cpp)
  target_include_directories(io_priority_test PRIVATE include)
  add_test(NAME io_priority_test COMMAND io_priority_test)
  set_tests_properties(io_priority_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built when a GPU backend is on, so the Phase 1 targets keep working without a toolkit.
strata_gpu_engine()

strata_cpu_kernels()

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from the tree vendored in
# third_party/strata-ggml (frozen at the llama.cpp commit in third_party/ggml/VERSION.txt, see its UPSTREAM.md): no download.
# -DSTRATA_GGML_DIR=<llama.cpp checkout> still builds from that checkout instead (an override kept for two releases).
strata_ggml_resolve("${CMAKE_CURRENT_SOURCE_DIR}")
if(STRATA_PREFILL_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_ORCA_Q4KS_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORCA_Q4KS_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  strata_ggml_configure_root()
  strata_cpu_native_experts()
  strata_ggml_mmq_types()
  # #296: OrcaRouter Q4_K_S's Q5_0 experts (Q4_K and Q5_1 come with STRATA_MMQ_KQUANTS on CUDA, S5's branch)
  if(STRATA_ORCA_Q4KS_MMQ)
    list(APPEND _strata_mmq_types q5_0)
  endif()
  if(STRATA_ENABLE_CUDA)
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    set(_strata_quantize_cu ${_strata_mmq}/quantize.cu)   # the overlay's copy when the trellis overlay is on: it includes mmq.cuh
    if(_strata_mmq_tq_overlay)
      set(_strata_quantize_cu ${_strata_tq_overlay_dir}/quantize.cu)
    endif()
    # #136: the Q2_0 pack's prompt experts on Strata's own int8 kernels (moe_fused.cu, opt-in STRATA_PF_FUSED=1; CUDA
    # only: the HIP branch below keeps MMQ, and an sm_75 device or an sm_75-only build reports them unavailable)
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu
                src/prefill/mmq_nvfp4_w4a8.cu
                src/prefill/ggml_cuda_host.cu ${_strata_quantize_cu})
    if(STRATA_EXPERIMENTAL_SM60 AND NOT STRATA_HIP_GFX906)
      # GP100 (sm_60): a resident layer's experts weight-stationary, one launch per product (moe_ws.hpp; opt-in at
      # run time, STRATA_MMQ_WS=1, and only where every visible GPU is cc 6.0)
      target_sources(strata_mmq PRIVATE src/prefill/moe_ws.cu src/prefill/moe_ws_gu_iq2.cu src/prefill/moe_ws_gu_iq3.cu
                     src/prefill/moe_ws_down.cu)
    endif()
    # q8_0: the drafter's matrices (E-9); STRATA_ORCA_Q4KS_MMQ adds q5_0 (#296, in _strata_mmq_types), and
    # STRATA_MMQ_KQUANTS q4_k q5_k q5_1 q6_k, the experts of Unsloth's UD-Q4/Q5/Q6_K_XL (docs/QUANTS.md; the dispatch
    # follows eddoursul/Strata 7706c1b).  MODEL-UPDATE Phase 1: on CUDA they are a library of their own, strata_kq
    # (below), loaded only for a model with these experts: the engine loads every kernel of its own binary at start
    # (CUDA_MODULE_LOADING=EAGER), and these four took ~3.8 MB of VRAM (3 expert slots of IQ3_XXS on a P100) from every
    # model.  gfx906 (HIP: code objects load on first use) keeps them in strata_mmq.
    set(_strata_mmq_cuda_types ${_strata_mmq_types})
    set(_strata_kq_types q4_k q5_k q5_1 q6_k)
    set(_strata_kq_dso OFF)
    if(STRATA_MMQ_KQUANTS AND STRATA_HIP_GFX906)
      list(APPEND _strata_mmq_cuda_types ${_strata_kq_types})
    elseif(STRATA_MMQ_KQUANTS)
      set(_strata_kq_dso ON)
    endif()
    foreach(_t IN LISTS _strata_mmq_cuda_types)
      target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    if(_strata_mmq_tq_overlay)
      foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
        target_sources(strata_mmq PRIVATE ${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      # first on the include path: the patched mmq.cuh / ggml.h shadow the pinned ones for this target only
      target_include_directories(strata_mmq BEFORE PRIVATE ${_strata_tq_overlay_dir} ${_strata_tq_overlay_dir}/include)
    endif()
    # NVFP4: W4A8 (default) in mmq_nvfp4_w4a8.cu with the plain architectures; the opt-in W4A4 needs RTX 50's FP4 MMA,
    # which exists only in the arch-specific target, so its one unit alone is built with 12x -> 12xa (the default
    # binary is unchanged and runs on sm_121; a card without that image falls back to W4A8 at run time)
    set(_strata_w4a4_archs ${CMAKE_CUDA_ARCHITECTURES})
    # `native` is first replaced by the detected architectures (e.g. 120-real), so the 12x -> 12xa rewrite below applies
    # to it too: with plain sm_120 ptxas refuses the FP4 MMA of this one unit (#4023)
    if("native" IN_LIST _strata_w4a4_archs AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
      list(REMOVE_ITEM _strata_w4a4_archs native)
      list(APPEND _strata_w4a4_archs ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    endif()
    # (two steps: CMake before 3.29 rejects a \2 whose optional group did not match, "120" alone)
    list(TRANSFORM _strata_w4a4_archs REPLACE "^(12[0-9])$" "\\1a")
    list(TRANSFORM _strata_w4a4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
    add_library(strata_mmq_w4a4 STATIC src/prefill/mmq_nvfp4_w4a4.cu)
    set_target_properties(strata_mmq_w4a4 PROPERTIES CUDA_ARCHITECTURES "${_strata_w4a4_archs}")
    target_include_directories(strata_mmq_w4a4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
    target_compile_options(strata_mmq_w4a4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq_w4a4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(strata_mmq_w4a4 PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_mmq PRIVATE strata_mmq_w4a4)
    # STRATA_MMQ_TAB: the pointer-table copy of MMQ's scheduler mirrors mmq.cuh from mul_mat_q_process_tile through
    # mul_mat_q_case (struct mmq_args included).  A fingerprint of that text instead of a compile probe: nothing is
    # compiled at configure time, and a llama.cpp tree with another scheduler turns the feature off instead of
    # failing the build.  _strata_mmq_tab_known: the hashes moe_mmq_tab.cuh was verified against (3cf03257, the
    # fetched pin; the vendored third_party/llama.cpp tree has the same mmq.cuh).
    set(_strata_mmq_tab_on OFF)
    if(STRATA_MMQ_TAB AND NOT STRATA_HIP_GFX906)   # gfx906 compiles no .cu from here (its HIP glob): no tab at all
      set(_strata_mmq_tab_known 36e4dc716c504470ec09cc0852d29a6ac5eace257f6e5e72db3805ca4f043eb0)
      set(_strata_mmq_tab_hash "")
      if(EXISTS ${_strata_mmq}/mmq.cuh)
        file(READ ${_strata_mmq}/mmq.cuh _strata_mmq_src)
        string(REPLACE "\r\n" "\n" _strata_mmq_src "${_strata_mmq_src}")
        string(FIND "${_strata_mmq_src}" "static __device__ __forceinline__ void mul_mat_q_process_tile" _strata_tab_a)
        string(FIND "${_strata_mmq_src}" "#define DECL_MMQ_CASE" _strata_tab_b)
        if(_strata_tab_a GREATER -1 AND _strata_tab_b GREATER _strata_tab_a)
          math(EXPR _strata_tab_len "${_strata_tab_b} - ${_strata_tab_a}")
          string(SUBSTRING "${_strata_mmq_src}" ${_strata_tab_a} ${_strata_tab_len} _strata_tab_region)
          string(SHA256 _strata_mmq_tab_hash "${_strata_tab_region}")
        endif()
        unset(_strata_mmq_src)
        unset(_strata_tab_region)
      endif()
      if(_strata_mmq_tab_hash AND _strata_mmq_tab_hash IN_LIST _strata_mmq_tab_known)
        set(_strata_mmq_tab_on ON)
      else()
        message(STATUS "STRATA_MMQ_TAB off: mmq.cuh differs from the mirrored scheduler (${_strata_mmq_tab_hash})")
      endif()
    endif()
    if(_strata_mmq_tab_on)
      set(_strata_tab_cuh ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_mmq_tab.cuh)
      # the native packs' nine types, the ones prefill_mmq_tab_test checks bit for bit; the optional Q5_0 / K-quant
      # instances (STRATA_ORCA_Q4KS_MMQ, STRATA_MMQ_KQUANTS) get no pointer table and keep the gather
      set(_strata_mmq_tab_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
      foreach(_t IN LISTS _strata_mmq_tab_types)
        string(TOUPPER ${_t} _strata_tab_T)
        configure_file(src/prefill/mmq_tab_instance.cu.in ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu @ONLY)
        target_sources(strata_mmq PRIVATE ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu)
      endforeach()
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TAB=1)
      message(STATUS "STRATA_MMQ_TAB on: pointer-table MMQ for ${_strata_mmq_tab_types}")
    endif()
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    # GGML's CUDA error reporting calls cuGetErrorString from the driver API.
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    if(TARGET CUDA::cuda_driver)   # not in the HIP (gfx906) build, whose CUDA::cudart is a stand-in
      target_link_libraries(strata_mmq PUBLIC CUDA::cuda_driver)
    endif()
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(_strata_kq_dso)
      # strata_kq: the K-quant MMQ instances with their own copy of ggml_cuda_host.cu's symbols and the static CUDA
      # runtime, next to the engine (libstrata_kq.so / strata_kq.dll); moe_mmq.cu opens it when a model asks for one
      # of these formats (STRATA_MMQ_KQ=1).  Without the file the engine runs them on the FP16 prompt path, as before.
      target_compile_definitions(strata_mmq PRIVATE STRATA_KQ_DSO=1)
      target_sources(strata_mmq PRIVATE src/prefill/mmq_kq_loader.cpp)
      if(NOT WIN32)
        target_link_libraries(strata_mmq PUBLIC ${CMAKE_DL_LIBS})
      endif()
      add_library(strata_kq SHARED src/prefill/mmq_kq_lib.cu src/prefill/ggml_cuda_host.cu)
      foreach(_t IN LISTS _strata_kq_types)
        target_sources(strata_kq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      target_include_directories(strata_kq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/include)
      target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
      if(MSVC)
        target_compile_options(strata_kq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
      endif()
      set_target_properties(strata_kq PROPERTIES CUDA_RUNTIME_LIBRARY Static POSITION_INDEPENDENT_CODE ON
                            CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
      # ggml-base is static: position-independent so a shared library can hold it (the same code otherwise)
      set_target_properties(ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
      target_link_libraries(strata_kq PRIVATE ggml-base)
      if(TARGET CUDA::cuda_driver)
        target_link_libraries(strata_kq PRIVATE CUDA::cuda_driver)
      endif()
      add_dependencies(strata_mmq strata_kq)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    if(_strata_mmq_tq)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQ=1)
    endif()
    if(_strata_mmq_tqp)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQP=1)
    endif()
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    if(STRATA_EXPERIMENTAL_SM60 AND NOT STRATA_HIP_GFX906)
      # the weight-stationary experts (moe_ws.hpp) are in strata_mmq: the prompt path may take them (STRATA_MMQ_WS=1)
      target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_WS=1)
    endif()
    if(STRATA_BUILD_TESTS)
      # #136: the fused int8 experts against a double-precision reference and the MMQ path (exit 77 below sm_80)
      add_executable(prefill_fused_moe_test tests/cuda/prefill_fused_moe_test.cpp)
      target_link_libraries(prefill_fused_moe_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_moe_test COMMAND prefill_fused_moe_test)
      set_tests_properties(prefill_fused_moe_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # #136: the same for the native packs' formats (IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL)
      add_executable(prefill_fused_iq_test tests/cuda/prefill_fused_iq_test.cpp)
      target_link_libraries(prefill_fused_iq_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_iq_test COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_Q2N=1 on its own (the test leaves STRATA_PF_FUSED unset): the native Q2_0 pack's Q2_0 / Q2_0
      # layers on sm_80+ (77 below sm_80 and without a card)
      add_test(NAME prefill_fused_iq_test_q2n COMMAND prefill_fused_iq_test "--only=Q2_0 / Q2_0" --unset-pf-fused)
      set_tests_properties(prefill_fused_iq_test_q2n PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_Q2N=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_NT=256 kernels (256 threads, two blocks a multiprocessor, two-pass weight loads) against the
      # same references; the batches of the test reach them through the Batch overload of experts_native (77 below sm_80)
      add_test(NAME prefill_fused_iq_test_nt256 COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_nt256 PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_NT=256" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the opt-in STRATA_PF_FUSED_F16=1 (f16 mma, rounding level) and STRATA_PF_FUSED_WS=1 (warp-specialized, the same F16 products;
      # about 101 KB of shared memory at its defaults, more than an sm_86 / sm_89 / sm_120 card's 99 KB, so it is set up on
      # sm_80 and sm_90 cards and the test skips (77) elsewhere)
      add_test(NAME prefill_fused_iq_test_f16 COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_f16 PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_F16=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      add_test(NAME prefill_fused_iq_test_ws COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test_ws PROPERTIES ENVIRONMENT "STRATA_PF_FUSED_WS=1" SKIP_RETURN_CODE 77 TIMEOUT 600)
      # the MoE input quantized per token and scattered to its rows (mmq::quantize_scatter) against the per-row
      # gather: the same q8_1 bytes for every MMQ layout (--bench: both at 2K, 8K and 32K tokens)
      add_executable(prefill_mmq_scatter_test tests/cuda/prefill_mmq_scatter_test.cpp)
      target_link_libraries(prefill_mmq_scatter_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_mmq_scatter_test COMMAND prefill_mmq_scatter_test)
      set_tests_properties(prefill_mmq_scatter_test PROPERTIES TIMEOUT 120)
      if(STRATA_EXPERIMENTAL_SM60)
        # mmq::swiglu_quant: swiglu + the down product's q8_1 rows in one kernel against the two, byte for byte
        add_executable(prefill_swiglu_quant_test tests/cuda/prefill_swiglu_quant_test.cpp)
        target_link_libraries(prefill_swiglu_quant_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
        add_test(NAME prefill_swiglu_quant_test COMMAND prefill_swiglu_quant_test)
        set_tests_properties(prefill_swiglu_quant_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        # the weight-stationary experts (STRATA_MMQ_WS) against the per-group MMQ, byte for byte, on random experts; the
        # verdict is given on a compute capability 6.0 card only (77 elsewhere)
        add_executable(prefill_ws_parity_test tests/cuda/prefill_ws_parity_test.cpp)
        target_link_libraries(prefill_ws_parity_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
        add_test(NAME prefill_ws_parity_test COMMAND prefill_ws_parity_test)
        set_tests_properties(prefill_ws_parity_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
      endif()
    endif()
    if(STRATA_BUILD_TESTS AND _strata_mmq_tab_on)
      # STRATA_MMQ_DIRECT: the pointer-table MMQ bit for bit against the gathered MMQ (exit 77 below sm_80)
      add_executable(prefill_mmq_tab_test tests/cuda/prefill_mmq_tab_test.cu)
      target_link_libraries(prefill_mmq_tab_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_mmq_tab_test COMMAND prefill_mmq_tab_test)
      set_tests_properties(prefill_mmq_tab_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    endif()
    if(STRATA_BUILD_TESTS AND STRATA_MMQ_KQUANTS)
      # the UD-Q4_K_XL expert formats through MMQ against ggml's dequantized weights (synthetic, a few seconds)
      add_executable(prefill_mmq_kquant_test tests/cuda/prefill_mmq_kquant_test.cpp)
      target_link_libraries(prefill_mmq_kquant_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_mmq_kquant_test COMMAND prefill_mmq_kquant_test)
      # the K-quant MMQ is opt-in at run time (STRATA_MMQ_KQ=1), and on CUDA it is strata_kq next to the test
      set_tests_properties(prefill_mmq_kquant_test PROPERTIES TIMEOUT 120 ENVIRONMENT STRATA_MMQ_KQ=1)
    endif()
    # the NVFP4 expert products (w4a8 / w4a4, STRATA_PREFILL_NVFP4) against a double-precision reference
    add_executable(mmq_nvfp4_parity src/prefill/mmq_nvfp4_parity.cpp)
    target_link_libraries(mmq_nvfp4_parity PRIVATE strata_mmq ggml-base CUDA::cudart)
    if(NOT MSVC)   # its FP16 rounding helper uses the F16C intrinsics (GCC/Clang need the target flag)
      target_compile_options(mmq_nvfp4_parity PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mf16c>)
    endif()
    # the routed-expert products (MMQ) timed at the prompt path's shapes (16 experts a launch)
    add_executable(mmq_expert_bench src/prefill/mmq_expert_bench.cpp)
    target_link_libraries(mmq_expert_bench PRIVATE strata_mmq ggml-base CUDA::cudart)
    # the same products at DS4.1's shapes, three ways: the engine's set_tc tiles, MMQ, and a fused probe.
    # The DS4.1 kernels exist only in a CUDA or HIP build, so the target is added where they do.
    if(STRATA_BUILD_DSV41 AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP OR STRATA_HIP_GFX906))
      add_executable(ds4_expert_bench tools/dsv41_lab/ds4_expert_bench.cpp src/dsv41/ds4_iq.cu src/dsv41/ds4_tq.cu)
      target_include_directories(ds4_expert_bench PRIVATE src/dsv41 third_party/llama.cpp/ggml/src ${CMAKE_BINARY_DIR}/ggml-tq-overlay)
      target_link_libraries(ds4_expert_bench PRIVATE strata_dsv41_kernels strata_prefill strata_mmq ggml-base ${_strata_gpu_runtime_target})
    endif()
  elseif(STRATA_PREFILL_MMQ)
    strata_hip_mmq()
  endif()
  if(STRATA_ENABLE_HIP)
    # Optional real-model diagnostic: pass the GGUF shard containing the native PLE key.
    add_executable(hip_ple_iq4 tests/hip/ple_iq4.cpp)
    target_link_libraries(hip_ple_iq4 PRIVATE strata_kernels strata_hip_runtime ggml-base)
  endif()
  # the NVFP4 AVX-512 and AVX2 rows vs ggml-cpu's vec_dot, and their speed (CPU only; optional arg: a pack's experts.bin)
  add_executable(nvfp4_avx512_parity src/kernels/cpu/nvfp4_avx512_parity.cpp)
  target_link_libraries(nvfp4_avx512_parity PRIVATE strata_kernels_cpu)
  add_test(NAME nvfp4_cpu_parity COMMAND nvfp4_avx512_parity)   # random blocks; the AVX2 rows bit-equal to ggml's
  set_tests_properties(nvfp4_cpu_parity PROPERTIES TIMEOUT 120)
  if(STRATA_ENABLE_CUDA)   # the NVFP4 decode kernels vs ggml's reference
    add_executable(nvfp4_expert_gpu_parity src/kernels/nvfp4_expert_gpu_parity.cpp)
    target_link_libraries(nvfp4_expert_gpu_parity PRIVATE strata_kernels ggml-base CUDA::cudart)
    # the decode experts' weight bandwidth on random blobs of a format pair, and an output hash (bit-for-bit checks)
    add_executable(nvfp4_decode_bench src/kernels/nvfp4_decode_bench.cpp)
    target_link_libraries(nvfp4_decode_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${_strata_gpu_include_directories})
    # strata_engine: the --load-check mode runs the engine's start-up checks (NativeDense, check_experts_gguf)
    target_link_libraries(native_expert_parity PRIVATE strata_engine strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
    target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    if(STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
      # STRATA_EXPERT_NIBBLE (include/strata/core/expert_nibble.hpp): the 4-bit gate/up expert slots against the
      # production kernels and the GGUF blobs, bit for bit, on a real model's layers (needs a GP100); --header: the
      # contract's host half, no GPU
      add_executable(iq_nibble_parity src/kernels/iq_nibble_parity.cpp)
      target_include_directories(iq_nibble_parity PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(iq_nibble_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ggml-base ${_strata_gpu_runtime_target})
      if(STRATA_BUILD_TESTS)
        add_test(NAME iq_nibble_parity_header COMMAND iq_nibble_parity --header)
        # the synthetic experts of every gate/up x down format pair, no model (skips with 77 on a card that is not a GP100)
        add_test(NAME iq_nibble_parity_synthetic COMMAND iq_nibble_parity --synthetic-only)
        set_tests_properties(iq_nibble_parity_synthetic PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
      endif()
    endif()
    # MODEL-UPDATE: the dense Q8_0 GEMVs at the engine's shapes against the memory bandwidth (a benchmark, no test)
    add_executable(q8_gemv_bench src/kernels/q8_gemv_bench.cpp)
    target_include_directories(q8_gemv_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q8_gemv_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    # EXPDEC (docs/EXPERTDEC.md): the grouped experts' default kernels against STRATA_EXPDEC=1 at T = 1..4, bit for bit
    # and in microseconds, on synthetic blobs read cold from DRAM
    add_executable(expdec_bench EXCLUDE_FROM_ALL src/kernels/expdec_bench.cpp)
    target_include_directories(expdec_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(expdec_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
      # Unsloth UD-Q4_K_XL's three expert pairs (and Q8_0 alone, its embedding) on synthetic ggml-quantized weights
      # at 2560/640: GPU and CPU experts against the float reference, GPU dequantizers against to_float.
      # q4_K/q5_0 and q5_0/q8_0: a plain llama-quantize "_M" tier GGUF (not Unsloth's dynamic method) commonly
      # puts Q5_0 on down-projections (sometimes gate/up too) where this model's native_experts.txt had no GPU
      # kernel for it before; q5_0/q5_0 (both roles Q5_0) is left out of this list since stacking the same
      # lower-precision format on both roles pushes the GPU path over this test's 3% threshold (quantization
      # noise, not a kernel bug -- see the PR).
      # q4_0/q4_0 and q4_0/q4_1: a plain llama-quantize Q4_0 file (bartowski's Qwen3.8-Flash-Next-Q4_0: gate/up Q4_0,
      # down Q4_0 on 42 layers and Q4_1 on 6)
      # q6_K/q8_0: Unsloth UD-Q6_K_XL gate/up Q6_K with Q8_0 down (docs/UNSLOTH_Q6.md). Layer 2 is q8_0/q8_0, already listed.
      set(_strata_parity_pairs q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0 q4_0/q4_0 q4_0/q4_1)
      if(STRATA_Q6K_EXPERTS)
        list(APPEND _strata_parity_pairs q6_K/q8_0)
      else()   # a pair without GPU kernels (Q6_K gate/up) is refused by native_expert_supported, not run
        add_test(NAME native_expert_parity_refuses_q6_K COMMAND native_expert_parity --synthetic q6_K/q8_0)
        set_tests_properties(native_expert_parity_refuses_q6_K PROPERTIES WILL_FAIL TRUE)
      endif()
      foreach(_pair ${_strata_parity_pairs})
        string(REPLACE "/" "_" _name "${_pair}")
        add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
        set_tests_properties(native_expert_parity_${_name} PROPERTIES TIMEOUT 120)
      endforeach()
      add_test(NAME native_expert_parity_q5_1_min COMMAND native_expert_parity --q5_1-min)
      # Unsloth v3.1 Qwen3.8-Flash-Next sets (UD-IQ1_M .. UD-IQ4_XS, round 0214): every gate/up x down pair their layers use,
      # at the real shapes (hidden 2560, FF 640): gate/up IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ4_XS (+ IQ1_M, the GSQ-RCO packs'),
      # down Q2_0 IQ4_NL Q5_0 Q8_0.  Q5_0 down (UD-IQ4_XS 33 layers, UD-Q3_K_XL 20) had no test on this model before.
      foreach(_gu iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq4_xs)
        foreach(_dn q2_0 iq4_nl q5_0 q8_0)
          add_test(NAME native_expert_parity_v31_${_gu}_${_dn} COMMAND native_expert_parity --synthetic ${_gu}/${_dn})
          set_tests_properties(native_expert_parity_v31_${_gu}_${_dn} PROPERTIES TIMEOUT 180)
        endforeach()
      endforeach()
      # the dense projections (Q5_0, Q6_K, Q5_K, Q4_K, Q8_0, IQ4_XS, IQ4_NL ...): decode MMVQ and the prompt's dequantizers
      add_executable(native_dense_parity src/kernels/native_dense_parity.cpp)
      target_include_directories(native_dense_parity PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(native_dense_parity PRIVATE strata_kernels ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME native_dense_parity COMMAND native_dense_parity)
      set_tests_properties(native_dense_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # APR: the trellis expert types and the Hadamard activation rotation, when ggml comes from the agention fork
      # (the only tree with these types; marked by its ggml-cuda/tq.cuh)
      if(STRATA_GGML_DIR AND EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/tq.cuh")
        set(_strata_tq_pairs tqk6/tqk7 tq2_t/tq2_t)
        if(EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/template-instances/mmq-instance-tqp5.cu")   # Gyro-L's pair trellis
          list(APPEND _strata_tq_pairs tqp5/tqp5 tqp6/tqp6 tqp7/tqp7)
        endif()
        foreach(_pair ${_strata_tq_pairs})
          string(REPLACE "/" "_" _name "${_pair}")
          add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
          add_test(NAME native_expert_parity_hadamard_${_name} COMMAND native_expert_parity --synthetic-hadamard ${_pair})
          set_tests_properties(native_expert_parity_${_name} native_expert_parity_hadamard_${_name} PROPERTIES TIMEOUT 300)
        endforeach()
      endif()
      # #290: the BF16 token embedding (--embd-gguf) gathered and dequantized bit for bit
      add_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
    endif()
  endif()
  # Round 0214 (the new Unsloth Qwen3.8-Flash-Next quants UD-IQ1_M .. UD-IQ4_XS), CUDA and HIP: every expert type pair the
  # sets use at 2560/640 (GPU decode kernels, the CPU pool and the prompt path's dequantizers against ggml), and the dense
  # tensor types (Q5_0, Q6_K, the K-quants, IQ4_NL, IQ4_XS, Q2_0) through native_mmvq.
  if(STRATA_BUILD_TESTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
    set(_strata_new_gu iq1_m iq2_xxs iq2_xs iq2_s q2_0 iq3_xxs iq4_xs)
    set(_strata_new_dn q2_0 iq4_nl q5_0 q8_0)
    foreach(_g ${_strata_new_gu})
      foreach(_d ${_strata_new_dn})
        add_test(NAME native_expert_parity_new_${_g}_${_d} COMMAND native_expert_parity --synthetic ${_g}/${_d})
        set_tests_properties(native_expert_parity_new_${_g}_${_d} PROPERTIES TIMEOUT 300)
      endforeach()
    endforeach()
    add_executable(native_dense_types_parity src/kernels/native_dense_types_parity.cpp)
    target_include_directories(native_dense_types_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_dense_types_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
    add_test(NAME native_dense_types_parity COMMAND native_dense_types_parity)
    set_tests_properties(native_dense_types_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  endif()
  # #296's real-weight checks (OrcaRouter Q4_K_S files; docs/ORCA_Q4_K_S.md) - test programs, not part of a release
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS)
    add_executable(ple_q5_parity src/kernels/ple_q5_parity.cpp)
    target_link_libraries(ple_q5_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    add_executable(q5_projection_parity src/kernels/q5_projection_parity.cpp)
    target_include_directories(q5_projection_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q5_projection_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    # #291: the FP8 PLE table (tools/ple_fp8_pack.py) through PleTable vs the checkpoint's own values
    add_executable(ple_fp8_parity src/kernels/ple_fp8_parity.cpp)
    target_link_libraries(ple_fp8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()
endif()

if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/hit_cpu_order_parity.cu")
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
endif()

strata_cpu_tests()

# CUDA release kernels must not spill or keep a per-thread array in local memory: every access there is a round trip
# to device memory in the hottest loops the engine has, and nothing on the build machine says so without -Xptxas -v
# or a GPU.  tools/check_cuda_spills.py reads the resource usage the compiler wrote into the kernel libraries
# (cuobjdump: STACK and LOCAL per entry function and per sm_) and fails on any kernel outside its ALLOWED table, which
# carries the measured allowance of the kernels that hold such an array on purpose.  CUDA only: HIP has no cuobjdump.
strata_cuda_spill_guard()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # Model-level checks may be absent from source distributions.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

# ---- HIP gfx906: every CUDA source of the tree (and the ggml-cuda MMQ sources) is compiled as HIP ----
strata_hip_gfx906_sources()

# ---- DeepSeek-V4.1 on a box with no GPU toolkit ------------------------------------------------
# The whole DSV41 block above lives inside if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP), so with neither backend on
# -DSTRATA_BUILD_DSV41=ON was silently unused and the host-side benchmark could not be built at all (found on p1004,
# 2026-10-23: the CPU lane is exactly what a box without a usable GPU needs to measure).  This branch builds the one target
# that has no device code, so the CPU-lane A/B runs anywhere.
strata_cpu_dsv41_host_bench()

# ---- strata-split: the opt-in engine that splits every layer across 2-4 NVIDIA cards (src/split, from sFNX, MIT, John Fink) ----
# OFF by default, Linux + CUDA or HIP only, no other target depends on it; see src/split/CMakeLists.txt and docs/SPLIT_ENGINE.md.
option(STRATA_BUILD_SPLIT "Build strata-split: sFNX's split-every-layer engine, host-memory exchange (Linux, CUDA or HIP gfx1201, 2-4 cards)" OFF)
if(STRATA_BUILD_SPLIT)
  add_subdirectory(src/split)
endif()

# What this configure registered, so a missing capability is a stated skip (tools/check prints this line too).
if(STRATA_BUILD_TESTS)
  set(_strata_test_kinds "cpu")
  if(STRATA_ENABLE_CUDA)
    string(APPEND _strata_test_kinds ", cuda")
  elseif(STRATA_ENABLE_HIP)
    string(APPEND _strata_test_kinds ", hip")
  else()
    string(APPEND _strata_test_kinds " (gpu tests not registered: neither STRATA_ENABLE_CUDA nor STRATA_ENABLE_HIP is on)")
  endif()
  if(NOT STRATA_NATIVE_EXPERTS)
    string(APPEND _strata_test_kinds " (native-expert tests not registered: STRATA_NATIVE_EXPERTS=OFF)")
  endif()
  message(STATUS "strata tests: ON - ${_strata_test_kinds}")
else()
  message(STATUS "strata tests: OFF (STRATA_BUILD_TESTS=OFF)")
endif()
