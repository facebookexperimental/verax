# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.14)

# Set the project name.
project(Verax)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable generation of compile_commands.json for clang tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(DISABLE_META_INTERNAL_COMPRESSOR=1)

# Sets new behavior for CMP0135, which controls how timestamps are extracted
# when using ExternalProject_Add():
# https://cmake.org/cmake/help/latest/policy/CMP0135.html
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

# Use ThirdPartyToolchain dependencies macros from Velox.
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake"
     "${PROJECT_SOURCE_DIR}/velox/CMake")
include(ResolveDependency)

set(VELOX_BUILD_VECTOR_TEST_UTILS
    ON
    CACHE BOOL "Velox vector test utilities (VectorMaker).")

set(VELOX_ENABLE_PARQUET ON)
set(VELOX_ENABLE_BENCHMARKS ON)
set(VELOX_BUILD_TEST_UTILS ON)
set(VELOX_BUILD_TESTING OFF)

set(VELOX_DEPENDENCY_SOURCE
    AUTO
    CACHE STRING "Default dependency source: AUTO SYSTEM or BUNDLED.")

# Add ccache support option
option(VERAX_ENABLE_CCACHE "Use ccache if installed." ON)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(BOOST_INCLUDE_LIBRARIES
    atomic
    context
    date_time
    filesystem
    program_options
    regex
    system
    thread)

velox_set_source(Boost)
velox_resolve_dependency(Boost 1.77.0 COMPONENTS ${BOOST_INCLUDE_LIBRARIES})

# Ignore known compiler warnings.
check_cxx_compiler_flag("-Wstringop-overread" COMPILER_HAS_W_STRINGOP_OVERREAD)
if(COMPILER_HAS_W_STRINGOP_OVERREAD)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-stringop-overread")
endif()

check_cxx_compiler_flag("-Wdeprecated-declarations"
                        COMPILER_HAS_W_DEPRECATED_DECLARATIONS)
if(COMPILER_HAS_W_DEPRECATED_DECLARATIONS)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")
endif()

check_cxx_compiler_flag("-Wmaybe-uninitialized"
                        COMPILER_HAS_W_MAYBE_UNINITIALIZED)
if(COMPILER_HAS_W_MAYBE_UNINITIALIZED)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-maybe-uninitialized")
endif()

check_cxx_compiler_flag("-Wunknown-warning-option"
                        COMPILER_HAS_W_UNKNOWN_WARNING_OPTION)
if(COMPILER_HAS_W_UNKNOWN_WARNING_OPTION)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unknown-warning-option")
endif()

check_cxx_compiler_flag("-Wnullability-completeness"
                        COMPILER_HAS_W_NULLABILITY_COMPLETENESS)
if(COMPILER_HAS_W_NULLABILITY_COMPLETENESS)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-nullability-completeness")
endif()

# Verax, Velox and folly need to be compiled with the same compiler flags.
execute_process(
  COMMAND
    bash -c
    "( source ${CMAKE_CURRENT_SOURCE_DIR}/velox/scripts/setup-helper-functions.sh && echo -n $(get_cxx_flags $ENV{CPU_TARGET}))"
  OUTPUT_VARIABLE SCRIPT_CXX_FLAGS
  RESULT_VARIABLE COMMAND_STATUS)

if(COMMAND_STATUS EQUAL "1")
  message(FATAL_ERROR "Unable to determine compiler flags!")
endif()
message("Setting CMAKE_CXX_FLAGS=${SCRIPT_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCRIPT_CXX_FLAGS}")

# Configure ccache if enabled
message(STATUS "🔧 Configuring ccache support...")
message(STATUS "  VERAX_ENABLE_CCACHE: ${VERAX_ENABLE_CCACHE}")
message(STATUS "  CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
message(STATUS "  CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
message(STATUS "  CMAKE_CUDA_COMPILER_LAUNCHER: ${CMAKE_CUDA_COMPILER_LAUNCHER}")

if(VERAX_ENABLE_CCACHE
   AND NOT CMAKE_C_COMPILER_LAUNCHER
   AND NOT CMAKE_CXX_COMPILER_LAUNCHER
   AND NOT CMAKE_CUDA_COMPILER_LAUNCHER)

  message(STATUS "🔍 Searching for ccache executable...")
  find_program(CCACHE_FOUND ccache)

  if(CCACHE_FOUND)
    message(STATUS "✅ ccache found and enabled!")
    message(STATUS "  📍 ccache location: ${CCACHE_FOUND}")

    # Check ccache version
    execute_process(
      COMMAND ${CCACHE_FOUND} --version
      OUTPUT_VARIABLE CCACHE_VERSION_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "ccache version [0-9]+\\.[0-9]+\\.[0-9]+" CCACHE_VERSION "${CCACHE_VERSION_OUTPUT}")
    message(STATUS "  📦 ${CCACHE_VERSION}")

    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_FOUND})

    # Configure ccache settings
    set(ENV{CCACHE_COMMENTS} "1")
    message(STATUS "  ⚙️  CCACHE_COMMENTS=1 (preserves comments)")

    # Show ccache configuration
    execute_process(
      COMMAND ${CCACHE_FOUND} --show-config
      OUTPUT_VARIABLE CCACHE_CONFIG
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "  📋 ccache configuration:")
    message(STATUS "     Cache directory: $ENV{CCACHE_DIR}")
    if(NOT "$ENV{CCACHE_DIR}" STREQUAL "")
      message(STATUS "     Using custom CCACHE_DIR: $ENV{CCACHE_DIR}")
    endif()

  else()
    message(WARNING "❌ ccache not found! Install ccache for faster builds:")
    message(WARNING "  📦 On Ubuntu/Debian: sudo apt install ccache")
    message(WARNING "  📦 On macOS: brew install ccache")
    message(WARNING "  📦 On CentOS/RHEL: sudo dnf install ccache")
  endif()

else()
  if(CMAKE_C_COMPILER_LAUNCHER OR CMAKE_CXX_COMPILER_LAUNCHER OR CMAKE_CUDA_COMPILER_LAUNCHER)
    message(STATUS "✅ ccache already configured by build system!")
    message(STATUS "  🎯 Existing compiler launcher detected:")
    message(STATUS "     C: ${CMAKE_C_COMPILER_LAUNCHER}")
    message(STATUS "     CXX: ${CMAKE_CXX_COMPILER_LAUNCHER}")
    message(STATUS "     CUDA: ${CMAKE_CUDA_COMPILER_LAUNCHER}")
    if(CMAKE_CXX_COMPILER_LAUNCHER STREQUAL "ccache")
      message(STATUS "  🚀 ccache is active and will accelerate builds!")
    endif()
  else()
    message(STATUS "❌ ccache disabled")
    if(NOT VERAX_ENABLE_CCACHE)
      message(STATUS "  💡 Enable with: -DVERAX_ENABLE_CCACHE=ON")
    endif()
  endif()
endif()

message("FINAL CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

include(CTest) # include after project() but before add_subdirectory()

velox_set_source(gtest)
velox_resolve_dependency(gtest)

velox_set_source(glog)
velox_resolve_dependency(glog)

velox_set_source(gflags)
velox_resolve_dependency(gflags COMPONENTS shared)

velox_set_source(Boost)
velox_resolve_dependency(Boost 1.77.0 COMPONENTS ${BOOST_INCLUDE_LIBRARIES})

velox_set_source(folly)
velox_resolve_dependency(folly)

# Use xxhash and xsimd from Velox for now.
include_directories(.)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/velox)
include_directories(SYSTEM velox)
include_directories(SYSTEM velox/velox/external/xxhash axiom)
include_directories(SYSTEM ${CMAKE_BINARY_DIR}/_deps/xsimd-src/include/)

add_subdirectory(velox)
add_subdirectory(axiom/logical_plan)
add_subdirectory(axiom/runner)
add_subdirectory(axiom/optimizer)
