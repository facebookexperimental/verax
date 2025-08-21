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
include_guard(GLOBAL)

set(VELOX_ANTLR4-RUNTIME_VERSION 4.13.2)
set(VELOX_ANTLR4-RUNTIME_BUILD_SHA256_CHECKSUM
    0ed13668906e86dbc0dcddf30fdee68c10203dea4e83852b4edb810821bee3c4)
set(VELOX_ANTLR4-RUNTIME_SOURCE_URL
    "https://www.antlr.org/download/antlr4-cpp-runtime-${VELOX_ANTLR4-RUNTIME_VERSION}-source.zip"
)

velox_resolve_dependency_url(ANTLR4-RUNTIME)

message(STATUS "Building antlr4-runtime from source")

FetchContent_Declare(
  antlr4-runtime
  URL ${VELOX_ANTLR4-RUNTIME_SOURCE_URL}
  URL_HASH ${VELOX_ANTLR4-RUNTIME_BUILD_SHA256_CHECKSUM})

set(ANTLR4_INSTALL
    OFF
    CACHE BOOL "Disable installation for antlr4-runtime")
set(ANTLR_BUILD_CPP_TESTS
    OFF
    CACHE BOOL "Disable tests for antlr4-runtime")

FetchContent_MakeAvailable(antlr4-runtime)

# Setup ANTLR4 JAR for code generation
find_package(
  Java
  COMPONENTS Runtime
  QUIET)

if(Java_FOUND)
  # Set ANTLR JAR path - try multiple potential locations
  set(ANTLR_JAR_NAME "antlr-${VELOX_ANTLR4-RUNTIME_VERSION}-complete.jar")

  # Check for locally provided JAR first
  set(ANTLR_EXECUTABLE
      "${CMAKE_SOURCE_DIR}/third-party/antlr/${ANTLR_JAR_NAME}"
      CACHE FILEPATH "Path to ANTLR4 JAR file")

  # If JAR doesn't exist, download it to build directory
  if(NOT EXISTS ${ANTLR_EXECUTABLE})
    message(
      STATUS "ANTLR4 JAR not found locally, downloading to build directory...")
    set(ANTLR_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/antlr4")
    file(MAKE_DIRECTORY ${ANTLR_DOWNLOAD_DIR})
    set(ANTLR_EXECUTABLE "${ANTLR_DOWNLOAD_DIR}/${ANTLR_JAR_NAME}")

    if(NOT EXISTS ${ANTLR_EXECUTABLE})
      file(
        DOWNLOAD "https://www.antlr.org/download/${ANTLR_JAR_NAME}"
        ${ANTLR_EXECUTABLE}
        EXPECTED_HASH
          SHA256=548c1c3e654a08bbcaf3e6e8f0dc74bf48ce0e3c79476d2e8d1ca7b34ba46b3e
        SHOW_PROGRESS)
    endif()
  endif()

  # Add the FindANTLR module path from the fetched content
  get_property(
    antlr4_source_dir
    TARGET antlr4_shared
    PROPERTY SOURCE_DIR)
  if(antlr4_source_dir)
    list(APPEND CMAKE_MODULE_PATH "${antlr4_source_dir}/cmake")
  else()
    # Fallback to the FetchContent source directory
    list(APPEND CMAKE_MODULE_PATH "${antlr4-runtime_SOURCE_DIR}/cmake")
  endif()

  # Export variables for use in other CMakeLists.txt files
  set(ANTLR_EXECUTABLE
      ${ANTLR_EXECUTABLE}
      CACHE FILEPATH "Path to ANTLR4 JAR file" FORCE)
  set(ANTLR4_AVAILABLE
      TRUE
      CACHE BOOL "ANTLR4 code generation available" FORCE)

  message(STATUS "ANTLR4 JAR configured at: ${ANTLR_EXECUTABLE}")
else()
  message(
    WARNING "Java not found - ANTLR4 code generation will not be available")
  set(ANTLR4_AVAILABLE
      FALSE
      CACHE BOOL "ANTLR4 code generation available" FORCE)
endif()
