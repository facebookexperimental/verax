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

# Check if ANTLR4 code generation is available (configured by
# antlr4-runtime.cmake)
if(ANTLR4_AVAILABLE)
  # Include ANTLR module (path was set up by antlr4-runtime.cmake)
  include(FindANTLR)

  # Generate ANTLR4 target for PrestoSql grammar
  antlr_target(
    PrestoSqlGenerated
    PrestoSql.g4
    VISITOR
    LISTENER
    OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/antlr4_generated)

  # Set include directories for generated files
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/antlr4_generated)

  message(STATUS "ANTLR4 code generation enabled for PrestoSql grammar")
else()
  message(
    WARNING "ANTLR4 code generation not available - Java runtime may be missing"
  )
endif()

add_subdirectory(tests)

# Create the library with the appropriate source files
if(ANTLR4_AVAILABLE)
  # Use auto-generated files
  add_library(velox_presto_grammar_lib ${ANTLR_PrestoSqlGenerated_CXX_OUTPUTS})

  target_include_directories(
    velox_presto_grammar_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/antlr4_generated
                                    ${CMAKE_CURRENT_SOURCE_DIR})
else()
  # Fallback to manually committed files (if they exist)
  message(WARNING "Using manually committed ANTLR4 files as fallback")
  add_library(
    velox_presto_grammar_lib
    PrestoSqlBaseListener.cpp
    PrestoSqlBaseVisitor.cpp
    PrestoSqlLexer.cpp
    PrestoSqlListener.cpp
    PrestoSqlParser.cpp
    PrestoSqlVisitor.cpp)

  target_include_directories(velox_presto_grammar_lib
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Always add dependency on ANTLR4 runtime
add_dependencies(velox_presto_grammar_lib antlr4_shared)

target_link_libraries(
  velox_presto_grammar_lib antlr4_shared)
