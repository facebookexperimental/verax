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

velox_add_library(velox_connector_metadata ConnectorMetadata.cpp
                  StatisticsBuilder.cpp)

velox_link_libraries(velox_connector_metadata velox_common_base velox_memory
                     velox_connector velox_dwio_dwrf_writer)

add_subdirectory(hive)

add_subdirectory(tests)

velox_add_library(velox_connector_split_source ConnectorSplitSource.cpp)

velox_link_libraries(
  velox_connector_split_source
  velox_connector_metadata
  velox_local_runner
  velox_multifragment_plan
  velox_common_base
  velox_memory
  velox_hive_connector
  velox_dwio_common
  velox_dwio_dwrf_writer
  velox_exec
  velox_cursor)
