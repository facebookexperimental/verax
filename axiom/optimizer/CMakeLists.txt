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

add_subdirectory(tests)

add_subdirectory(connectors)

add_library(velox_schema_resolver SchemaResolver.cpp SchemaUtils.cpp)

target_link_libraries(velox_schema_resolver velox_connector_metadata
                      velox_exception velox_vector)

add_library(
  velox_verax
  DerivedTable.cpp
  ToGraph.cpp
  LogicalPlanToGraph.cpp
  Subfields.cpp
  LogicalPlanSubfields.cpp
  Plan.cpp
  BitSet.cpp
  ParallelExpr.cpp
  PlanObject.cpp
  Schema.cpp
  QueryGraph.cpp
  QueryGraphContext.cpp
  Filters.cpp
  Cost.cpp
  FunctionRegistry.cpp
  PlanUtils.cpp
  RelationOp.cpp
  ToVelox.cpp
  VeloxHistory.cpp
  JoinSample.cpp)

add_dependencies(velox_verax velox_hive_connector)

target_link_libraries(
  velox_verax
  velox_core
  velox_connector_metadata
  velox_fe_logical_plan
  velox_multifragment_plan
  velox_connector
  velox_schema_resolver)
