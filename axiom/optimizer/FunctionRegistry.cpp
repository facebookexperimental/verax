/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "axiom/optimizer/FunctionRegistry.h"

namespace facebook::velox::optimizer {

FunctionMetadata* FunctionRegistry::metadata(const std::string& name) {
  auto it = metadata_.find(name);
  if (it == metadata_.end()) {
    return nullptr;
  }
  return it->second.get();
}

void FunctionRegistry::registerFunction(
    const std::string& name,
    std::unique_ptr<FunctionMetadata> metadata) {
  metadata_[name] = std::move(metadata);
}

// static
FunctionRegistry* FunctionRegistry::instance() {
  static auto registry = std::make_unique<FunctionRegistry>();
  return registry.get();
}

const FunctionMetadata* functionMetadata(Name name) {
  return FunctionRegistry::instance()->metadata(name);
}

bool declareBuiltIn() {
  {
    LambdaInfo info{
        .ordinal = 1,
        .lambdaArg = {LambdaArg::kKey, LambdaArg::kValue},
        .argOrdinal = {0, 0}};

    auto metadata = std::make_unique<FunctionMetadata>();
    metadata->lambdas.push_back(std::move(info));
    metadata->subfieldArg = 0;
    metadata->cost = 40;
    FunctionRegistry::instance()->registerFunction(
        "transform_values", std::move(metadata));
  }
  {
    LambdaInfo info{
        .ordinal = 1, .lambdaArg = {LambdaArg::kElement}, .argOrdinal = {0}};

    auto metadata = std::make_unique<FunctionMetadata>();
    metadata->lambdas.push_back(std::move(info));
    metadata->subfieldArg = 0;
    metadata->cost = 20;
    FunctionRegistry::instance()->registerFunction(
        "transform", std::move(metadata));
  }
  {
    LambdaInfo info{
        .ordinal = 2,
        .lambdaArg = {LambdaArg::kElement, LambdaArg::kElement},
        .argOrdinal = {0, 1}};

    auto metadata = std::make_unique<FunctionMetadata>();
    metadata->lambdas.push_back(std::move(info));
    metadata->cost = 20;
    FunctionRegistry::instance()->registerFunction("zip", std::move(metadata));
  }
  return true;
}

namespace {
bool temp = declareBuiltIn();
}

} // namespace facebook::velox::optimizer
