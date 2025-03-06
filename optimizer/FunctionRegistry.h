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
#pragma once

#include "optimizer/QueryGraph.h" //@manual

namespace facebook::velox::optimizer {

class FunctionRegistry {
 public:
  FunctionMetadata* metadata(const std::string& name);

  void registerFunction(
      const std::string& function,
      std::unique_ptr<FunctionMetadata> metadata);

  static FunctionRegistry* instance();

 private:
  std::unordered_map<std::string, std::unique_ptr<FunctionMetadata>> metadata_;
};
} // namespace facebook::velox::optimizer
