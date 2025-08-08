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

#include "velox/core/PlanNode.h"
#include "velox/type/Filter.h"

namespace facebook::velox::core {

class PlanMatcher {
 public:
  virtual ~PlanMatcher() = default;

  virtual bool match(const PlanNodePtr& plan) const = 0;
};

class PlanMatcherBuilder {
 public:
  PlanMatcherBuilder& tableScan();

  PlanMatcherBuilder& tableScan(const std::string& tableName);

  PlanMatcherBuilder& hiveScan(
      const std::string& tableName,
      common::SubfieldFilters subfieldFilters,
      const std::string& remainingFilter = "");

  PlanMatcherBuilder& filter();

  PlanMatcherBuilder& filter(const std::string& predicate);

  PlanMatcherBuilder& project();

  PlanMatcherBuilder& aggregation();

  PlanMatcherBuilder& partialAggregation();

  PlanMatcherBuilder& finalAggregation();

  PlanMatcherBuilder& hashJoin(
      const std::shared_ptr<PlanMatcher>& rightMatcher);

  PlanMatcherBuilder& hashJoin(
      const std::shared_ptr<PlanMatcher>& rightMatcher,
      JoinType joinType);

  PlanMatcherBuilder& localPartition();

  PlanMatcherBuilder& localPartition(
      const std::shared_ptr<PlanMatcher>& matcher);

  PlanMatcherBuilder& partitionedOutput();

  PlanMatcherBuilder& exchange();

  PlanMatcherBuilder& limit();

  PlanMatcherBuilder& limit(int64_t offset, int64_t count);

  std::shared_ptr<PlanMatcher> build() {
    return matcher_;
  }

 private:
  std::shared_ptr<PlanMatcher> matcher_;
};

} // namespace facebook::velox::core
