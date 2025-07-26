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

#include "axiom/logical_plan/LogicalPlanNode.h"
#include "axiom/logical_plan/NameAllocator.h"
#include "axiom/logical_plan/NameMappings.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/PlanNodeIdGenerator.h"

namespace facebook::velox::logical_plan {

class PlanBuilder {
 public:
  struct Context {
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator;
    std::shared_ptr<NameAllocator> nameAllocator;

    Context()
        : planNodeIdGenerator{std::make_shared<core::PlanNodeIdGenerator>()},
          nameAllocator{std::make_shared<NameAllocator>()} {}
  };

  PlanBuilder()
      : planNodeIdGenerator_(std::make_shared<core::PlanNodeIdGenerator>()),
        nameAllocator_(std::make_shared<NameAllocator>()) {}

  explicit PlanBuilder(const Context& context)
      : planNodeIdGenerator_{context.planNodeIdGenerator},
        nameAllocator_{context.nameAllocator} {
    VELOX_CHECK_NOT_NULL(planNodeIdGenerator_);
    VELOX_CHECK_NOT_NULL(nameAllocator_);
  }

  explicit PlanBuilder(
      const Context& context,
      std::vector<std::shared_ptr<NameMappings>> outputMapping)
      : planNodeIdGenerator_{context.planNodeIdGenerator},
        nameAllocator_{context.nameAllocator} {
    VELOX_CHECK_NOT_NULL(planNodeIdGenerator_);
    VELOX_CHECK_NOT_NULL(nameAllocator_);
    outputMapping_ = outputMapping;
  }

  PlanBuilder& values(const RowTypePtr& rowType, std::vector<Variant> rows);

  PlanBuilder& tableScan(
      const std::string& connectorId,
      const std::string& tableName,
      const std::vector<std::string>& columnNames);

  PlanBuilder& filter(const std::string& predicate);

  PlanBuilder& project(const std::vector<std::string>& projections);

  /// An alias for 'project'.
  PlanBuilder& map(const std::vector<std::string>& projections) {
    return project(projections);
  }

  /// Similar to 'project', but appends 'projections' to the existing columns.
  PlanBuilder& with(const std::vector<std::string>& projections);

  // Create a projection mapping subquery output to name corresponding to hints
  PlanBuilder& with(
      const std::unordered_map<ExprPtr, std::string>& subqueryOutput);

  PlanBuilder& aggregate(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates);

  PlanBuilder& join(
      const PlanBuilder& right,
      const std::string& condition,
      JoinType joinType);

  PlanBuilder& unionAll(const PlanBuilder& other);

  PlanBuilder& sort(const std::vector<std::string>& sortingKeys);

  /// An alias for 'sort'.
  PlanBuilder& orderBy(const std::vector<std::string>& sortingKeys) {
    return sort(sortingKeys);
  }

  PlanBuilder& limit(int32_t count) {
    return limit(0, count);
  }

  PlanBuilder& limit(int32_t offset, int32_t count);

  PlanBuilder& as(const std::string& alias);

  ExprPtr buildSubquery(
      LogicalPlanNodePtr subquery,
      std::optional<SpecialForm> form,
      const std::optional<std::string>& input);

  LogicalPlanNodePtr build();

  std::vector<std::shared_ptr<NameMappings>> getOutputMapping() const {
    return outputMapping_;
  };

 private:
  std::string nextId() {
    return planNodeIdGenerator_->next();
  }

  std::string newName(const std::string& hint);

  ExprPtr resolveInputName(
      const std::optional<std::string>& alias,
      const std::string& name) const;

  ExprPtr resolveScalarTypes(const core::ExprPtr& expr) const;

  AggregateExprPtr resolveAggregateTypes(const core::ExprPtr& expr) const;

  void resolveProjections(
      const std::vector<std::string>& projections,
      std::vector<std::string>& outputNames,
      std::vector<ExprPtr>& exprs,
      NameMappings& mappings);

  std::vector<NameMappings::QualifiedName> reverseLookup(const std::string& id);

  const std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator_;
  const std::shared_ptr<NameAllocator> nameAllocator_;
  const parse::ParseOptions parseOptions_;

  LogicalPlanNodePtr node_;

  // Mapping from user-provided to auto-generated output column names.
  std::vector<std::shared_ptr<NameMappings>> outputMapping_;
};

} // namespace facebook::velox::logical_plan
