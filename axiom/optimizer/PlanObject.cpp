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

#include "axiom/optimizer/QueryGraph.h"

namespace facebook::axiom::optimizer {

namespace {
const auto& planTypeNames() {
  static const folly::F14FastMap<PlanType, std::string_view> kNames = {
      {PlanType::kColumnExpr, "ColumnExpr"},
      {PlanType::kLiteralExpr, "LiteralExpr"},
      {PlanType::kCallExpr, "CallExpr"},
      {PlanType::kAggregateExpr, "AggregateExpr"},
      {PlanType::kFieldExpr, "FieldExpr"},
      {PlanType::kLambdaExpr, "LambdaExpr"},
      {PlanType::kTableNode, "TableNode"},
      {PlanType::kValuesTableNode, "ValuesTableNode"},
      {PlanType::kUnnestTableNode, "UnnestTableNode"},
      {PlanType::kDerivedTableNode, "DerivedTableNode"},
      {PlanType::kAggregationNode, "AggregationNode"},
      {PlanType::kProjectNode, "ProjectNode"},
      {PlanType::kFilterNode, "FilterNode"},
      {PlanType::kJoinNode, "JoinNode"},
      {PlanType::kOrderByNode, "OrderByNode"},
      {PlanType::kLimitNode, "LimitNode"},
  };
  return kNames;
}
} // namespace

AXIOM_DEFINE_ENUM_NAME(PlanType, planTypeNames);

size_t PlanObject::hash() const {
  auto h = static_cast<size_t>(id_);
  for (auto& child : children()) {
    h = velox::bits::hashMix(h, child->hash());
  }
  return h;
}

void PlanObjectSet::unionColumns(ExprCP expr) {
  switch (expr->type()) {
    case PlanType::kLiteralExpr:
      return;
    case PlanType::kColumnExpr:
      add(expr);
      return;
    case PlanType::kFieldExpr:
      unionColumns(expr->as<Field>()->base());
      return;
    case PlanType::kAggregateExpr: {
      auto condition = expr->as<Aggregate>()->condition();
      if (condition) {
        unionColumns(condition);
      }
    }
      [[fallthrough]];
    case PlanType::kCallExpr: {
      auto call = reinterpret_cast<const Call*>(expr);
      unionSet(call->columns());
      return;
    }
    case PlanType::kWindowExpr: {
      auto window = expr->as<Window>();
      unionSet(window->columns());
      return;
    }
    default:
      VELOX_UNREACHABLE();
  }
}

void PlanObjectSet::unionColumns(const ExprVector& exprs) {
  for (auto& expr : exprs) {
    unionColumns(expr);
  }
}

std::string PlanObjectSet::toString(bool names) const {
  std::stringstream out;
  forEach([&](auto object) {
    out << object->id();
    if (names) {
      out << ": " << object->toString() << std::endl;
    } else {
      out << " ";
    }
  });
  return out.str();
}

// Debug helper functions. Must be extern to be callable from debugger.

extern std::string planObjectString(const PlanObject* o) {
  return o->toString();
}

} // namespace facebook::axiom::optimizer
