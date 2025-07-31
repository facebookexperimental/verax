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
#include "axiom/logical_plan/ExprApi.h"
#include "axiom/logical_plan/LogicalPlanNode.h"
#include "velox/common/base/Exceptions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"

namespace facebook::velox::logical_plan {

namespace {
constexpr auto kNoAlias = std::nullopt;
}

namespace detail {

core::ExprPtr
BinaryCall(std::string name, core::ExprPtr left, core::ExprPtr right) {
  return std::make_shared<const core::CallExpr>(
      std::move(name),
      std::vector<core::ExprPtr>{std::move(left), std::move(right)},
      kNoAlias);
}

} // namespace detail

ExprApi ExprApi::operator+(const Variant& value) const {
  return Plus(expr_, Val(value).expr());
}

ExprApi ExprApi::operator-(const Variant& value) const {
  return Minus(expr_, Val(value).expr());
}

ExprApi ExprApi::operator*(const Variant& value) const {
  return Multiply(expr_, Val(value).expr());
}

ExprApi ExprApi::operator/(const Variant& value) const {
  return Divide(expr_, Val(value).expr());
}

ExprApi ExprApi::operator%(const Variant& value) const {
  return Modulus(expr_, Val(value).expr());
}

ExprApi ExprApi::operator<(const Variant& value) const {
  return Lt(expr_, Val(value).expr());
}

ExprApi ExprApi::operator<=(const Variant& value) const {
  return Lte(expr_, Val(value).expr());
}

ExprApi ExprApi::operator>(const Variant& value) const {
  return Gt(expr_, Val(value).expr());
}

ExprApi ExprApi::operator>=(const Variant& value) const {
  return Gte(expr_, Val(value).expr());
}

ExprApi ExprApi::operator&&(const Variant& value) const {
  return And(expr_, Val(value).expr());
}

ExprApi ExprApi::operator||(const Variant& value) const {
  return Or(expr_, Val(value).expr());
}

ExprApi ExprApi::operator==(const Variant& value) const {
  return Eq(expr_, Val(value).expr());
}

ExprApi ExprApi::operator!=(const Variant& value) const {
  return NEq(expr_, Val(value).expr());
}

ExprApi Col(std::string name) {
  return ExprApi{
      std::make_shared<const core::FieldAccessExpr>(name, kNoAlias),
      std::move(name)};
}

ExprApi Col(std::string name, const ExprApi& input) {
  std::vector<core::ExprPtr> inputs{input.expr()};
  return ExprApi{
      std::make_shared<const core::FieldAccessExpr>(
          name, kNoAlias, std::move(inputs)),
      std::move(name)};
}

ExprApi Cast(velox::TypePtr type, const ExprApi& input) {
  return ExprApi{std::make_shared<const core::CastExpr>(
      type, input.expr(), /* isTryCat */ false, kNoAlias)};
}

ExprApi Val(Variant&& val) {
  auto type = val.inferType();
  return ExprApi{std::make_shared<const core::ConstantExpr>(
      std::move(type), std::move(val), kNoAlias)};
}

ExprApi Val(Variant&& val, velox::TypePtr type) {
  return ExprApi{std::make_shared<const core::ConstantExpr>(
      std::move(type), std::move(val), kNoAlias)};
}

ExprApi Val(const Variant& val) {
  auto type = val.inferType();
  return ExprApi{std::make_shared<const core::ConstantExpr>(
      std::move(type), val, kNoAlias)};
}

ExprApi Val(const Variant& val, TypePtr type) {
  return ExprApi{std::make_shared<const core::ConstantExpr>(
      std::move(type), val, kNoAlias)};
}

ExprApi Call(std::string name, const std::vector<ExprApi>& args) {
  std::vector<core::ExprPtr> argExpr;
  argExpr.reserve(args.size());
  for (auto& arg : args) {
    argExpr.push_back(arg.expr());
  }
  return ExprApi{std::make_shared<const core::CallExpr>(
      std::move(name), std::move(argExpr), kNoAlias)};
}

ExprApi Lambda(std::vector<std::string> names, const ExprApi& body) {
  return ExprApi{
      std::make_shared<core::LambdaExpr>(std::move(names), body.expr())};
}

ExprApi Subquery(std::shared_ptr<const LogicalPlanNode> subquery) {
  VELOX_CHECK_NOT_NULL(subquery);
  VELOX_CHECK_EQ(1, subquery->outputType()->size());

  const auto& name = subquery->outputType()->nameOf(0);
  const auto expr = std::make_shared<core::SubqueryExpr>(std::move(subquery));
  if (name.empty()) {
    return ExprApi(expr);
  } else {
    return ExprApi(expr, name);
  }
}

ExprApi Sql(const std::string& sql) {
  return ExprApi{parse::parseExpr(sql, {})};
}
} // namespace facebook::velox::logical_plan
