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

#include "axiom/logical_plan/PlanBuilder.h"
#include "axiom/logical_plan/NameMappings.h"
#include "axiom/optimizer/connectors/ConnectorMetadata.h"
#include "velox/connectors/Connector.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/parse/Expressions.h"

namespace facebook::velox::logical_plan {

PlanBuilder& PlanBuilder::values(
    const RowTypePtr& rowType,
    std::vector<Variant> rows) {
  VELOX_USER_CHECK_NULL(node_, "Values node must be the leaf node");

  outputMapping_.push_back(std::make_shared<NameMappings>());

  const auto numColumns = rowType->size();
  std::vector<std::string> outputNames;
  outputNames.reserve(numColumns);
  for (int i = 0; i < rowType->size(); ++i) {
    outputNames.push_back(newName(rowType->nameOf(i)));
    outputMapping_.back()->add(
        rowType->nameOf(i), outputNames.back(), rowType->childAt(i));
  }
  node_ = std::make_shared<ValuesNode>(
      nextId(), ROW(outputNames, rowType->children()), std::move(rows));

  return *this;
}

PlanBuilder& PlanBuilder::tableScan(
    const std::string& connectorId,
    const std::string& tableName,
    const std::vector<std::string>& columnNames) {
  VELOX_USER_CHECK_NULL(node_, "Table scan node must be the leaf node");

  auto* metadata = connector::getConnector(connectorId)->metadata();
  auto* table = metadata->findTable(tableName);
  const auto& schema = table->rowType();

  const auto numColumns = columnNames.size();

  std::vector<TypePtr> columnTypes;
  columnTypes.reserve(numColumns);

  std::vector<std::string> outputNames;
  outputNames.reserve(numColumns);

  outputMapping_.push_back(std::make_shared<NameMappings>());

  for (const auto& name : columnNames) {
    columnTypes.push_back(schema->findChild(name));

    outputNames.push_back(newName(name));
    outputMapping_.back()->add(
        name, outputNames.back(), schema->findChild(name));
  }

  node_ = std::make_shared<TableScanNode>(
      nextId(),
      ROW(outputNames, columnTypes),
      connectorId,
      tableName,
      columnNames);

  return *this;
}

PlanBuilder& PlanBuilder::filter(const std::string& predicate) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Filter node cannot be a leaf node");

  auto untypedExpr = parse::parseExpr(predicate, parseOptions_);
  auto expr = resolveScalarTypes(untypedExpr);

  node_ = std::make_shared<FilterNode>(nextId(), node_, expr);

  return *this;
}

namespace {
std::optional<std::string> tryGetRootName(const core::ExprPtr& expr) {
  if (const auto* fieldAccess =
          dynamic_cast<const core::FieldAccessExpr*>(expr.get())) {
    if (fieldAccess->isRootColumn()) {
      return fieldAccess->name();
    }
  }

  return std::nullopt;
}
} // namespace

void PlanBuilder::resolveProjections(
    const std::vector<std::string>& projections,
    std::vector<std::string>& outputNames,
    std::vector<ExprPtr>& exprs,
    NameMappings& mappings) {
  for (const auto& sql : projections) {
    auto untypedExpr = parse::parseExpr(sql, parseOptions_);
    auto expr = resolveScalarTypes(untypedExpr);

    if (untypedExpr->alias().has_value()) {
      const auto& alias = untypedExpr->alias().value();
      outputNames.push_back(newName(alias));
      mappings.add(alias, outputNames.back(), expr->type());
    } else if (expr->isInputReference()) {
      // Identity projection
      const auto& id = expr->asUnchecked<InputReferenceExpr>()->name();
      outputNames.push_back(id);

      auto names = reverseLookup(id);
      VELOX_USER_CHECK(!names.empty());

      for (const auto& name : names) {
        mappings.add(name, id, expr->type());
      }
    } else {
      outputNames.push_back(newName("expr"));
    }

    exprs.push_back(expr);
  }
}

PlanBuilder& PlanBuilder::project(const std::vector<std::string>& projections) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Project node cannot be a leaf node");

  std::vector<std::string> outputNames;
  outputNames.reserve(projections.size());

  std::vector<ExprPtr> exprs;
  exprs.reserve(projections.size());

  auto newOutputMapping = std::make_shared<NameMappings>();

  resolveProjections(projections, outputNames, exprs, *newOutputMapping);

  node_ = std::make_shared<ProjectNode>(nextId(), node_, outputNames, exprs);
  outputMapping_.back() = newOutputMapping;

  return *this;
}

PlanBuilder& PlanBuilder::with(const std::vector<std::string>& projections) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Project node cannot be a leaf node");

  std::vector<std::string> outputNames;
  outputNames.reserve(projections.size());

  std::vector<ExprPtr> exprs;
  exprs.reserve(projections.size());

  auto newOutputMapping = std::make_shared<NameMappings>();

  const auto& inputType = node_->outputType();

  for (auto i = 0; i < inputType->size(); i++) {
    const auto& id = inputType->nameOf(i);

    outputNames.push_back(id);

    auto names = reverseLookup(id);
    for (const auto& name : names) {
      newOutputMapping->add(name, id, inputType->childAt(i));
    }

    exprs.push_back(
        std::make_shared<InputReferenceExpr>(inputType->childAt(i), id));
  }

  resolveProjections(projections, outputNames, exprs, *newOutputMapping);

  node_ = std::make_shared<ProjectNode>(nextId(), node_, outputNames, exprs);
  outputMapping_.back() = newOutputMapping;

  return *this;
}

PlanBuilder PlanBuilder::subqueryBuilder() {
  Context context;
  context.planNodeIdGenerator = planNodeIdGenerator_;
  context.nameAllocator = nameAllocator_;
  return PlanBuilder(context, outputMapping_);
}

PlanBuilder& PlanBuilder::aggregate(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Aggregate node cannot be a leaf node");

  std::vector<std::string> outputNames;
  outputNames.reserve(groupingKeys.size() + aggregates.size());

  std::vector<ExprPtr> keyExprs;
  keyExprs.reserve(groupingKeys.size());

  auto newOutputMapping = std::make_shared<NameMappings>();

  resolveProjections(groupingKeys, outputNames, keyExprs, *newOutputMapping);

  std::vector<AggregateExprPtr> exprs;
  exprs.reserve(aggregates.size());

  for (const auto& sql : aggregates) {
    auto untypedExpr = parse::parseExpr(sql, parseOptions_);
    auto expr = resolveAggregateTypes(untypedExpr);

    if (untypedExpr->alias().has_value()) {
      const auto& alias = untypedExpr->alias().value();
      outputNames.push_back(newName(alias));
      newOutputMapping->add(alias, outputNames.back(), expr->type());
    } else {
      outputNames.push_back(newName(expr->name()));
    }

    exprs.push_back(expr);
  }

  node_ = std::make_shared<AggregateNode>(
      nextId(),
      node_,
      keyExprs,
      std::vector<AggregateNode::GroupingSet>{},
      exprs,
      outputNames);

  outputMapping_.back() = newOutputMapping;

  return *this;
}

namespace {

std::string nameMappingListToString(
    const std::vector<std::shared_ptr<NameMappings>>& mappings) {
  std::vector<std::string> mappingStrings;
  mappingStrings.reserve(mappings.size());

  for (const auto& mapping : mappings) {
    mappingStrings.push_back(mapping->toString());
  }

  return fmt::format("{}", fmt::join(mappingStrings, ", "));
}

ExprPtr resolveJoinInputName(
    const std::optional<std::string>& alias,
    const std::string& name,
    const std::vector<std::shared_ptr<NameMappings>>& mapping,
    const RowTypePtr& inputRowType) {
  if (alias.has_value()) {
    for (auto i = mapping.size() - 1; i >= 0; --i) {
      if (auto id = mapping[i]->lookup(alias.value(), name)) {
        return std::make_shared<InputReferenceExpr>(
            inputRowType->findChild(id.value().id), id.value().id);
      }
    }

    return nullptr;
  }

  for (auto i = mapping.size() - 1; i >= 0; --i) {
    if (auto id = mapping[i]->lookup(name)) {
      return std::make_shared<InputReferenceExpr>(
          inputRowType->findChild(id.value().id), id.value().id);
    }
  }

  VELOX_USER_FAIL(
      "Cannot resolve column in join input: {} not found in [{}]",
      NameMappings::QualifiedName{alias, name}.toString(),
      nameMappingListToString(mapping));
}

std::string toString(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  std::ostringstream signature;
  signature << functionName << "(";
  for (auto i = 0; i < argTypes.size(); i++) {
    if (i > 0) {
      signature << ", ";
    }
    signature << argTypes[i]->toString();
  }
  signature << ")";
  return signature.str();
}

std::string toString(
    const std::vector<const exec::FunctionSignature*>& signatures) {
  std::stringstream out;
  for (auto i = 0; i < signatures.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << signatures[i]->toString();
  }
  return out.str();
}

TypePtr resolveScalarFunction(
    const std::string& name,
    const std::vector<TypePtr>& argTypes) {
  if (auto type = resolveFunction(name, argTypes)) {
    return type;
  }

  auto allSignatures = getFunctionSignatures();
  auto it = allSignatures.find(name);
  if (it == allSignatures.end()) {
    VELOX_USER_FAIL("Scalar function doesn't exist: {}.", name);
  } else {
    const auto& functionSignatures = it->second;
    VELOX_USER_FAIL(
        "Scalar function signature is not supported: {}. Supported signatures: {}.",
        toString(name, argTypes),
        toString(functionSignatures));
  }
}

ExprPtr tryResolveSpecialForm(
    const std::string& name,
    const std::vector<ExprPtr>& resolvedInputs) {
  if (name == "and") {
    return std::make_shared<SpecialFormExpr>(
        BOOLEAN(), SpecialForm::kAnd, resolvedInputs);
  }

  if (name == "or") {
    return std::make_shared<SpecialFormExpr>(
        BOOLEAN(), SpecialForm::kOr, resolvedInputs);
  }

  if (name == "try") {
    return std::make_shared<SpecialFormExpr>(
        resolvedInputs.at(0)->type(), SpecialForm::kTry, resolvedInputs);
  }

  if (name == "coalesce") {
    return std::make_shared<SpecialFormExpr>(
        resolvedInputs.at(0)->type(), SpecialForm::kCoalesce, resolvedInputs);
  }

  if (name == "if") {
    return std::make_shared<SpecialFormExpr>(
        resolvedInputs.at(1)->type(), SpecialForm::kIf, resolvedInputs);
  }

  if (name == "switch") {
    return std::make_shared<SpecialFormExpr>(
        resolvedInputs.at(1)->type(), SpecialForm::kSwitch, resolvedInputs);
  }

  if (name == "subscript" && resolvedInputs.at(0)->type()->isRow()) {
    VELOX_USER_CHECK_EQ(2, resolvedInputs.size());

    const auto& rowType = resolvedInputs.at(0)->type()->asRow();

    const auto& fieldExpr = resolvedInputs.at(1);
    VELOX_USER_CHECK(fieldExpr->isConstant());
    VELOX_USER_CHECK_EQ(TypeKind::BIGINT, fieldExpr->type()->kind());

    const auto index =
        fieldExpr->asUnchecked<ConstantExpr>()->value()->value<int64_t>();

    VELOX_USER_CHECK_GE(index, 1);
    VELOX_USER_CHECK_LE(index, rowType.size());

    const int32_t zeroBasedIndex = index - 1;

    std::vector<ExprPtr> newInputs = {
        resolvedInputs.at(0),
        std::make_shared<ConstantExpr>(
            INTEGER(), std::make_shared<Variant>(zeroBasedIndex))};

    return std::make_shared<SpecialFormExpr>(
        rowType.childAt(zeroBasedIndex), SpecialForm::kDereference, newInputs);
  }

  if (name == "exists") {
    VELOX_USER_CHECK_EQ(1, resolvedInputs.size());
    return std::make_shared<SpecialFormExpr>(
        BOOLEAN(), SpecialForm::kExists, resolvedInputs);
  }

  return nullptr;
}

using InputNameResolver = std::function<ExprPtr(
    const std::optional<std::string>& alias,
    const std::string& fieldName)>;

ExprPtr resolveScalarTypesImpl(
    const core::ExprPtr& expr,
    const InputNameResolver& inputNameResolver);

ExprPtr resolveLambdaExpr(
    const core::LambdaExpr* lambdaExpr,
    const std::vector<TypePtr>& lambdaInputTypes,
    const InputNameResolver& inputNameResolver) {
  const auto& names = lambdaExpr->arguments();
  const auto& body = lambdaExpr->body();

  VELOX_CHECK_LE(names.size(), lambdaInputTypes.size());
  std::vector<TypePtr> types;
  types.reserve(names.size());
  for (auto i = 0; i < names.size(); ++i) {
    types.push_back(lambdaInputTypes[i]);
  }

  auto signature =
      ROW(std::vector<std::string>(names), std::vector<TypePtr>(types));
  auto lambdaResolver = [inputNameResolver, signature](
                            const std::optional<std::string>& alias,
                            const std::string& fieldName) -> ExprPtr {
    if (!alias.has_value()) {
      auto maybeIdx = signature->getChildIdxIfExists(fieldName);
      if (maybeIdx.has_value()) {
        return std::make_shared<InputReferenceExpr>(
            signature->childAt(maybeIdx.value()), fieldName);
      }
    }
    return inputNameResolver(alias, fieldName);
  };

  return std::make_shared<LambdaExpr>(
      signature, resolveScalarTypesImpl(body, lambdaResolver));
}

bool isLambdaArgument(const exec::TypeSignature& typeSignature) {
  return typeSignature.baseName() == "function";
}

bool hasLambdaArgument(const exec::FunctionSignature& signature) {
  for (const auto& type : signature.argumentTypes()) {
    if (isLambdaArgument(type)) {
      return true;
    }
  }

  return false;
}

bool isLambdaArgument(const exec::TypeSignature& typeSignature, int numInputs) {
  return isLambdaArgument(typeSignature) &&
      (typeSignature.parameters().size() == numInputs + 1);
}

bool isLambdaSignature(
    const exec::FunctionSignature* signature,
    const std::shared_ptr<const core::CallExpr>& callExpr) {
  if (!hasLambdaArgument(*signature)) {
    return false;
  }

  const auto numArguments = callExpr->inputs().size();

  if (numArguments != signature->argumentTypes().size()) {
    return false;
  }

  bool match = true;
  for (auto i = 0; i < numArguments; ++i) {
    if (auto lambda =
            dynamic_cast<const core::LambdaExpr*>(callExpr->inputAt(i).get())) {
      const auto numLambdaInputs = lambda->arguments().size();
      const auto& argumentType = signature->argumentTypes()[i];
      if (!isLambdaArgument(argumentType, numLambdaInputs)) {
        match = false;
        break;
      }
    }
  }

  return match;
}

const exec::FunctionSignature* FOLLY_NULLABLE findLambdaSignature(
    const std::vector<std::shared_ptr<exec::AggregateFunctionSignature>>&
        signatures,
    const std::shared_ptr<const core::CallExpr>& callExpr) {
  const exec::FunctionSignature* matchingSignature = nullptr;
  for (const auto& signature : signatures) {
    if (isLambdaSignature(signature.get(), callExpr)) {
      VELOX_CHECK_NULL(
          matchingSignature,
          "Cannot resolve ambiguous lambda function signatures for {}.",
          callExpr->name());
      matchingSignature = signature.get();
    }
  }

  return matchingSignature;
}

const exec::FunctionSignature* FOLLY_NULLABLE findLambdaSignature(
    const std::vector<const exec::FunctionSignature*>& signatures,
    const std::shared_ptr<const core::CallExpr>& callExpr) {
  const exec::FunctionSignature* matchingSignature = nullptr;
  for (const auto& signature : signatures) {
    if (isLambdaSignature(signature, callExpr)) {
      VELOX_CHECK_NULL(
          matchingSignature,
          "Cannot resolve ambiguous lambda function signatures for {}.",
          callExpr->name());
      matchingSignature = signature;
    }
  }

  return matchingSignature;
}

const exec::FunctionSignature* findLambdaSignature(
    const std::shared_ptr<const core::CallExpr>& callExpr) {
  // Look for a scalar lambda function.
  auto scalarSignatures = getFunctionSignatures(callExpr->name());
  if (!scalarSignatures.empty()) {
    return findLambdaSignature(scalarSignatures, callExpr);
  }

  // Look for an aggregate lambda function.
  if (auto signatures =
          exec::getAggregateFunctionSignatures(callExpr->name())) {
    return findLambdaSignature(signatures.value(), callExpr);
  }

  return nullptr;
}

ExprPtr tryResolveCallWithLambdas(
    const std::shared_ptr<const core::CallExpr>& callExpr,
    const InputNameResolver& inputNameResolver) {
  if (callExpr == nullptr) {
    return nullptr;
  }
  auto signature = findLambdaSignature(callExpr);

  if (signature == nullptr) {
    return nullptr;
  }

  // Resolve non-lambda arguments first.
  auto numArgs = callExpr->inputs().size();
  std::vector<ExprPtr> children(numArgs);
  std::vector<TypePtr> childTypes(numArgs);
  for (auto i = 0; i < numArgs; ++i) {
    if (!isLambdaArgument(signature->argumentTypes()[i])) {
      children[i] =
          resolveScalarTypesImpl(callExpr->inputAt(i), inputNameResolver);
      childTypes[i] = children[i]->type();
    }
  }

  // Resolve lambda arguments.
  exec::SignatureBinder binder(*signature, childTypes);
  binder.tryBind();
  for (auto i = 0; i < numArgs; ++i) {
    auto argSignature = signature->argumentTypes()[i];
    if (isLambdaArgument(argSignature)) {
      std::vector<TypePtr> lambdaTypes;
      for (auto j = 0; j < argSignature.parameters().size() - 1; ++j) {
        auto type = binder.tryResolveType(argSignature.parameters()[j]);
        if (type == nullptr) {
          return nullptr;
        }
        lambdaTypes.push_back(type);
      }

      children[i] = resolveLambdaExpr(
          dynamic_cast<const core::LambdaExpr*>(callExpr->inputs()[i].get()),
          lambdaTypes,
          inputNameResolver);
    }
  }

  std::vector<TypePtr> types;
  types.reserve(children.size());
  for (auto& child : children) {
    types.push_back(child->type());
  }

  auto returnType = resolveScalarFunction(callExpr->name(), types);

  return std::make_shared<CallExpr>(returnType, callExpr->name(), children);
}

ExprPtr resolveScalarTypesImpl(
    const core::ExprPtr& expr,
    const InputNameResolver& inputNameResolver) {
  if (const auto* fieldAccess =
          dynamic_cast<const core::FieldAccessExpr*>(expr.get())) {
    const auto& name = fieldAccess->name();

    if (fieldAccess->isRootColumn()) {
      return inputNameResolver(std::nullopt, name);
    }

    if (auto rootName = tryGetRootName(fieldAccess->input())) {
      if (auto resolved = inputNameResolver(rootName, name)) {
        return resolved;
      }
    }

    auto input =
        resolveScalarTypesImpl(fieldAccess->input(), inputNameResolver);

    return std::make_shared<SpecialFormExpr>(
        input->type()->asRow().findChild(name),
        SpecialForm::kDereference,
        std::vector<ExprPtr>{
            input,
            std::make_shared<ConstantExpr>(
                VARCHAR(), std::make_shared<Variant>(name))});
  }

  if (const auto& constant =
          dynamic_cast<const core::ConstantExpr*>(expr.get())) {
    return std::make_shared<ConstantExpr>(
        constant->type(), std::make_shared<Variant>(constant->value()));
  }

  if (auto lambdaCall = tryResolveCallWithLambdas(
          std::dynamic_pointer_cast<const core::CallExpr>(expr),
          inputNameResolver)) {
    return lambdaCall;
  }

  std::vector<ExprPtr> inputs;
  inputs.reserve(expr->inputs().size());
  for (const auto& input : expr->inputs()) {
    inputs.push_back(resolveScalarTypesImpl(input, inputNameResolver));
  }

  if (const auto* call = dynamic_cast<const core::CallExpr*>(expr.get())) {
    const auto& name = call->name();

    if (auto specialForm = tryResolveSpecialForm(name, inputs)) {
      return specialForm;
    }

    std::vector<TypePtr> inputTypes;
    inputTypes.reserve(inputs.size());
    for (const auto& input : inputs) {
      inputTypes.push_back(input->type());
    }

    auto type = resolveScalarFunction(name, inputTypes);

    return std::make_shared<CallExpr>(type, name, inputs);
  }

  if (const auto* cast = dynamic_cast<const core::CastExpr*>(expr.get())) {
    return std::make_shared<SpecialFormExpr>(
        cast->type(),
        cast->isTryCast() ? SpecialForm::kTryCast : SpecialForm::kCast,
        inputs);
  }

  VELOX_NYI("Can't resolve {}", expr->toString());
}

AggregateExprPtr resolveAggregateTypesImpl(
    const core::ExprPtr& expr,
    const InputNameResolver& inputNameResolver) {
  const auto* call = dynamic_cast<const core::CallExpr*>(expr.get());
  VELOX_USER_CHECK_NOT_NULL(call, "Aggregate must be a call expression");

  const auto& name = call->name();

  std::vector<ExprPtr> inputs;
  inputs.reserve(expr->inputs().size());
  for (const auto& input : expr->inputs()) {
    inputs.push_back(resolveScalarTypesImpl(input, inputNameResolver));
  }

  std::vector<TypePtr> inputTypes;
  inputTypes.reserve(inputs.size());
  for (const auto& input : inputs) {
    inputTypes.push_back(input->type());
  }

  if (auto type = exec::resolveAggregateFunction(name, inputTypes).first) {
    return std::make_shared<AggregateExpr>(type, name, inputs);
  }

  auto allSignatures = exec::getAggregateFunctionSignatures();
  auto it = allSignatures.find(name);
  if (it == allSignatures.end()) {
    VELOX_USER_FAIL("Aggregate function doesn't exist: {}.", name);
  } else {
    const auto& functionSignatures = it->second;
    VELOX_USER_FAIL(
        "Aggregate function signature is not supported: {}. Supported signatures: {}.",
        toString(name, inputTypes),
        toString(functionSignatures));
  }
}

} // namespace

PlanBuilder& PlanBuilder::join(
    const PlanBuilder& right,
    const std::string& condition,
    JoinType joinType) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Join node cannot be a leaf node");
  VELOX_USER_CHECK_NOT_NULL(right.node_);
  VELOX_CHECK_EQ(outputMapping_.size(), right.outputMapping_.size());

  // User-facing column names may have duplicates between left and right side.
  // Columns that are unique can be referenced as is. Columns that are not
  // unique must be referenced using an alias.
  outputMapping_.back()->merge(*right.outputMapping_.back());

  auto inputRowType = node_->outputType()->unionWith(right.node_->outputType());

  ExprPtr expr;
  if (!condition.empty()) {
    auto untypedExpr = parse::parseExpr(condition, parseOptions_);
    expr = resolveScalarTypesImpl(
        untypedExpr, [&](const auto& alias, const auto& name) {
          return resolveJoinInputName(
              alias, name, outputMapping_, inputRowType);
        });
  }

  node_ =
      std::make_shared<JoinNode>(nextId(), node_, right.node_, joinType, expr);

  return *this;
}

PlanBuilder& PlanBuilder::unionAll(const PlanBuilder& other) {
  VELOX_USER_CHECK_NOT_NULL(node_, "UnionAll node cannot be a leaf node");
  VELOX_USER_CHECK_NOT_NULL(other.node_);

  node_ = std::make_shared<SetNode>(
      nextId(),
      std::vector<LogicalPlanNodePtr>{node_, other.node_},
      SetOperation::kUnionAll);

  return *this;
}

ExprPtr PlanBuilder::buildSubquery(
    LogicalPlanNodePtr subquery,
    std::optional<SpecialForm> form,
    const std::optional<std::string>& input) {
  std::vector<ExprPtr> resolvedInputs;
  if (input.has_value()) {
    VELOX_CHECK(form.has_value() && form.value() == SpecialForm::kIn);
    auto untypedExpr = parse::parseExpr(input.value(), parseOptions_);
    resolvedInputs.push_back(resolveScalarTypes(untypedExpr));
  }

  SubqueryType subqueryType;
  if (!form.has_value()) {
    subqueryType = SubqueryType::SCALAR;
    return std::make_shared<SubqueryExpr>(subquery, subqueryType);
  }

  if (form.value() == SpecialForm::kExists) {
    subqueryType = SubqueryType::EXISTS;
  } else if (form.value() == SpecialForm::kIn) {
    subqueryType = SubqueryType::IN;
  } else {
    VELOX_USER_FAIL("Unsupported subquery for special form: {}", form.value());
  }

  resolvedInputs.push_back(
      std::make_shared<SubqueryExpr>(subquery, subqueryType));
  return std::make_shared<SpecialFormExpr>(
      BOOLEAN(), form.value(), resolvedInputs);
}

PlanBuilder& PlanBuilder::withSubquery(
    const std::vector<std::string>& output,
    const std::vector<LogicalPlanNodePtr>& subquery) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Project node cannot be a leaf node");

  std::vector<std::string> outputNames;
  outputNames.reserve(subquery.size());

  std::vector<ExprPtr> exprs;
  exprs.reserve(subquery.size());

  auto newOutputMapping = std::make_shared<NameMappings>();

  const auto& inputType = node_->outputType();

  for (auto i = 0; i < inputType->size(); i++) {
    const auto& id = inputType->nameOf(i);

    outputNames.push_back(id);

    auto names = reverseLookup(id);
    for (const auto& name : names) {
      newOutputMapping->add(name, id, inputType->childAt(i));
    }

    exprs.push_back(
        std::make_shared<InputReferenceExpr>(inputType->childAt(i), id));
  }

  for (int i = 0; i < subquery.size(); ++i) {
    const auto& name = output[i];
    auto id = newName(name);
    outputNames.push_back(id);
    // user is responsible for making sure the name is unique
    newOutputMapping->add(name, id, BOOLEAN());
    exprs.push_back(
        std::make_shared<SubqueryExpr>(subquery[i], SubqueryType::EXISTS));
  }

  node_ = std::make_shared<ProjectNode>(nextId(), node_, outputNames, exprs);
  outputMapping_.back() = newOutputMapping;

  return *this;
}

PlanBuilder& PlanBuilder::withExistsSubquery(
    const std::string& output,
    const LogicalPlanNodePtr subquery) {
  std::vector<std::string> outputs = {output};
  std::vector<LogicalPlanNodePtr> subqueries = {subquery};
  return withExistsSubquery(outputs, subqueries);
}

PlanBuilder& PlanBuilder::withExistsSubquery(
    const std::vector<std::string>& output,
    const std::vector<LogicalPlanNodePtr>& subquery) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Project node cannot be a leaf node");

  std::vector<std::string> outputNames;
  outputNames.reserve(subquery.size());

  std::vector<ExprPtr> exprs;
  exprs.reserve(subquery.size());

  auto newOutputMapping = std::make_shared<NameMappings>();

  const auto& inputType = node_->outputType();

  for (auto i = 0; i < inputType->size(); i++) {
    const auto& id = inputType->nameOf(i);

    outputNames.push_back(id);

    auto names = reverseLookup(id);
    for (const auto& name : names) {
      newOutputMapping->add(name, id, inputType->childAt(i));
    }

    exprs.push_back(
        std::make_shared<InputReferenceExpr>(inputType->childAt(i), id));
  }

  for (int i = 0; i < subquery.size(); ++i) {
    const auto& name = output[i];
    auto id = newName(name);
    outputNames.push_back(id);
    // user is responsible for making sure the name is unique
    newOutputMapping->add(name, id, BOOLEAN());
    std::vector<ExprPtr> resolvedInput = {
        std::make_shared<SubqueryExpr>(subquery[i], SubqueryType::EXISTS)};
    exprs.push_back(std::make_shared<SpecialFormExpr>(
        BOOLEAN(), SpecialForm::kExists, resolvedInput));
  }

  node_ = std::make_shared<ProjectNode>(nextId(), node_, outputNames, exprs);
  outputMapping_.back() = newOutputMapping;

  return *this;
}

PlanBuilder& PlanBuilder::withInSubquery(
    const std::string& output,
    const LogicalPlanNodePtr subquery,
    const std::string& leftInput) {
  std::vector<std::string> outputs = {output};
  std::vector<LogicalPlanNodePtr> subqueries = {subquery};
  std::vector<std::string> leftInputs = {leftInput};
  return withInSubquery(outputs, subqueries, leftInputs);
}

PlanBuilder& PlanBuilder::withInSubquery(
    const std::vector<std::string>& output,
    const std::vector<LogicalPlanNodePtr>& subquery,
    const std::vector<std::string>& leftInput) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Project node cannot be a leaf node");

  std::vector<std::string> outputNames;
  outputNames.reserve(subquery.size());

  std::vector<ExprPtr> exprs;
  exprs.reserve(subquery.size());

  auto newOutputMapping = std::make_shared<NameMappings>();

  const auto& inputType = node_->outputType();

  for (auto i = 0; i < inputType->size(); i++) {
    const auto& id = inputType->nameOf(i);

    outputNames.push_back(id);

    auto names = reverseLookup(id);
    for (const auto& name : names) {
      newOutputMapping->add(name, id, inputType->childAt(i));
    }

    exprs.push_back(
        std::make_shared<InputReferenceExpr>(inputType->childAt(i), id));
  }

  for (int i = 0; i < subquery.size(); ++i) {
    const auto& name = output[i];
    auto id = newName(name);
    outputNames.push_back(id);
    // user is responsible for making sure the name is unique
    newOutputMapping->add(name, id, BOOLEAN());
    std::vector<ExprPtr> resolvedInputs;
    auto untypedExpr = parse::parseExpr(leftInput[i], parseOptions_);
    resolvedInputs.push_back(resolveScalarTypes(untypedExpr));
    resolvedInputs.push_back(
        std::make_shared<SubqueryExpr>(subquery[i], SubqueryType::IN));
    exprs.push_back(std::make_shared<SpecialFormExpr>(
        BOOLEAN(), SpecialForm::kIn, resolvedInputs));
  }

  node_ = std::make_shared<ProjectNode>(nextId(), node_, outputNames, exprs);
  outputMapping_.back() = newOutputMapping;

  return *this;
}

PlanBuilder& PlanBuilder::withScalarSubquery(
    const std::string& output,
    const LogicalPlanNodePtr subquery) {
  std::vector<std::string> outputs = {output};
  std::vector<LogicalPlanNodePtr> subqueries = {subquery};
  return withScalarSubquery(outputs, subqueries);
}

PlanBuilder& PlanBuilder::withScalarSubquery(
    const std::vector<std::string>& output,
    const std::vector<LogicalPlanNodePtr>& subquery) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Project node cannot be a leaf node");

  std::vector<std::string> outputNames;
  outputNames.reserve(subquery.size());

  std::vector<ExprPtr> exprs;
  exprs.reserve(subquery.size());

  auto newOutputMapping = std::make_shared<NameMappings>();

  const auto& inputType = node_->outputType();

  for (auto i = 0; i < inputType->size(); i++) {
    const auto& id = inputType->nameOf(i);

    outputNames.push_back(id);

    auto names = reverseLookup(id);
    for (const auto& name : names) {
      newOutputMapping->add(name, id, inputType->childAt(i));
    }

    exprs.push_back(
        std::make_shared<InputReferenceExpr>(inputType->childAt(i), id));
  }

  for (int i = 0; i < subquery.size(); ++i) {
    const auto& name = output[i];
    auto id = newName(name);
    outputNames.push_back(id);
    // user is responsible for making sure the name is unique
    newOutputMapping->add(name, id, subquery[i]->outputType()->childAt(0));
    std::vector<ExprPtr> resolvedInput = {
        std::make_shared<SubqueryExpr>(subquery[i], SubqueryType::SCALAR)};
    exprs.push_back(
        std::make_shared<SubqueryExpr>(subquery[i], SubqueryType::SCALAR));
  }

  node_ = std::make_shared<ProjectNode>(nextId(), node_, outputNames, exprs);
  outputMapping_.back() = newOutputMapping;

  return *this;
}

PlanBuilder& PlanBuilder::with(
    const std::unordered_map<ExprPtr, std::string>& subqueryOutput) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Project node cannot be a leaf node");

  std::vector<std::string> outputNames;
  outputNames.reserve(subqueryOutput.size());

  std::vector<ExprPtr> exprs;
  exprs.reserve(subqueryOutput.size());

  auto newOutputMapping = std::make_shared<NameMappings>();

  const auto& inputType = node_->outputType();

  for (auto i = 0; i < inputType->size(); i++) {
    const auto& id = inputType->nameOf(i);

    outputNames.push_back(id);

    auto names = reverseLookup(id);
    for (const auto& name : names) {
      newOutputMapping->add(name, id, inputType->childAt(i));
    }

    exprs.push_back(
        std::make_shared<InputReferenceExpr>(inputType->childAt(i), id));
  }

  for (const auto& [expr, name] : subqueryOutput) {
    auto id = newName(name);
    outputNames.push_back(id);
    // user is responsible for making sure the name is unique
    newOutputMapping->add(name, id, expr->type());
    exprs.push_back(expr);
  }

  node_ = std::make_shared<ProjectNode>(nextId(), node_, outputNames, exprs);
  outputMapping_.back() = newOutputMapping;

  return *this;
}

PlanBuilder& PlanBuilder::sort(const std::vector<std::string>& sortingKeys) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Sort node cannot be a leaf node");

  std::vector<SortingField> sortingFields;
  sortingFields.reserve(sortingKeys.size());

  for (const auto& key : sortingKeys) {
    auto orderBy = parse::parseOrderByExpr(key);
    auto expr = resolveScalarTypes(orderBy.expr);

    sortingFields.push_back(
        SortingField{expr, SortOrder(orderBy.ascending, orderBy.nullsFirst)});
  }

  node_ = std::make_shared<SortNode>(nextId(), node_, sortingFields);

  return *this;
}

PlanBuilder& PlanBuilder::limit(int32_t offset, int32_t count) {
  VELOX_USER_CHECK_NOT_NULL(node_, "Limit node cannot be a leaf node");

  node_ = std::make_shared<LimitNode>(nextId(), node_, offset, count);

  return *this;
}

ExprPtr PlanBuilder::resolveInputName(
    const std::optional<std::string>& alias,
    const std::string& name) const {
  if (alias.has_value()) {
    for (int i = outputMapping_.size() - 1; i >= 0; --i) {
      if (auto id = outputMapping_.at(i)->lookup(alias.value(), name)) {
        return std::make_shared<InputReferenceExpr>(
            id.value().type, id.value().id);
      }
    }

    return nullptr;
  }

  for (int i = outputMapping_.size() - 1; i >= 0; --i) {
    if (auto id = outputMapping_.at(i)->lookup(name)) {
      return std::make_shared<InputReferenceExpr>(
          id.value().type, id.value().id);
    }
  }

  VELOX_USER_FAIL(
      "Cannot resolve column: {} not in [{}]",
      NameMappings::QualifiedName{alias, name}.toString(),
      nameMappingListToString(outputMapping_));
}

ExprPtr PlanBuilder::resolveScalarTypes(const core::ExprPtr& expr) const {
  return resolveScalarTypesImpl(expr, [&](const auto& alias, const auto& name) {
    return resolveInputName(alias, name);
  });
}

AggregateExprPtr PlanBuilder::resolveAggregateTypes(
    const core::ExprPtr& expr) const {
  return resolveAggregateTypesImpl(
      expr, [&](const auto& alias, const auto& name) {
        return resolveInputName(alias, name);
      });
}

PlanBuilder& PlanBuilder::as(const std::string& alias) {
  outputMapping_.back()->setAlias(alias);
  return *this;
}

std::string PlanBuilder::newName(const std::string& hint) {
  return nameAllocator_->newName(hint);
}

std::vector<NameMappings::QualifiedName> PlanBuilder::reverseLookup(
    const std::string& id) {
  for (int i = outputMapping_.size() - 1; i >= 0; --i) {
    auto names = outputMapping_[i]->reverseLookup(id);
    if (!names.empty()) {
      return names;
    }
  }
  return {};
}

LogicalPlanNodePtr PlanBuilder::build() {
  VELOX_USER_CHECK_NOT_NULL(node_);

  // Use user-specified names for the output. Should we add an OutputNode?

  const auto names = outputMapping_.back()->uniqueNames();

  bool needRename = false;

  const auto& rowType = node_->outputType();

  std::vector<std::string> outputNames;
  outputNames.reserve(rowType->size());

  std::vector<ExprPtr> exprs;
  exprs.reserve(rowType->size());

  for (auto i = 0; i < rowType->size(); i++) {
    const auto& id = rowType->nameOf(i);

    auto it = names.find(id);
    if (it != names.end()) {
      outputNames.push_back(it->second);
    } else {
      outputNames.push_back(id);
    }

    if (id != outputNames.back()) {
      needRename = true;
    }

    exprs.push_back(
        std::make_shared<InputReferenceExpr>(rowType->childAt(i), id));
  }

  if (needRename) {
    return std::make_shared<ProjectNode>(nextId(), node_, outputNames, exprs);
  }

  return node_;
}

} // namespace facebook::velox::logical_plan
