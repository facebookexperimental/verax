/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "axiom/optimizer/tests/PrestoParser.h"
#include "axiom/logical_plan/PlanBuilder.h"
#include "axiom/sql/presto/ParserHelper.h"
#include "axiom/sql/presto/ast/AstBuilder.h"
#include "axiom/sql/presto/ast/AstPrinter.h"
#include "velox/exec/Aggregate.h"

namespace lp = facebook::velox::logical_plan;
namespace sql = axiom::sql::presto;

namespace facebook::velox::optimizer::test {
namespace {

// Analizes the expression to collect a list of dependent column names and
// number of aggregate function calls.
class ExprAnalyzer : public sql::AstVisitor {
 public:
  const std::unordered_set<std::string>& dependencies() const {
    return names_;
  }

  bool hasAggregate() const {
    return aggregateName_.has_value();
  }

  bool containsAny(const std::unordered_set<std::string>& names) const {
    for (const auto& name : names) {
      if (names_.contains(name)) {
        return true;
      }
    }
    return false;
  }

 private:
  void defaultVisit(sql::Node* node) override {
    if (dynamic_cast<sql::Literal*>(node) != nullptr) {
      // Literals have no dependencies.
      return;
    }

    VELOX_NYI(
        "Not yet supported node type: {}",
        sql::NodeTypeName::toName(node->type()));
  }

  void visitFunctionCall(sql::FunctionCall* node) override {
    const auto& name = node->name()->suffix();
    if (velox::exec::getAggregateFunctionEntry(name)) {
      if (aggregateName_.has_value()) {
      }
      VELOX_USER_CHECK(
          !aggregateName_.has_value(),
          "Cannot nest aggregations inside aggregation: {}({})",
          aggregateName_.value(),
          name);
      aggregateName_ = name;
    }

    for (const auto& arg : node->arguments()) {
      arg->accept(this);
    }
  }

  void visitArithmeticBinaryExpression(
      sql::ArithmeticBinaryExpression* node) override {
    node->left()->accept(this);
    node->right()->accept(this);
  }

  void visitIdentifier(sql::Identifier* node) override {
    names_.insert(node->value());
  }

  std::unordered_set<std::string> names_;
  std::optional<std::string> aggregateName_;
};

class RelationPlanner : public sql::AstVisitor {
 public:
  RelationPlanner(const std::string& defaultConnectorId)
      : context_{defaultConnectorId},
        builder_(std::make_shared<lp::PlanBuilder>(context_)) {}

  lp::LogicalPlanNodePtr getPlan() {
    return builder_->build();
  }

 private:
  static std::string toFunctionName(sql::ComparisonExpression::Operator op) {
    switch (op) {
      case sql::ComparisonExpression::Operator::kEqual:
        return "eq";
      case sql::ComparisonExpression::Operator::kNotEqual:
        return "neq";
      case sql::ComparisonExpression::Operator::kLessThan:
        return "lt";
      case sql::ComparisonExpression::Operator::kLessThanOrEqual:
        return "lte";
      case sql::ComparisonExpression::Operator::kGreaterThan:
        return "gt";
      case sql::ComparisonExpression::Operator::kGreaterThanOrEqual:
        return "gte";
      default:
        VELOX_NYI("Not yet supported comparison operator: {}", op);
    }
  }

  static std::string toFunctionName(
      sql::ArithmeticBinaryExpression::Operator op) {
    switch (op) {
      case sql::ArithmeticBinaryExpression::Operator::kAdd:
        return "plus";
      case sql::ArithmeticBinaryExpression::Operator::kSubtract:
        return "minus";
      case sql::ArithmeticBinaryExpression::Operator::kMultiply:
        return "multiply";
      case sql::ArithmeticBinaryExpression::Operator::kDivide:
        return "divide";
      case sql::ArithmeticBinaryExpression::Operator::kModulus:
        return "modulus";
    }
  }

  lp::ExprApi toExpr(const sql::ExpressionPtr& node) {
    switch (node->type()) {
      case sql::NodeType::kIdentifier:
        return lp::Col(node->as<sql::Identifier>()->value());

      case sql::NodeType::kComparisonExpression: {
        auto* comparison = node->as<sql::ComparisonExpression>();
        return lp::Call(
            toFunctionName(comparison->op()),
            toExpr(comparison->left()),
            toExpr(comparison->right()));
      }

      case sql::NodeType::kArithmeticBinaryExpression: {
        auto* binary = node->as<sql::ArithmeticBinaryExpression>();
        return lp::Call(
            toFunctionName(binary->op()),
            toExpr(binary->left()),
            toExpr(binary->right()));
      }

      case sql::NodeType::kLongLiteral:
        return lp::Lit(node->as<sql::LongLiteral>()->value());

      case sql::NodeType::kFunctionCall: {
        auto* call = node->as<sql::FunctionCall>();

        std::vector<lp::ExprApi> args;
        for (const auto& arg : call->arguments()) {
          args.push_back(toExpr(arg));
        }
        return lp::Call(call->name()->suffix(), args);
      }

      default:
        VELOX_NYI(
            "Unsupported expression type: {}",
            sql::NodeTypeName::toName(node->type()));
    }
  }

  void addFilter(const sql::ExpressionPtr& filter) {
    if (filter != nullptr) {
      builder_->filter(toExpr(filter));
    }
  }

  static lp::JoinType toJoinType(sql::Join::Type type) {
    switch (type) {
      case sql::Join::Type::kCross:
        return lp::JoinType::kInner;
      case sql::Join::Type::kImplicit:
        return lp::JoinType::kInner;
      case sql::Join::Type::kInner:
        return lp::JoinType::kInner;
      case sql::Join::Type::kLeft:
        return lp::JoinType::kLeft;
      case sql::Join::Type::kRight:
        return lp::JoinType::kRight;
      case sql::Join::Type::kFull:
        return lp::JoinType::kFull;
    }
  }

  void processFrom(const sql::RelationPtr& relation) {
    if (relation->is(sql::NodeType::kTable)) {
      auto* table = relation->as<sql::Table>();
      builder_->tableScan(table->name()->suffix());
      return;
    }

    if (relation->is(sql::NodeType::kJoin)) {
      auto* join = relation->as<sql::Join>();
      processFrom(join->left());

      auto leftBuilder = builder_;

      builder_ = std::make_shared<lp::PlanBuilder>(context_);
      processFrom(join->right());
      auto rightBuilder = builder_;

      builder_ = leftBuilder;

      std::optional<lp::ExprApi> condition;

      if (const auto& criteria = join->criteria()) {
        if (criteria->is(sql::NodeType::kJoinOn)) {
          condition = toExpr(criteria->as<sql::JoinOn>()->expression());
        } else {
          VELOX_NYI(
              "Join criteria type is not supported yet: {}",
              sql::NodeTypeName::toName(criteria->type()));
        }
      }

      builder_->join(*rightBuilder, condition, toJoinType(join->joinType()));
      return;
    }

    VELOX_NYI(
        "Relation type is not supported yet: {}",
        sql::NodeTypeName::toName(relation->type()));
  }

  // Returns true if 'selectItems' contains a single SELECT *.
  static bool isSelectAll(const std::vector<sql::SelectItemPtr>& selectItems) {
    if (selectItems.size() == 1 &&
        selectItems.at(0)->is(sql::NodeType::kAllColumns)) {
      return true;
    }

    return false;
  }

  void addProject(const std::vector<sql::SelectItemPtr>& selectItems) {
    std::vector<lp::ExprApi> exprs;
    for (const auto& item : selectItems) {
      VELOX_CHECK(item->is(sql::NodeType::kSingleColumn));
      auto* singleColumn = item->as<sql::SingleColumn>();

      lp::ExprApi expr = toExpr(singleColumn->expression());

      if (singleColumn->alias() != nullptr) {
        expr = expr.as(singleColumn->alias()->value());
      }
      exprs.push_back(expr);
    }

    builder_->project(exprs);
  }

  lp::ExprApi toSortingKey(const sql::ExpressionPtr& expr) {
    if (expr->is(sql::NodeType::kLongLiteral)) {
      const auto n = expr->as<sql::LongLiteral>()->value();
      const auto name = builder_->findOrAssignOutputNameAt(n - 1);

      return lp::Col(name);
    }

    return toExpr(expr);
  }

  lp::ExprApi toGroupingKey(const sql::ExpressionPtr& expr) {
    if (expr->is(sql::NodeType::kLongLiteral)) {
      const auto n = expr->as<sql::LongLiteral>()->value();
      const auto name = builder_->findOrAssignOutputNameAt(n);

      return lp::Col(name);
    }

    VELOX_CHECK_EQ(
        expr->type(),
        sql::NodeType::kIdentifier,
        "Grouping key must be a simple column reference. "
        "Expressions are not supported yet.");

    return toExpr(expr);
  }

  bool tryAddGlobalAgg(const std::vector<sql::SelectItemPtr>& selectItems) {
    bool hasAggregate = false;
    for (const auto& item : selectItems) {
      VELOX_CHECK(item->is(sql::NodeType::kSingleColumn));
      auto* singleColumn = item->as<sql::SingleColumn>();

      ExprAnalyzer exprAnalyzer;
      singleColumn->expression()->accept(&exprAnalyzer);

      if (exprAnalyzer.hasAggregate()) {
        hasAggregate = true;
        break;
      }
    }

    if (!hasAggregate) {
      return false;
    }

    // Verify that all expressions are aggregates or constant.
    std::vector<lp::ExprApi> aggregates;
    for (const auto& item : selectItems) {
      VELOX_CHECK(item->is(sql::NodeType::kSingleColumn));
      auto* singleColumn = item->as<sql::SingleColumn>();

      ExprAnalyzer exprAnalyzer;
      singleColumn->expression()->accept(&exprAnalyzer);

      lp::ExprApi expr = toExpr(singleColumn->expression());

      // TODO Allow constant expressions: SELECT 1 + 2, count(1) FROM t

      VELOX_USER_CHECK(
          exprAnalyzer.hasAggregate(),
          "[{}] must be an aggregate expression or appear in GROUP BY clause",
          expr.expr()->toString());

      if (singleColumn->alias() != nullptr) {
        expr = expr.as(singleColumn->alias()->value());
      }

      aggregates.push_back(expr);
    }

    builder_->aggregate({}, aggregates);
    return true;
  }

  void addGroupBy(
      const std::vector<sql::SelectItemPtr>& selectItems,
      const sql::GroupByPtr& groupBy) {
    VELOX_USER_CHECK(
        !groupBy->isDistinct(), "GROUP BY with DISTINCT is not supported yet");

    const auto& groupingElements = groupBy->groupingElements();

    // Collect grouping keys that refer to output columns. Add projection if
    // needed.
    std::vector<lp::ExprApi> groupingKeys;
    std::vector<int32_t> groupingKeyExprIndices;
    std::vector<int32_t> groupingKeyIndices(selectItems.size(), -1);
    {
      std::vector<lp::ExprApi> groupingKeyProjections;
      for (const auto& element : groupingElements) {
        VELOX_CHECK_EQ(element->type(), sql::NodeType::kSimpleGroupBy);

        const auto* simple = element->as<sql::SimpleGroupBy>();

        for (const auto& expr : simple->expressions()) {
          if (expr->is(sql::NodeType::kLongLiteral)) {
            // 1-based index.
            const auto n = expr->as<sql::LongLiteral>()->value();

            VELOX_CHECK_GE(n, 1);
            VELOX_CHECK_LE(n, selectItems.size());

            const auto& item = selectItems.at(n - 1);
            VELOX_CHECK(item->is(sql::NodeType::kSingleColumn));
            auto* singleColumn = item->as<sql::SingleColumn>();

            if (!singleColumn->expression()->is(sql::NodeType::kIdentifier)) {
              groupingKeyProjections.push_back(
                  toExpr(singleColumn->expression()));

              groupingKeyIndices[n - 1] = groupingKeys.size();

              groupingKeyExprIndices.push_back(groupingKeys.size());
              groupingKeys.push_back(lp::Lit(1)); // placeholder
            } else {
              groupingKeys.push_back(lp::Col(
                  singleColumn->expression()->as<sql::Identifier>()->value()));
            }
          } else {
            groupingKeys.push_back(toGroupingKey(expr));
          }
        }
      }

      if (!groupingKeyProjections.empty()) {
        const auto numOutput = builder_->numOutput();

        builder_->with(groupingKeyProjections);

        for (auto i = 0; i < groupingKeyExprIndices.size(); ++i) {
          groupingKeys[groupingKeyExprIndices[i]] =
              lp::Col(builder_->findOrAssignOutputNameAt(numOutput + i));
        }
      }
    }

    const auto numGroupingKeys = groupingKeys.size();

    // The following cases require a Project after Aggregation:
    // - SELECT contains expressions over grouping keys;
    // - The number or order of columns in SELECT doesn't match the output of
    // Aggregation.
    // - SELECT contains expressions over aggregate function calls.
    bool needsProjection = false;
    std::vector<lp::ExprApi> projections;

    // Grouping keys must be column references. Expressions are not allowed yet.
    std::unordered_set<std::string> keyNames;
    for (const auto& key : groupingKeys) {
      VELOX_CHECK(key.name().has_value());
      keyNames.insert(key.name().value());
    }

    std::vector<lp::ExprApi> aggregates;
    std::vector<int32_t> aggregateIndices(selectItems.size(), -1);
    for (auto i = 0; i < selectItems.size(); ++i) {
      const auto& item = selectItems.at(i);

      VELOX_CHECK(item->is(sql::NodeType::kSingleColumn));
      auto* singleColumn = item->as<sql::SingleColumn>();

      if (groupingKeyIndices[i] != -1) {
        projections.push_back(groupingKeys[groupingKeyIndices[i]]);
        continue;
      }

      ExprAnalyzer exprAnalyzer;
      singleColumn->expression()->accept(&exprAnalyzer);

      lp::ExprApi expr = toExpr(singleColumn->expression());
      if (singleColumn->alias() != nullptr) {
        expr = expr.as(singleColumn->alias()->value());
      }

      if (exprAnalyzer.hasAggregate()) {
        aggregateIndices[i] = aggregates.size();
        projections.push_back(lp::Lit(1)); // placeholder

        aggregates.push_back(expr);
      } else {
        if (!singleColumn->expression()->is(sql::NodeType::kIdentifier)) {
          needsProjection = true;
        } else if (
            i >= numGroupingKeys || expr.name() != groupingKeys.at(i).name()) {
          needsProjection = true;
        }

        projections.push_back(expr);
      }
    }

    builder_->aggregate(groupingKeys, aggregates);

    if (selectItems.size() != numGroupingKeys + aggregates.size()) {
      needsProjection = true;
    }

    if (needsProjection) {
      for (auto i = 0; i < aggregateIndices.size(); ++i) {
        const auto index = aggregateIndices[i];
        if (index != -1) {
          const auto name =
              builder_->findOrAssignOutputNameAt(index + numGroupingKeys);
          projections[i] = lp::Col(name);
        }
      }

      builder_->project(projections);
    }
  }

  void addOrderBy(const sql::OrderByPtr& orderBy) {
    if (orderBy == nullptr) {
      return;
    }

    std::vector<lp::SortKey> keys;

    const auto& sortItems = orderBy->sortItems();
    for (const auto& item : sortItems) {
      auto expr = toSortingKey(item->sortKey());
      keys.push_back(
          lp::SortKey(expr, item->isAscending(), item->isNullsFirst()));
    }

    builder_->sort(keys);
  }

  static int64_t parseInt64(const std::optional<std::string>& value) {
    return std::atol(value.value().c_str());
  }

  void addOffset(const sql::OffsetPtr& offset) {
    if (offset == nullptr) {
      return;
    }

    builder_->offset(std::atol(offset->offset().c_str()));
  }

  void addLimit(const std::optional<std::string>& limit) {
    if (!limit.has_value()) {
      return;
    }

    builder_->limit(parseInt64(limit));
  }

  void visitQuery(sql::Query* query) override {
    const auto& queryBody = query->queryBody();

    VELOX_CHECK_NOT_NULL(queryBody);
    VELOX_CHECK(queryBody->is(sql::NodeType::kQuerySpecification));
    auto* querySpec = queryBody->as<sql::QuerySpecification>();

    // FROM t -> builder.tableScan(t)

    {
      const auto& from = querySpec->from();
      VELOX_CHECK_NOT_NULL(from);
      processFrom(from);
    }

    // WHERE a > 1 -> builder.filter("a > 1")
    addFilter(querySpec->where());

    const auto& selectItems = querySpec->select()->selectItems();

    if (querySpec->groupBy()) {
      addGroupBy(selectItems, querySpec->groupBy());
      addFilter(querySpec->having());
    } else {
      if (isSelectAll(selectItems)) {
        // SELECT *. No project needed.
      } else if (tryAddGlobalAgg(selectItems)) {
        // Nothing else to do.
      } else {
        // SELECT a, b -> builder.project({a, b})
        addProject(selectItems);
      }
    }

    if (querySpec->select()->isDistinct()) {
      builder_->aggregate(builder_->findOrAssignOutputNames(), {});
    }

    addOrderBy(query->orderBy());
    addOffset(query->offset());
    addLimit(query->limit());
  }

  void visitQuerySpecification(sql::QuerySpecification* node) override {}

  lp::PlanBuilder::Context context_;
  std::shared_ptr<lp::PlanBuilder> builder_;
};

} // namespace

SqlStatementPtr PrestoParser::parse(
    const std::string& sql,
    bool enableTracing) {
  sql::ParserHelper helper(sql);
  auto* queryContext = helper.parser().query();

  VELOX_USER_CHECK_EQ(
      0,
      helper.parser().getNumberOfSyntaxErrors(),
      "Failed to parse SQL: {}",
      sql);

  sql::AstBuilder astBuilder(enableTracing);
  auto query = astBuilder.visit(queryContext).as<std::shared_ptr<sql::Query>>();

  if (enableTracing) {
    std::stringstream astString;
    sql::AstPrinter printer(astString);
    query->accept(&printer);

    std::cout << "AST: " << astString.str() << std::endl;
  }

  RelationPlanner planner(defaultConnectorId_);
  query->accept(&planner);

  return std::make_shared<SelectStatement>(planner.getPlan());
}

} // namespace facebook::velox::optimizer::test
