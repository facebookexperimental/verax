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

#include "AstBuilder.h"
#include "PrestoSqlParser.h"
#include <stdexcept>
#include <algorithm>

namespace facebook::velox::sql {

antlrcpp::Any AstBuilder::visitSingleStatement(
    PrestoSqlParser::SingleStatementContext* ctx) {
  return ctx->statement()->accept(this);
}

antlrcpp::Any AstBuilder::visitStandaloneExpression(
    PrestoSqlParser::StandaloneExpressionContext* ctx) {
  return ctx->expression()->accept(this);
}

antlrcpp::Any AstBuilder::visitQuery(
    PrestoSqlParser::QueryContext* ctx) {
  auto queryBody = visit<QueryBody>(ctx->queryNoWith());
  
  // Create Query with proper parameters
  std::optional<std::shared_ptr<With>> with = std::nullopt;
  std::optional<std::shared_ptr<OrderBy>> orderBy = std::nullopt;
  std::optional<std::shared_ptr<Offset>> offset = std::nullopt;
  std::optional<std::string> limit = std::nullopt;
  
  return std::make_shared<Query>(with, queryBody, orderBy, offset, limit);
}

antlrcpp::Any AstBuilder::visitQueryNoWith(
    PrestoSqlParser::QueryNoWithContext* ctx) {
  return ctx->queryTerm()->accept(this);
}

antlrcpp::Any AstBuilder::visitQueryTermDefault(
    PrestoSqlParser::QueryTermDefaultContext* ctx) {
  return ctx->queryPrimary()->accept(this);
}

antlrcpp::Any AstBuilder::visitQueryPrimaryDefault(
    PrestoSqlParser::QueryPrimaryDefaultContext* ctx) {
  return ctx->querySpecification()->accept(this);
}

antlrcpp::Any AstBuilder::visitQuerySpecification(
    PrestoSqlParser::QuerySpecificationContext* ctx) {
  // Build SELECT items from selectItem contexts
  std::vector<SelectItemPtr> selectItems;
  for (auto item : ctx->selectItem()) {
    try {
      auto selectItem = visit<SelectItem>(item);
      selectItems.push_back(selectItem);
    } catch (const std::bad_any_cast& e) {
      // Debug: print the actual type being returned
      auto anyResult = item->accept(this);
      throw std::runtime_error("Failed to cast selectItem - check visitor method return type");
    }
  }
  
  bool distinct = ctx->DISTINCT() != nullptr;
  auto select = std::make_shared<Select>(distinct, selectItems);
  
  std::optional<RelationPtr> from = std::nullopt;
  if (ctx->relation().size() > 0) {
    from = visit<Relation>(ctx->relation(0));
  }
  
  std::optional<ExpressionPtr> where = std::nullopt;
  if (ctx->WHERE()) {
    where = visit<Expression>(ctx->booleanExpression());
  }
  
  return std::make_shared<QuerySpecification>(select, from, where);
}

antlrcpp::Any AstBuilder::visitSelectSingle(
    PrestoSqlParser::SelectSingleContext* ctx) {
  auto expression = visit<Expression>(ctx->expression());
  
  std::optional<std::shared_ptr<Identifier>> alias = std::nullopt;
  if (ctx->identifier()) {
    alias = std::make_shared<Identifier>(ctx->identifier()->getText());
  }
  
  return std::make_shared<SingleColumn>(expression, alias);
}

antlrcpp::Any AstBuilder::visitSelectAll(
    PrestoSqlParser::SelectAllContext* ctx) {
  // For now, implement as a simple identifier "*"
  std::optional<std::shared_ptr<QualifiedName>> prefix = std::nullopt;
  return std::make_shared<AllColumns>(prefix);
}

antlrcpp::Any AstBuilder::visitTable(
    PrestoSqlParser::TableContext* ctx) {
  return ctx->tableName()->accept(this);
}

antlrcpp::Any AstBuilder::visitTableName(
    PrestoSqlParser::TableNameContext* ctx) {
  auto qualifiedName = visit<QualifiedName>(ctx->qualifiedName());
  return std::make_shared<Table>(qualifiedName);
}

antlrcpp::Any AstBuilder::visitLogicalBinary(
    PrestoSqlParser::LogicalBinaryContext* ctx) {
  auto left = visit<Expression>(ctx->left);
  auto right = visit<Expression>(ctx->right);
  auto op = getLogicalBinaryOperator(ctx->operator_->getType());
  
  return std::make_shared<LogicalBinaryExpression>(op, left, right);
}

antlrcpp::Any AstBuilder::visitComparison(
    PrestoSqlParser::ComparisonContext* ctx) {
  auto left = visit<Expression>(ctx->left);
  auto right = visit<Expression>(ctx->right);
  auto op = getComparisonOperator(ctx->comparisonOperator()->getStart()->getType());
  
  return std::make_shared<ComparisonExpression>(op, left, right);
}

antlrcpp::Any AstBuilder::visitArithmeticBinary(
    PrestoSqlParser::ArithmeticBinaryContext* ctx) {
  auto left = visit<Expression>(ctx->left);
  auto right = visit<Expression>(ctx->right);
  auto op = getArithmeticBinaryOperator(ctx->operator_->getType());
  
  return std::make_shared<ArithmeticBinaryExpression>(op, left, right);
}

antlrcpp::Any AstBuilder::visitArithmeticUnary(
    PrestoSqlParser::ArithmeticUnaryContext* ctx) {
  // For unary expressions, we could extend ArithmeticBinaryExpression or create a separate class
  // For now, treating as a simple case
  auto expression = visit<Expression>(ctx->expression());
  return expression;
}

antlrcpp::Any AstBuilder::visitFunctionCall(
    PrestoSqlParser::FunctionCallContext* ctx) {
  std::string name = ctx->qualifiedName()->getText();
  
  std::vector<ExpressionPtr> arguments;
  if (ctx->expression().size() > 0) {
    for (auto expr : ctx->expression()) {
      arguments.push_back(visit<Expression>(expr));
    }
  }
  
  bool distinct = ctx->DISTINCT() != nullptr;
  
  return std::make_shared<FunctionCall>(name, arguments, distinct);
}

antlrcpp::Any AstBuilder::visitBooleanLiteral(
    PrestoSqlParser::BooleanLiteralContext* ctx) {
  bool value = ctx->TRUE() != nullptr;
  return std::make_shared<BooleanLiteral>(value);
}

antlrcpp::Any AstBuilder::visitStringLiteral(
    PrestoSqlParser::StringLiteralContext* ctx) {
  std::string value = unquote(ctx->getText());
  return std::make_shared<StringLiteral>(value);
}

antlrcpp::Any AstBuilder::visitIntegerLiteral(
    PrestoSqlParser::IntegerLiteralContext* ctx) {
  int64_t value = std::stoll(ctx->getText());
  return std::make_shared<LongLiteral>(value);
}

antlrcpp::Any AstBuilder::visitDecimalLiteral(
    PrestoSqlParser::DecimalLiteralContext* ctx) {
  double value = std::stod(ctx->getText());
  return std::make_shared<DoubleLiteral>(value);
}

antlrcpp::Any AstBuilder::visitNullLiteral(
    PrestoSqlParser::NullLiteralContext* ctx) {
  return std::make_shared<NullLiteral>();
}

antlrcpp::Any AstBuilder::visitUnquotedIdentifier(
    PrestoSqlParser::UnquotedIdentifierContext* ctx) {
  return std::make_shared<Identifier>(ctx->getText());
}

antlrcpp::Any AstBuilder::visitQuotedIdentifier(
    PrestoSqlParser::QuotedIdentifierContext* ctx) {
  std::string value = unquote(ctx->getText(), '"');
  return std::make_shared<Identifier>(value);
}

antlrcpp::Any AstBuilder::visitQualifiedName(
    PrestoSqlParser::QualifiedNameContext* ctx) {
  std::vector<std::string> parts;
  for (auto identifier : ctx->identifier()) {
    parts.push_back(getIdentifier(identifier->getText()));
  }
  return std::make_shared<QualifiedName>(parts);
}

// Helper methods
std::string AstBuilder::getIdentifier(const std::string& text) {
  return unquote(text, '"');
}

std::string AstBuilder::unquote(const std::string& value, char quote) {
  if (value.length() >= 2 && value.front() == quote && value.back() == quote) {
    return value.substr(1, value.length() - 2);
  }
  return value;
}

ArithmeticBinaryExpression::Operator AstBuilder::getArithmeticBinaryOperator(int tokenType) {
  switch (tokenType) {
    case PrestoSqlParser::PLUS:
      return ArithmeticBinaryExpression::Operator::ADD;
    case PrestoSqlParser::MINUS:
      return ArithmeticBinaryExpression::Operator::SUBTRACT;
    case PrestoSqlParser::ASTERISK:
      return ArithmeticBinaryExpression::Operator::MULTIPLY;
    case PrestoSqlParser::SLASH:
      return ArithmeticBinaryExpression::Operator::DIVIDE;
    case PrestoSqlParser::PERCENT:
      return ArithmeticBinaryExpression::Operator::MODULUS;
    default:
      throw std::invalid_argument("Unsupported arithmetic operator: " + std::to_string(tokenType));
  }
}

ComparisonExpression::Operator AstBuilder::getComparisonOperator(int tokenType) {
  switch (tokenType) {
    case PrestoSqlParser::EQ:
      return ComparisonExpression::Operator::EQUAL;
    case PrestoSqlParser::NEQ:
      return ComparisonExpression::Operator::NOT_EQUAL;
    case PrestoSqlParser::LT:
      return ComparisonExpression::Operator::LESS_THAN;
    case PrestoSqlParser::LTE:
      return ComparisonExpression::Operator::LESS_THAN_OR_EQUAL;
    case PrestoSqlParser::GT:
      return ComparisonExpression::Operator::GREATER_THAN;
    case PrestoSqlParser::GTE:
      return ComparisonExpression::Operator::GREATER_THAN_OR_EQUAL;
    default:
      throw std::invalid_argument("Unsupported comparison operator: " + std::to_string(tokenType));
  }
}

LogicalBinaryExpression::Operator AstBuilder::getLogicalBinaryOperator(int tokenType) {
  switch (tokenType) {
    case PrestoSqlParser::AND:
      return LogicalBinaryExpression::Operator::AND;
    case PrestoSqlParser::OR:
      return LogicalBinaryExpression::Operator::OR;
    default:
      throw std::invalid_argument("Unsupported logical operator: " + std::to_string(tokenType));
  }
}

} // namespace facebook::velox::sql