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
#pragma once

#include "PrestoSqlBaseVisitor.h"
#include "AstNodesAll.h"
#include <functional>

namespace facebook::velox::sql {

struct ParsingOptions {
  bool allowIncompleteStatements = false;
  std::function<void(const std::string&)> warningConsumer;
};

class AstBuilder : public PrestoSqlBaseVisitor {
public:
  explicit AstBuilder(const ParsingOptions& options = ParsingOptions{})
      : parsingOptions_(options), parameterPosition_(0) {}

  // Override visitor methods from PrestoSqlBaseVisitor
  antlrcpp::Any visitSingleStatement(
      PrestoSqlParser::SingleStatementContext* ctx) override;

  antlrcpp::Any visitStandaloneExpression(
      PrestoSqlParser::StandaloneExpressionContext* ctx) override;

  // Query visitors
  antlrcpp::Any visitQuery(
      PrestoSqlParser::QueryContext* ctx) override;

  antlrcpp::Any visitQueryNoWith(
      PrestoSqlParser::QueryNoWithContext* ctx) override;

  antlrcpp::Any visitQueryTermDefault(
      PrestoSqlParser::QueryTermDefaultContext* ctx) override;

  antlrcpp::Any visitQueryPrimaryDefault(
      PrestoSqlParser::QueryPrimaryDefaultContext* ctx) override;

  antlrcpp::Any visitQuerySpecification(
      PrestoSqlParser::QuerySpecificationContext* ctx) override;

  antlrcpp::Any visitSelectSingle(
      PrestoSqlParser::SelectSingleContext* ctx) override;

  antlrcpp::Any visitSelectAll(
      PrestoSqlParser::SelectAllContext* ctx) override;

  antlrcpp::Any visitTable(
      PrestoSqlParser::TableContext* ctx) override;

  antlrcpp::Any visitTableName(
      PrestoSqlParser::TableNameContext* ctx) override;

  // Expression visitors
  antlrcpp::Any visitLogicalBinary(
      PrestoSqlParser::LogicalBinaryContext* ctx) override;

  antlrcpp::Any visitComparison(
      PrestoSqlParser::ComparisonContext* ctx) override;

  antlrcpp::Any visitArithmeticBinary(
      PrestoSqlParser::ArithmeticBinaryContext* ctx) override;

  antlrcpp::Any visitArithmeticUnary(
      PrestoSqlParser::ArithmeticUnaryContext* ctx) override;

  antlrcpp::Any visitFunctionCall(
      PrestoSqlParser::FunctionCallContext* ctx) override;

  // Literal visitors
  antlrcpp::Any visitBooleanLiteral(
      PrestoSqlParser::BooleanLiteralContext* ctx) override;

  antlrcpp::Any visitStringLiteral(
      PrestoSqlParser::StringLiteralContext* ctx) override;

  antlrcpp::Any visitIntegerLiteral(
      PrestoSqlParser::IntegerLiteralContext* ctx) override;

  antlrcpp::Any visitDecimalLiteral(
      PrestoSqlParser::DecimalLiteralContext* ctx) override;

  antlrcpp::Any visitNullLiteral(
      PrestoSqlParser::NullLiteralContext* ctx) override;

  // Identifier visitors
  antlrcpp::Any visitUnquotedIdentifier(
      PrestoSqlParser::UnquotedIdentifierContext* ctx) override;

  antlrcpp::Any visitQuotedIdentifier(
      PrestoSqlParser::QuotedIdentifierContext* ctx) override;

  antlrcpp::Any visitQualifiedName(
      PrestoSqlParser::QualifiedNameContext* ctx) override;

private:
  // Helper methods
  std::string getIdentifier(const std::string& text);
  std::string unquote(const std::string& value, char quote = '\'');
  
  // Type conversion helpers
  ArithmeticBinaryExpression::Operator getArithmeticBinaryOperator(int tokenType);
  ComparisonExpression::Operator getComparisonOperator(int tokenType);
  LogicalBinaryExpression::Operator getLogicalBinaryOperator(int tokenType);

  // Template helper to extract nodes from Any
  template<typename T>
  std::shared_ptr<T> getNode(const antlrcpp::Any& any) {
    try {
      return std::any_cast<std::shared_ptr<T>>(any);
    } catch (const std::bad_any_cast& e) {
      // Try to cast to base Node first, then dynamic_cast
      try {
        auto nodePtr = std::any_cast<NodePtr>(any);
        auto castPtr = std::dynamic_pointer_cast<T>(nodePtr);
        if (castPtr) {
          return castPtr;
        }
      } catch (...) {
        // Fall through to throw original error
      }
      throw;
    }
  }

  template<typename T>
  std::shared_ptr<T> visit(antlr4::tree::ParseTree* tree) {
    return getNode<T>(tree->accept(this));
  }

  const ParsingOptions parsingOptions_;
  int parameterPosition_;
};

} // namespace facebook::velox::sql