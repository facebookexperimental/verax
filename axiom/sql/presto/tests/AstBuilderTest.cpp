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

#include "../PrestoSqlAstParser.h"
#include "../AstNodesAll.h"
#include <gtest/gtest.h>

namespace facebook::velox::sql {

class AstBuilderTest : public ::testing::Test {
protected:
  PrestoSqlAstParser parser;
};

TEST_F(AstBuilderTest, ParseSimpleSelect) {
  std::string sql = "SELECT 1";
  auto result = parser.parseStatement(sql);
  
  EXPECT_FALSE(result.hasErrors) << "Parse error: " << result.errorMessage;
  EXPECT_TRUE(result.ast != nullptr);
  
  auto query = result.ast->as<Query>();
  EXPECT_TRUE(query != nullptr);
}

TEST_F(AstBuilderTest, ParseSelectFromTable) {
  std::string sql = "SELECT * FROM users";
  auto result = parser.parseStatement(sql);
  
  EXPECT_FALSE(result.hasErrors) << "Parse error: " << result.errorMessage;
  EXPECT_TRUE(result.ast != nullptr);
  
  auto query = result.ast->as<Query>();
  EXPECT_TRUE(query != nullptr);
  
  auto querySpec = query->getQueryBody()->as<QuerySpecification>();
  EXPECT_TRUE(querySpec != nullptr);
  EXPECT_TRUE(querySpec->getFrom().has_value());
  
  auto table = querySpec->getFrom().value()->as<Table>();
  EXPECT_TRUE(table != nullptr);
  EXPECT_EQ(table->getName(), "users");
}

TEST_F(AstBuilderTest, ParseSelectWithWhere) {
  std::string sql = "SELECT id FROM users WHERE id > 10";
  auto result = parser.parseStatement(sql);
  
  EXPECT_FALSE(result.hasErrors) << "Parse error: " << result.errorMessage;
  EXPECT_TRUE(result.ast != nullptr);
  
  auto query = result.ast->as<Query>();
  EXPECT_TRUE(query != nullptr);
  
  auto querySpec = query->getQueryBody()->as<QuerySpecification>();
  EXPECT_TRUE(querySpec != nullptr);
  EXPECT_TRUE(querySpec->getWhere().has_value());
  
  auto comparison = querySpec->getWhere().value()->as<ComparisonExpression>();
  EXPECT_TRUE(comparison != nullptr);
  EXPECT_EQ(comparison->getOperator(), ComparisonExpression::Operator::GREATER_THAN);
}

TEST_F(AstBuilderTest, ParseArithmeticExpression) {
  std::string sql = "SELECT a + b * 2 FROM table1";
  auto result = parser.parseStatement(sql);
  
  EXPECT_FALSE(result.hasErrors) << "Parse error: " << result.errorMessage;
  EXPECT_TRUE(result.ast != nullptr);
  
  auto query = result.ast->as<Query>();
  EXPECT_TRUE(query != nullptr);
}

TEST_F(AstBuilderTest, ParseFunctionCall) {
  std::string sql = "SELECT COUNT(*) FROM users";
  auto result = parser.parseStatement(sql);
  
  EXPECT_FALSE(result.hasErrors) << "Parse error: " << result.errorMessage;
  EXPECT_TRUE(result.ast != nullptr);
  
  auto query = result.ast->as<Query>();
  EXPECT_TRUE(query != nullptr);
}

TEST_F(AstBuilderTest, ParseLiterals) {
  std::string sql = "SELECT 42, 'hello', true, 3.14, null";
  auto result = parser.parseStatement(sql);
  
  EXPECT_FALSE(result.hasErrors) << "Parse error: " << result.errorMessage;
  EXPECT_TRUE(result.ast != nullptr);
}

TEST_F(AstBuilderTest, ParseExpressionStandalone) {
  std::string expr = "1 + 2 * 3";
  auto result = parser.parseExpression(expr);
  
  EXPECT_FALSE(result.hasErrors) << "Parse error: " << result.errorMessage;
  EXPECT_TRUE(result.ast != nullptr);
  
  auto arithmeticExpr = result.ast->as<ArithmeticBinaryExpression>();
  EXPECT_TRUE(arithmeticExpr != nullptr);
}

TEST_F(AstBuilderTest, HandleInvalidSql) {
  std::string sql = "SELECT FROM"; // Invalid SQL
  auto result = parser.parseStatement(sql);
  
  EXPECT_TRUE(result.hasErrors);
  EXPECT_FALSE(result.errorMessage.empty());
}

} // namespace facebook::velox::sql