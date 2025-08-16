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

#include "AstNodesAll.h"
#include "PrestoSqlAstParser.h"
#include <iostream>

namespace facebook::velox::sql {

class ComprehensiveAstPrinter : public AstVisitor {
public:
  void visitQuery(Query* node) override {
    printIndent();
    std::cout << "Query:\n";
    indent_++;
    
    if (node->getWith().has_value()) {
      node->getWith().value()->accept(this);
    }
    
    node->getQueryBody()->accept(this);
    
    if (node->getOrderBy().has_value()) {
      node->getOrderBy().value()->accept(this);
    }
    
    if (node->getOffset().has_value()) {
      node->getOffset().value()->accept(this);
    }
    
    if (node->getLimit().has_value()) {
      printIndent();
      std::cout << "LIMIT: " << node->getLimit().value() << "\n";
    }
    
    indent_--;
  }

  void visitQuerySpecification(QuerySpecification* node) override {
    printIndent();
    std::cout << "QuerySpecification:\n";
    indent_++;
    
    node->getSelect()->accept(this);
    
    if (node->getFrom().has_value()) {
      printIndent();
      std::cout << "FROM:\n";
      indent_++;
      node->getFrom().value()->accept(this);
      indent_--;
    }
    
    if (node->getWhere().has_value()) {
      printIndent();
      std::cout << "WHERE:\n";
      indent_++;
      node->getWhere().value()->accept(this);
      indent_--;
    }
    
    if (node->getGroupBy().has_value()) {
      node->getGroupBy().value()->accept(this);
    }
    
    if (node->getHaving().has_value()) {
      printIndent();
      std::cout << "HAVING:\n";
      indent_++;
      node->getHaving().value()->accept(this);
      indent_--;
    }
    
    indent_--;
  }

  void visitSelect(Select* node) override {
    printIndent();
    std::cout << "SELECT" << (node->isDistinct() ? " DISTINCT" : "") << ":\n";
    indent_++;
    for (const auto& item : node->getSelectItems()) {
      item->accept(this);
    }
    indent_--;
  }

  void visitSingleColumn(SingleColumn* node) override {
    printIndent();
    std::cout << "Column";
    if (node->getAlias().has_value()) {
      std::cout << " (alias: " << node->getAlias().value()->getValue() << ")";
    }
    std::cout << ":\n";
    indent_++;
    node->getExpression()->accept(this);
    indent_--;
  }

  void visitAllColumns(AllColumns* node) override {
    printIndent();
    std::cout << "All Columns (*)";
    if (node->getPrefix().has_value()) {
      std::cout << " with prefix";
    }
    std::cout << "\n";
  }

  void visitTable(Table* node) override {
    printIndent();
    std::cout << "Table: ";
    const auto& parts = node->getName()->getParts();
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) std::cout << ".";
      std::cout << parts[i];
    }
    std::cout << "\n";
  }

  void visitJoin(Join* node) override {
    printIndent();
    std::cout << "Join (";
    switch (node->getType()) {
      case Join::Type::INNER: std::cout << "INNER"; break;
      case Join::Type::LEFT: std::cout << "LEFT"; break;
      case Join::Type::RIGHT: std::cout << "RIGHT"; break;
      case Join::Type::FULL: std::cout << "FULL"; break;
      case Join::Type::CROSS: std::cout << "CROSS"; break;
      case Join::Type::IMPLICIT: std::cout << "IMPLICIT"; break;
    }
    std::cout << "):\n";
    
    indent_++;
    printIndent();
    std::cout << "Left:\n";
    indent_++;
    node->getLeft()->accept(this);
    indent_--;
    
    printIndent();
    std::cout << "Right:\n";
    indent_++;
    node->getRight()->accept(this);
    indent_--;
    
    if (node->getCriteria().has_value()) {
      printIndent();
      std::cout << "Criteria:\n";
      indent_++;
      node->getCriteria().value()->accept(this);
      indent_--;
    }
    indent_--;
  }

  void visitJoinOn(JoinOn* node) override {
    printIndent();
    std::cout << "JOIN ON:\n";
    indent_++;
    node->getExpression()->accept(this);
    indent_--;
  }

  void visitIdentifier(Identifier* node) override {
    printIndent();
    std::cout << "Identifier: " << node->getValue() << "\n";
  }

  void visitQualifiedName(QualifiedName* node) override {
    printIndent();
    std::cout << "QualifiedName: ";
    const auto& parts = node->getParts();
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) std::cout << ".";
      std::cout << parts[i];
    }
    std::cout << "\n";
  }

  // Literals
  void visitLongLiteral(LongLiteral* node) override {
    printIndent();
    std::cout << "Long: " << node->getValue() << "\n";
  }

  void visitStringLiteral(StringLiteral* node) override {
    printIndent();
    std::cout << "String: '" << node->getValue() << "'\n";
  }

  void visitBooleanLiteral(BooleanLiteral* node) override {
    printIndent();
    std::cout << "Boolean: " << (node->getValue() ? "true" : "false") << "\n";
  }

  void visitDoubleLiteral(DoubleLiteral* node) override {
    printIndent();
    std::cout << "Double: " << node->getValue() << "\n";
  }

  void visitNullLiteral(NullLiteral* node) override {
    printIndent();
    std::cout << "NULL\n";
  }

  // Expressions
  void visitArithmeticBinaryExpression(ArithmeticBinaryExpression* node) override {
    printIndent();
    std::cout << "Arithmetic (";
    switch (node->getOperator()) {
      case ArithmeticBinaryExpression::Operator::ADD: std::cout << "+"; break;
      case ArithmeticBinaryExpression::Operator::SUBTRACT: std::cout << "-"; break;
      case ArithmeticBinaryExpression::Operator::MULTIPLY: std::cout << "*"; break;
      case ArithmeticBinaryExpression::Operator::DIVIDE: std::cout << "/"; break;
      case ArithmeticBinaryExpression::Operator::MODULUS: std::cout << "%"; break;
    }
    std::cout << "):\n";
    
    indent_++;
    printIndent();
    std::cout << "Left:\n";
    indent_++;
    node->getLeft()->accept(this);
    indent_--;
    
    printIndent();
    std::cout << "Right:\n";
    indent_++;
    node->getRight()->accept(this);
    indent_--;
    indent_--;
  }

  void visitComparisonExpression(ComparisonExpression* node) override {
    printIndent();
    std::cout << "Comparison (";
    switch (node->getOperator()) {
      case ComparisonExpression::Operator::EQUAL: std::cout << "="; break;
      case ComparisonExpression::Operator::NOT_EQUAL: std::cout << "!="; break;
      case ComparisonExpression::Operator::LESS_THAN: std::cout << "<"; break;
      case ComparisonExpression::Operator::LESS_THAN_OR_EQUAL: std::cout << "<="; break;
      case ComparisonExpression::Operator::GREATER_THAN: std::cout << ">"; break;
      case ComparisonExpression::Operator::GREATER_THAN_OR_EQUAL: std::cout << ">="; break;
      case ComparisonExpression::Operator::IS_DISTINCT_FROM: std::cout << "IS DISTINCT FROM"; break;
    }
    std::cout << "):\n";
    
    indent_++;
    printIndent();
    std::cout << "Left:\n";
    indent_++;
    node->getLeft()->accept(this);
    indent_--;
    
    printIndent();
    std::cout << "Right:\n";
    indent_++;
    node->getRight()->accept(this);
    indent_--;
    indent_--;
  }

  void visitLogicalBinaryExpression(LogicalBinaryExpression* node) override {
    printIndent();
    std::cout << "Logical (";
    switch (node->getOperator()) {
      case LogicalBinaryExpression::Operator::AND: std::cout << "AND"; break;
      case LogicalBinaryExpression::Operator::OR: std::cout << "OR"; break;
    }
    std::cout << "):\n";
    
    indent_++;
    printIndent();
    std::cout << "Left:\n";
    indent_++;
    node->getLeft()->accept(this);
    indent_--;
    
    printIndent();
    std::cout << "Right:\n";
    indent_++;
    node->getRight()->accept(this);
    indent_--;
    indent_--;
  }

  void visitFunctionCall(FunctionCall* node) override {
    printIndent();
    std::cout << "FunctionCall: ";
    const auto& parts = node->getName()->getParts();
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) std::cout << ".";
      std::cout << parts[i];
    }
    
    if (node->isDistinct()) {
      std::cout << " (DISTINCT)";
    }
    std::cout << "\n";
    
    if (!node->getArguments().empty()) {
      indent_++;
      printIndent();
      std::cout << "Arguments:\n";
      indent_++;
      for (const auto& arg : node->getArguments()) {
        arg->accept(this);
      }
      indent_--;
      indent_--;
    }
  }

  void visitCast(Cast* node) override {
    printIndent();
    std::cout << "Cast to " << node->getType();
    if (node->isSafe()) {
      std::cout << " (SAFE)";
    }
    std::cout << ":\n";
    indent_++;
    node->getExpression()->accept(this);
    indent_--;
  }

  void visitBetweenPredicate(BetweenPredicate* node) override {
    printIndent();
    std::cout << "Between:\n";
    indent_++;
    printIndent();
    std::cout << "Value:\n";
    indent_++;
    node->getValue()->accept(this);
    indent_--;
    printIndent();
    std::cout << "Min:\n";
    indent_++;
    node->getMin()->accept(this);
    indent_--;
    printIndent();
    std::cout << "Max:\n";
    indent_++;
    node->getMax()->accept(this);
    indent_--;
    indent_--;
  }

  void visitInPredicate(InPredicate* node) override {
    printIndent();
    std::cout << "In:\n";
    indent_++;
    printIndent();
    std::cout << "Value:\n";
    indent_++;
    node->getValue()->accept(this);
    indent_--;
    printIndent();
    std::cout << "ValueList:\n";
    indent_++;
    node->getValueList()->accept(this);
    indent_--;
    indent_--;
  }

  // DDL Statements
  void visitCreateTable(CreateTable* node) override {
    printIndent();
    std::cout << "CREATE TABLE";
    if (node->isNotExists()) {
      std::cout << " IF NOT EXISTS";
    }
    std::cout << " ";
    const auto& parts = node->getName()->getParts();
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) std::cout << ".";
      std::cout << parts[i];
    }
    std::cout << "\n";
    
    if (!node->getElements().empty()) {
      indent_++;
      printIndent();
      std::cout << "Elements:\n";
      indent_++;
      for (const auto& element : node->getElements()) {
        element->accept(this);
      }
      indent_--;
      indent_--;
    }
  }

  void visitColumnDefinition(ColumnDefinition* node) override {
    printIndent();
    std::cout << "Column: " << node->getName()->getValue() 
              << " " << node->getType();
    if (!node->isNullable()) {
      std::cout << " NOT NULL";
    }
    if (node->getComment().has_value()) {
      std::cout << " COMMENT '" << node->getComment().value() << "'";
    }
    std::cout << "\n";
  }

private:
  void printIndent() {
    for (int i = 0; i < indent_; i++) {
      std::cout << "  ";
    }
  }
  
  int indent_ = 0;
};

} // namespace facebook::velox::sql

int main() {
  using namespace facebook::velox::sql;
  
  PrestoSqlAstParser parser;
  ComprehensiveAstPrinter printer;
  
  std::vector<std::string> examples = {
    "SELECT 1",
    "SELECT * FROM users",
    "SELECT id, name FROM users WHERE id > 10",
    "SELECT COUNT(*) FROM orders",
    "SELECT a + b * 2 FROM calculations",
    "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
    "SELECT id FROM users WHERE name BETWEEN 'A' AND 'Z'",
    "SELECT * FROM users WHERE id IN (1, 2, 3)",
    "SELECT CAST(id AS VARCHAR) FROM users",
    "SELECT * FROM users WHERE name IS NOT NULL",
    "SELECT * FROM users ORDER BY name ASC LIMIT 10"
  };
  
  for (const auto& sql : examples) {
    std::cout << "=== Parsing: " << sql << " ===\n";
    
    auto result = parser.parseStatement(sql);
    if (result.hasErrors) {
      std::cout << "Parse Error: " << result.errorMessage << "\n";
    } else {
      result.ast->accept(&printer);
      
      if (!result.warnings.empty()) {
        std::cout << "Warnings:\n";
        for (const auto& warning : result.warnings) {
          std::cout << "  - " << warning << "\n";
        }
      }
    }
    
    std::cout << "\n";
  }
  
  return 0;
}