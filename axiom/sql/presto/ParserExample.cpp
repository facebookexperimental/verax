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

#include "PrestoSqlAstParser.h"
#include "AstNodesAll.h"
#include <iostream>

namespace facebook::velox::sql {

class SimpleAstPrinter : public AstVisitor {
public:
  void visitQuery(Query* node) override {
    std::cout << "Query:\n";
    indent_++;
    node->getQueryBody()->accept(this);
    indent_--;
  }

  void visitQuerySpecification(QuerySpecification* node) override {
    printIndent();
    std::cout << "QuerySpecification:\n";
    indent_++;
    
    node->getSelect()->accept(this);
    
    if (node->getFrom().has_value()) {
      node->getFrom().value()->accept(this);
    }
    
    if (node->getWhere().has_value()) {
      printIndent();
      std::cout << "WHERE:\n";
      indent_++;
      node->getWhere().value()->accept(this);
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
      std::cout << " (alias: " << node->getAlias().value() << ")";
    }
    std::cout << ":\n";
    indent_++;
    node->getExpression()->accept(this);
    indent_--;
  }

  void visitTable(Table* node) override {
    printIndent();
    std::cout << "FROM Table: " << node->getName() << "\n";
  }

  void visitIdentifier(Identifier* node) override {
    printIndent();
    std::cout << "Identifier: " << node->getValue() << "\n";
  }

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
    std::cout << "FunctionCall: " << node->getName();
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
  SimpleAstPrinter printer;
  
  std::vector<std::string> examples = {
    "SELECT 1",
    "SELECT * FROM users",
    "SELECT id, name FROM users WHERE id > 10",
    "SELECT COUNT(*) FROM orders",
    "SELECT a + b * 2 FROM calculations"
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