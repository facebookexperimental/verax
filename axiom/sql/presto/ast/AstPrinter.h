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

#include <functional>

#include "axiom/sql/presto/ast/AstVisitor.h"

namespace axiom::sql::presto {

class AstPrinter : public AstVisitor {
 public:
  explicit AstPrinter(std::ostream& out) : out_(out) {}

  void visitQuery(Query* node) override;

  void visitQuerySpecification(QuerySpecification* node) override;

  void visitSelect(Select* node) override;

  void visitSingleColumn(SingleColumn* node) override;

  void visitAliasedRelation(AliasedRelation* node) override;

  void visitTable(Table* node) override;

  void visitIdentifier(Identifier* node) override;

  void visitLongLiteral(LongLiteral* node) override;

  void visitStringLiteral(StringLiteral* node) override;

  void visitBooleanLiteral(BooleanLiteral* node) override;

  void visitGroupBy(GroupBy* node) override;

  void visitSimpleGroupBy(SimpleGroupBy* node) override;

  void visitArithmeticBinaryExpression(
      ArithmeticBinaryExpression* node) override;

  void visitComparisonExpression(ComparisonExpression* node) override;

  void visitFunctionCall(FunctionCall* node) override;

  void visitOrderBy(OrderBy* node) override;

  void visitSortItem(SortItem* node) override;

  void visitDereferenceExpression(DereferenceExpression* node) override;

  void visitAllColumns(AllColumns* node) override;

  void visitJoin(Join* node) override;

  void visitJoinOn(JoinOn* node) override;

 private:
  void defaultVisit(Node* node) override {
    printIndent();
    out_ << "visit" << NodeTypeName::toName(node->type()) << std::endl;
  }

  void printHeader(
      const std::string& name,
      Node* node,
      const std::function<void(std::ostream& out)>& printDetails =
          nullptr) const;

  void printHeader(
      const std::string& name,
      const std::shared_ptr<Node>& node,
      const std::function<void(std::ostream& out)>& printDetails =
          nullptr) const;

  void printChild(const std::string& name, const std::shared_ptr<Node>& node);

  void printIndent();

  std::ostream& out_;
  int indent_ = 0;
};
} // namespace axiom::sql::presto
