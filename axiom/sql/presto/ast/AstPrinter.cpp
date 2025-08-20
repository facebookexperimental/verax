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

#include "axiom/sql/presto/ast/AstPrinter.h"
#include <fmt/format.h>

namespace axiom::sql::presto {

void AstPrinter::visitQuery(Query* node) {
  printHeader("Query", node);

  indent_++;
  node->queryBody()->accept(this);
  indent_--;

  if (node->orderBy() != nullptr) {
    indent_++;
    node->orderBy()->accept(this);
    indent_--;
  }

  if (const auto& offset = node->offset()) {
    indent_++;
    printHeader(
        "OFFSET", offset, [&](std::ostream& out) { out << offset->offset(); });
    indent_--;
  }

  if (node->limit().has_value()) {
    indent_++;
    printHeader("LIMIT", node, [&](std::ostream& out) {
      out << node->limit().value();
    });
    indent_--;
  }
}

void AstPrinter::visitQuerySpecification(QuerySpecification* node) {
  printHeader("QuerySpecification", node);
  indent_++;

  node->select()->accept(this);

  if (node->from() != nullptr) {
    node->from()->accept(this);
  }

  if (node->where() != nullptr) {
    printChild("WHERE", node->where());
  }

  if (node->groupBy() != nullptr) {
    node->groupBy()->accept(this);
  }

  if (node->having() != nullptr) {
    printChild("HAVING", node->having());
  }

  indent_--;
}

namespace {
std::string toString(ArithmeticBinaryExpression::Operator op) {
  switch (op) {
    case ArithmeticBinaryExpression::Operator::kAdd:
      return "+";
    case ArithmeticBinaryExpression::Operator::kSubtract:
      return "-";
    case ArithmeticBinaryExpression::Operator::kMultiply:
      return "*";
    case ArithmeticBinaryExpression::Operator::kDivide:
      return "/";
    case ArithmeticBinaryExpression::Operator::kModulus:
      return "%";
  }
  throw std::runtime_error("Unsupported arithmetic operator");
}

std::string toString(ComparisonExpression::Operator op) {
  switch (op) {
    case ComparisonExpression::Operator::kEqual:
      return "=";
    case ComparisonExpression::Operator::kNotEqual:
      return "!=";
    case ComparisonExpression::Operator::kLessThan:
      return "<";
    case ComparisonExpression::Operator::kLessThanOrEqual:
      return "<=";
    case ComparisonExpression::Operator::kGreaterThan:
      return ">";
    case ComparisonExpression::Operator::kGreaterThanOrEqual:
      return ">=";
    default:
      throw std::runtime_error("Unsupported comparison operator");
  }
}

std::string toString(SortItem::Ordering ordering) {
  switch (ordering) {
    case SortItem::Ordering::kAscending:
      return "ASC";
    case SortItem::Ordering::kDescending:
      return "DESC";
    default:
      throw std::runtime_error("Unsupported sort ordering");
  }
}

std::string toString(SortItem::NullOrdering nullOrdering) {
  switch (nullOrdering) {
    case SortItem::NullOrdering::kFirst:
      return "NULLS FIRST";
    case SortItem::NullOrdering::kLast:
      return "NULLS LAST";
    case SortItem::NullOrdering::kUndefined:
      return "NULLS UNDEFINED";
    default:
      throw std::runtime_error("Unsupported null ordering");
  }
}
} // namespace

void AstPrinter::visitArithmeticBinaryExpression(
    ArithmeticBinaryExpression* node) {
  printHeader("Arithmetic", node, [&](std::ostream& out) {
    out << toString(node->op());
  });

  indent_++;
  printChild("Left", node->left());
  printChild("Right", node->right());
  indent_--;
}

void AstPrinter::visitSelect(Select* node) {
  printHeader("SELECT", node, [&](std::ostream& out) {
    if (node->isDistinct()) {
      out << "DISTINCT";
    };
  });

  indent_++;
  for (const auto& item : node->selectItems()) {
    item->accept(this);
  }
  indent_--;
}

void AstPrinter::visitSingleColumn(SingleColumn* node) {
  printHeader("Column", node, [&](std::ostream& out) {
    if (node->alias() != nullptr) {
      out_ << node->alias()->value();
    }
  });

  indent_++;
  node->expression()->accept(this);
  indent_--;
}

void AstPrinter::visitAliasedRelation(AliasedRelation* node) {
  printHeader("AliasedRelation", node, [&](std::ostream& out) {
    out << node->alias()->value();
  });

  indent_++;
  node->relation()->accept(this);
  indent_--;
}

void AstPrinter::visitTable(Table* node) {
  printHeader("FROM", node, [&](std::ostream& out) {
    out << "Table(" << node->name()->suffix() << ")";
  });
}

void AstPrinter::visitIdentifier(Identifier* node) {
  printHeader(
      "Identifier", node, [&](std::ostream& out) { out << node->value(); });
}

void AstPrinter::visitLongLiteral(LongLiteral* node) {
  printHeader("Long", node, [&](std::ostream& out) { out << node->value(); });
}

void AstPrinter::visitStringLiteral(StringLiteral* node) {
  printHeader("String", node, [&](std::ostream& out) { out << node->value(); });
}

void AstPrinter::visitBooleanLiteral(BooleanLiteral* node) {
  printHeader("Boolean", node, [&](std::ostream& out) {
    out << (node->value() ? "true" : "false");
  });
}

void AstPrinter::visitGroupBy(GroupBy* node) {
  printHeader("GROUP BY", node);

  indent_++;
  for (const auto& item : node->groupingElements()) {
    item->accept(this);
  }
  indent_--;
}

void AstPrinter::visitSimpleGroupBy(SimpleGroupBy* node) {
  for (const auto& item : node->expressions()) {
    item->accept(this);
  }
}

void AstPrinter::visitFunctionCall(FunctionCall* node) {
  printHeader("FunctionCall", node, [&](std::ostream& out) {
    out << node->name()->suffix();

    if (node->isDistinct()) {
      out << " (DISTINCT)";
    }
  });

  if (!node->arguments().empty()) {
    indent_++;
    printIndent();
    out_ << "Arguments:\n";
    indent_++;
    for (const auto& arg : node->arguments()) {
      arg->accept(this);
    }
    indent_--;
    indent_--;
  }
}

void AstPrinter::visitOrderBy(OrderBy* node) {
  printHeader("ORDER BY", node);

  indent_++;
  for (const auto& item : node->sortItems()) {
    item->accept(this);
  }
  indent_--;
}

void AstPrinter::visitSortItem(SortItem* node) {
  printHeader("SortItem", node, [&](std::ostream& out) {
    out << toString(node->ordering());
    if (node->nullOrdering() != SortItem::NullOrdering::kUndefined) {
      out << " " << toString(node->nullOrdering());
    }
  });

  indent_++;
  node->sortKey()->accept(this);
  indent_--;
}

void AstPrinter::visitDereferenceExpression(DereferenceExpression* node) {
  printHeader("Dereference", node);

  indent_++;
  node->base()->accept(this);
  printChild("Field", node->field());
  indent_--;
}

void AstPrinter::visitAllColumns(AllColumns* node) {
  printHeader("AllColumns", node, [&](std::ostream& out) { out << "*"; });
}

void AstPrinter::visitJoin(Join* node) {
  printHeader("JOIN", node);

  indent_++;
  printChild("Left", node->left());
  printChild("Right", node->right());
  if (node->criteria() != nullptr) {
    printChild("Criteria", node->criteria());
  }
  indent_--;
}

void AstPrinter::visitJoinOn(JoinOn* node) {
  printHeader("JoinOn", node);

  indent_++;
  node->expression()->accept(this);
  indent_--;
}

void AstPrinter::visitComparisonExpression(ComparisonExpression* node) {
  printHeader("Comparison", node, [&](std::ostream& out) {
    out << toString(node->op());
  });

  indent_++;
  printChild("Left", node->left());
  printChild("Right", node->right());
  indent_--;
}

void AstPrinter::printHeader(
    const std::string& name,
    Node* node,
    const std::function<void(std::ostream& out)>& printDetails) const {
  out_ << std::string(indent_ * 2, ' ') << name << " ";

  out_ << fmt::format(
      "({}:{})", node->location().line, node->location().charPosition);

  out_ << ": ";

  if (printDetails != nullptr) {
    printDetails(out_);
  }

  out_ << std::endl;
}

void AstPrinter::printHeader(
    const std::string& name,
    const std::shared_ptr<Node>& node,
    const std::function<void(std::ostream& out)>& printDetails) const {
  if (node != nullptr) {
    printHeader(name, node.get(), printDetails);
  }
}

void AstPrinter::printChild(
    const std::string& name,
    const std::shared_ptr<Node>& node) {
  printHeader(name, node);
  indent_++;
  node->accept(this);
  indent_--;
}

void AstPrinter::printIndent() {
  out_ << std::string(indent_ * 2, ' ');
}

} // namespace axiom::sql::presto
