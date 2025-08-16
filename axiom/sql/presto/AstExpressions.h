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

#include "AstNode.h"
#include "AstVisitor.h"
#include <optional>
#include <vector>

namespace facebook::velox::sql {

// Forward declarations
class QualifiedName;

// Identifiers and References
class Identifier : public Expression {
public:
  explicit Identifier(const std::string& value) 
      : Expression(NodeType::kIdentifier), value_(value) {}
  
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
};

class QualifiedName : public Expression {
public:
  explicit QualifiedName(const std::vector<std::string>& parts)
      : Expression(NodeType::kQualifiedName), parts_(parts) {}
  
  const std::vector<std::string>& getParts() const { return parts_; }
  std::string getSuffix() const { return parts_.empty() ? "" : parts_.back(); }
  
  void accept(AstVisitor* visitor) override;

private:
  std::vector<std::string> parts_;
};

class DereferenceExpression : public Expression {
public:
  DereferenceExpression(ExpressionPtr base, std::shared_ptr<Identifier> field)
      : Expression(NodeType::kDereferenceExpression), base_(base), field_(field) {}
  
  const ExpressionPtr& getBase() const { return base_; }
  const std::shared_ptr<Identifier>& getField() const { return field_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr base_;
  std::shared_ptr<Identifier> field_;
};

class FieldReference : public Expression {
public:
  explicit FieldReference(int fieldIndex)
      : Expression(NodeType::kFieldReference), fieldIndex_(fieldIndex) {}
  
  int getFieldIndex() const { return fieldIndex_; }
  void accept(AstVisitor* visitor) override;

private:
  int fieldIndex_;
};

class SymbolReference : public Expression {
public:
  explicit SymbolReference(const std::string& name)
      : Expression(NodeType::kSymbolReference), name_(name) {}
  
  const std::string& getName() const { return name_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string name_;
};

class Parameter : public Expression {
public:
  explicit Parameter(int position)
      : Expression(NodeType::kParameter), position_(position) {}
  
  int getPosition() const { return position_; }
  void accept(AstVisitor* visitor) override;

private:
  int position_;
};

// Arithmetic Expressions
class ArithmeticBinaryExpression : public Expression {
public:
  enum class Operator { ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULUS };
  
  ArithmeticBinaryExpression(Operator op, ExpressionPtr left, ExpressionPtr right)
      : Expression(NodeType::kArithmeticBinaryExpression), 
        operator_(op), left_(left), right_(right) {}
  
  Operator getOperator() const { return operator_; }
  const ExpressionPtr& getLeft() const { return left_; }
  const ExpressionPtr& getRight() const { return right_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Operator operator_;
  ExpressionPtr left_;
  ExpressionPtr right_;
};

class ArithmeticUnaryExpression : public Expression {
public:
  enum class Sign { PLUS, MINUS };
  
  ArithmeticUnaryExpression(Sign sign, ExpressionPtr value)
      : Expression(NodeType::kArithmeticUnaryExpression), sign_(sign), value_(value) {}
  
  Sign getSign() const { return sign_; }
  const ExpressionPtr& getValue() const { return value_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Sign sign_;
  ExpressionPtr value_;
};

// Comparison Expressions
class ComparisonExpression : public Expression {
public:
  enum class Operator { 
    EQUAL, NOT_EQUAL, LESS_THAN, LESS_THAN_OR_EQUAL, 
    GREATER_THAN, GREATER_THAN_OR_EQUAL, IS_DISTINCT_FROM 
  };
  
  ComparisonExpression(Operator op, ExpressionPtr left, ExpressionPtr right)
      : Expression(NodeType::kComparisonExpression), 
        operator_(op), left_(left), right_(right) {}
  
  Operator getOperator() const { return operator_; }
  const ExpressionPtr& getLeft() const { return left_; }
  const ExpressionPtr& getRight() const { return right_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Operator operator_;
  ExpressionPtr left_;
  ExpressionPtr right_;
};

class BetweenPredicate : public Expression {
public:
  BetweenPredicate(ExpressionPtr value, ExpressionPtr min, ExpressionPtr max)
      : Expression(NodeType::kBetweenPredicate), value_(value), min_(min), max_(max) {}
  
  const ExpressionPtr& getValue() const { return value_; }
  const ExpressionPtr& getMin() const { return min_; }
  const ExpressionPtr& getMax() const { return max_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr value_;
  ExpressionPtr min_;
  ExpressionPtr max_;
};

class InPredicate : public Expression {
public:
  InPredicate(ExpressionPtr value, ExpressionPtr valueList)
      : Expression(NodeType::kInPredicate), value_(value), valueList_(valueList) {}
  
  const ExpressionPtr& getValue() const { return value_; }
  const ExpressionPtr& getValueList() const { return valueList_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr value_;
  ExpressionPtr valueList_;
};

class InListExpression : public Expression {
public:
  explicit InListExpression(const std::vector<ExpressionPtr>& values)
      : Expression(NodeType::kInListExpression), values_(values) {}
  
  const std::vector<ExpressionPtr>& getValues() const { return values_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> values_;
};

class IsNullPredicate : public Expression {
public:
  explicit IsNullPredicate(ExpressionPtr value)
      : Expression(NodeType::kIsNullPredicate), value_(value) {}
  
  const ExpressionPtr& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr value_;
};

class IsNotNullPredicate : public Expression {
public:
  explicit IsNotNullPredicate(ExpressionPtr value)
      : Expression(NodeType::kIsNotNullPredicate), value_(value) {}
  
  const ExpressionPtr& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr value_;
};

class LikePredicate : public Expression {
public:
  LikePredicate(ExpressionPtr value, ExpressionPtr pattern, std::optional<ExpressionPtr> escape = std::nullopt)
      : Expression(NodeType::kLikePredicate), value_(value), pattern_(pattern), escape_(escape) {}
  
  const ExpressionPtr& getValue() const { return value_; }
  const ExpressionPtr& getPattern() const { return pattern_; }
  const std::optional<ExpressionPtr>& getEscape() const { return escape_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr value_;
  ExpressionPtr pattern_;
  std::optional<ExpressionPtr> escape_;
};

class ExistsPredicate : public Expression {
public:
  explicit ExistsPredicate(ExpressionPtr subquery)
      : Expression(NodeType::kExistsPredicate), subquery_(subquery) {}
  
  const ExpressionPtr& getSubquery() const { return subquery_; }
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr subquery_;
};

class QuantifiedComparisonExpression : public Expression {
public:
  enum class Quantifier { ALL, ANY, SOME };
  
  QuantifiedComparisonExpression(
      ComparisonExpression::Operator operator_, 
      Quantifier quantifier, 
      ExpressionPtr value, 
      ExpressionPtr subquery)
      : Expression(NodeType::kQuantifiedComparisonExpression),
        operator_(operator_), quantifier_(quantifier), value_(value), subquery_(subquery) {}
  
  ComparisonExpression::Operator getOperator() const { return operator_; }
  Quantifier getQuantifier() const { return quantifier_; }
  const ExpressionPtr& getValue() const { return value_; }
  const ExpressionPtr& getSubquery() const { return subquery_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ComparisonExpression::Operator operator_;
  Quantifier quantifier_;
  ExpressionPtr value_;
  ExpressionPtr subquery_;
};

// Logical Expressions
class LogicalBinaryExpression : public Expression {
public:
  enum class Operator { AND, OR };
  
  LogicalBinaryExpression(Operator op, ExpressionPtr left, ExpressionPtr right)
      : Expression(NodeType::kLogicalBinaryExpression), 
        operator_(op), left_(left), right_(right) {}
  
  Operator getOperator() const { return operator_; }
  const ExpressionPtr& getLeft() const { return left_; }
  const ExpressionPtr& getRight() const { return right_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Operator operator_;
  ExpressionPtr left_;
  ExpressionPtr right_;
};

class NotExpression : public Expression {
public:
  explicit NotExpression(ExpressionPtr value)
      : Expression(NodeType::kNotExpression), value_(value) {}
  
  const ExpressionPtr& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr value_;
};

} // namespace facebook::velox::sql