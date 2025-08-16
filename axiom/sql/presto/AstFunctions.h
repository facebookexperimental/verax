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
class Window;
class OrderBy;

// Conditional Expressions
class IfExpression : public Expression {
public:
  IfExpression(ExpressionPtr condition, ExpressionPtr trueValue, std::optional<ExpressionPtr> falseValue = std::nullopt)
      : Expression(NodeType::kIfExpression), condition_(condition), trueValue_(trueValue), falseValue_(falseValue) {}
  
  const ExpressionPtr& getCondition() const { return condition_; }
  const ExpressionPtr& getTrueValue() const { return trueValue_; }
  const std::optional<ExpressionPtr>& getFalseValue() const { return falseValue_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr condition_;
  ExpressionPtr trueValue_;
  std::optional<ExpressionPtr> falseValue_;
};

class CoalesceExpression : public Expression {
public:
  explicit CoalesceExpression(const std::vector<ExpressionPtr>& operands)
      : Expression(NodeType::kCoalesceExpression), operands_(operands) {}
  
  const std::vector<ExpressionPtr>& getOperands() const { return operands_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> operands_;
};

class NullIfExpression : public Expression {
public:
  NullIfExpression(ExpressionPtr first, ExpressionPtr second)
      : Expression(NodeType::kNullIfExpression), first_(first), second_(second) {}
  
  const ExpressionPtr& getFirst() const { return first_; }
  const ExpressionPtr& getSecond() const { return second_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr first_;
  ExpressionPtr second_;
};

class WhenClause : public Node {
public:
  WhenClause(ExpressionPtr operand, ExpressionPtr result)
      : Node(NodeType::kWhenClause), operand_(operand), result_(result) {}
  
  const ExpressionPtr& getOperand() const { return operand_; }
  const ExpressionPtr& getResult() const { return result_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr operand_;
  ExpressionPtr result_;
};

class SearchedCaseExpression : public Expression {
public:
  SearchedCaseExpression(
      const std::vector<std::shared_ptr<WhenClause>>& whenClauses,
      std::optional<ExpressionPtr> defaultValue = std::nullopt)
      : Expression(NodeType::kSearchedCaseExpression), 
        whenClauses_(whenClauses), defaultValue_(defaultValue) {}
  
  const std::vector<std::shared_ptr<WhenClause>>& getWhenClauses() const { return whenClauses_; }
  const std::optional<ExpressionPtr>& getDefaultValue() const { return defaultValue_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::vector<std::shared_ptr<WhenClause>> whenClauses_;
  std::optional<ExpressionPtr> defaultValue_;
};

class SimpleCaseExpression : public Expression {
public:
  SimpleCaseExpression(
      ExpressionPtr operand,
      const std::vector<std::shared_ptr<WhenClause>>& whenClauses,
      std::optional<ExpressionPtr> defaultValue = std::nullopt)
      : Expression(NodeType::kSimpleCaseExpression),
        operand_(operand), whenClauses_(whenClauses), defaultValue_(defaultValue) {}
  
  const ExpressionPtr& getOperand() const { return operand_; }
  const std::vector<std::shared_ptr<WhenClause>>& getWhenClauses() const { return whenClauses_; }
  const std::optional<ExpressionPtr>& getDefaultValue() const { return defaultValue_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr operand_;
  std::vector<std::shared_ptr<WhenClause>> whenClauses_;
  std::optional<ExpressionPtr> defaultValue_;
};

class TryExpression : public Expression {
public:
  explicit TryExpression(ExpressionPtr innerExpression)
      : Expression(NodeType::kTryExpression), innerExpression_(innerExpression) {}
  
  const ExpressionPtr& getInnerExpression() const { return innerExpression_; }
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr innerExpression_;
};

// Function and Call Expressions
class FunctionCall : public Expression {
public:
  FunctionCall(
      std::shared_ptr<QualifiedName> name,
      std::optional<std::shared_ptr<Window>> window,
      bool distinct,
      const std::vector<ExpressionPtr>& arguments)
      : Expression(NodeType::kFunctionCall), 
        name_(name), window_(window), distinct_(distinct), arguments_(arguments) {}
  
  const std::shared_ptr<QualifiedName>& getName() const { return name_; }
  const std::optional<std::shared_ptr<Window>>& getWindow() const { return window_; }
  bool isDistinct() const { return distinct_; }
  const std::vector<ExpressionPtr>& getArguments() const { return arguments_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> name_;
  std::optional<std::shared_ptr<Window>> window_;
  bool distinct_;
  std::vector<ExpressionPtr> arguments_;
};

class Cast : public Expression {
public:
  Cast(ExpressionPtr expression, const std::string& type, bool safe = false)
      : Expression(NodeType::kCast), expression_(expression), type_(type), safe_(safe) {}
  
  const ExpressionPtr& getExpression() const { return expression_; }
  const std::string& getType() const { return type_; }
  bool isSafe() const { return safe_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr expression_;
  std::string type_;
  bool safe_;
};

class Extract : public Expression {
public:
  enum class Field { 
    YEAR, QUARTER, MONTH, WEEK, DAY, DAY_OF_MONTH, DAY_OF_WEEK, 
    DOW, DAY_OF_YEAR, DOY, HOUR, MINUTE, SECOND, TIMEZONE_HOUR, TIMEZONE_MINUTE
  };
  
  Extract(ExpressionPtr expression, Field field)
      : Expression(NodeType::kExtract), expression_(expression), field_(field) {}
  
  const ExpressionPtr& getExpression() const { return expression_; }
  Field getField() const { return field_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr expression_;
  Field field_;
};

class CurrentTime : public Expression {
public:
  enum class Function { 
    TIME, DATE, TIMESTAMP, LOCALTIME, LOCALTIMESTAMP 
  };
  
  CurrentTime(Function function, std::optional<int> precision = std::nullopt)
      : Expression(NodeType::kCurrentTime), function_(function), precision_(precision) {}
  
  Function getFunction() const { return function_; }
  const std::optional<int>& getPrecision() const { return precision_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Function function_;
  std::optional<int> precision_;
};

class CurrentUser : public Expression {
public:
  CurrentUser() : Expression(NodeType::kCurrentUser) {}
  void accept(AstVisitor* visitor) override;
};

class AtTimeZone : public Expression {
public:
  AtTimeZone(ExpressionPtr value, ExpressionPtr timeZone)
      : Expression(NodeType::kAtTimeZone), value_(value), timeZone_(timeZone) {}
  
  const ExpressionPtr& getValue() const { return value_; }
  const ExpressionPtr& getTimeZone() const { return timeZone_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr value_;
  ExpressionPtr timeZone_;
};

// Complex Expressions
class SubqueryExpression : public Expression {
public:
  explicit SubqueryExpression(StatementPtr query)
      : Expression(NodeType::kSubqueryExpression), query_(query) {}
  
  const StatementPtr& getQuery() const { return query_; }
  void accept(AstVisitor* visitor) override;

private:
  StatementPtr query_;
};

class ArrayConstructor : public Expression {
public:
  explicit ArrayConstructor(const std::vector<ExpressionPtr>& values)
      : Expression(NodeType::kArrayConstructor), values_(values) {}
  
  const std::vector<ExpressionPtr>& getValues() const { return values_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> values_;
};

class Row : public Expression {
public:
  explicit Row(const std::vector<ExpressionPtr>& items)
      : Expression(NodeType::kRow), items_(items) {}
  
  const std::vector<ExpressionPtr>& getItems() const { return items_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> items_;
};

class SubscriptExpression : public Expression {
public:
  SubscriptExpression(ExpressionPtr base, ExpressionPtr index)
      : Expression(NodeType::kSubscriptExpression), base_(base), index_(index) {}
  
  const ExpressionPtr& getBase() const { return base_; }
  const ExpressionPtr& getIndex() const { return index_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr base_;
  ExpressionPtr index_;
};

class LambdaArgumentDeclaration : public Node {
public:
  explicit LambdaArgumentDeclaration(std::shared_ptr<Identifier> name)
      : Node(NodeType::kLambdaArgumentDeclaration), name_(name) {}
  
  const std::shared_ptr<Identifier>& getName() const { return name_; }
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<Identifier> name_;
};

class LambdaExpression : public Expression {
public:
  LambdaExpression(
      const std::vector<std::shared_ptr<LambdaArgumentDeclaration>>& arguments,
      ExpressionPtr body)
      : Expression(NodeType::kLambdaExpression), arguments_(arguments), body_(body) {}
  
  const std::vector<std::shared_ptr<LambdaArgumentDeclaration>>& getArguments() const { return arguments_; }
  const ExpressionPtr& getBody() const { return body_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::vector<std::shared_ptr<LambdaArgumentDeclaration>> arguments_;
  ExpressionPtr body_;
};

class BindExpression : public Expression {
public:
  BindExpression(const std::vector<ExpressionPtr>& values, ExpressionPtr function)
      : Expression(NodeType::kBindExpression), values_(values), function_(function) {}
  
  const std::vector<ExpressionPtr>& getValues() const { return values_; }
  const ExpressionPtr& getFunction() const { return function_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> values_;
  ExpressionPtr function_;
};

class GroupingOperation : public Expression {
public:
  explicit GroupingOperation(const std::vector<std::shared_ptr<QualifiedName>>& groupingColumns)
      : Expression(NodeType::kGroupingOperation), groupingColumns_(groupingColumns) {}
  
  const std::vector<std::shared_ptr<QualifiedName>>& getGroupingColumns() const { return groupingColumns_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<std::shared_ptr<QualifiedName>> groupingColumns_;
};

class TableVersionExpression : public Expression {
public:
  enum class TableVersionType { TIMESTAMP, VERSION };
  
  TableVersionExpression(TableVersionType type, ExpressionPtr expression)
      : Expression(NodeType::kTableVersionExpression), type_(type), expression_(expression) {}
  
  TableVersionType getTableVersionType() const { return type_; }
  const ExpressionPtr& getExpression() const { return expression_; }
  
  void accept(AstVisitor* visitor) override;

private:
  TableVersionType type_;
  ExpressionPtr expression_;
};

} // namespace facebook::velox::sql