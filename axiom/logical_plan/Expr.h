/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/common/Enums.h"
#include "velox/type/Variant.h"

namespace facebook::velox::logical_plan {

enum class ExprKind {
  kInputReference = 0,
  kConstant = 1,
  kCall = 2,
  kSpecialForm = 3,
  kAggregate = 4,
  kWindow = 5,
  kLambda = 6,
  kSubquery = 7,
};

class Expr;
using ExprPtr = std::shared_ptr<const Expr>;

class ExprVisitor;
class ExprVisitorContext;

/// Base class for all expressions. Every expression has a return type and zero
/// or more inputs. Leaf nodes like Constant and InputReference have no inputs.
/// Call may have many inputs.
class Expr {
 public:
  Expr(ExprKind kind, const TypePtr& type, const std::vector<ExprPtr>& inputs)
      : kind_{kind}, type_{type}, inputs_{inputs} {
    VELOX_USER_CHECK_NOT_NULL(type);
    for (const auto& input : inputs) {
      VELOX_USER_CHECK_NOT_NULL(input);
    }
  }

  virtual ~Expr() = default;

  ExprKind kind() const {
    return kind_;
  }

  const TypePtr& type() const {
    return type_;
  }

  /// Convenience getter for the type kind. A shortcut for type()->kind().
  TypeKind typeKind() const {
    return type_->kind();
  }

  const std::vector<ExprPtr>& inputs() const {
    return inputs_;
  }

  /// Convenience getter for the input at the specified index. A shortcut for
  /// inputs().at(index).
  const ExprPtr& inputAt(int32_t index) const {
    VELOX_USER_CHECK_GE(index, 0);
    VELOX_USER_CHECK_LT(index, inputs_.size());
    return inputs_.at(index);
  }

  bool isInputReference() const {
    return kind_ == ExprKind::kInputReference;
  }

  bool isConstant() const {
    return kind_ == ExprKind::kConstant;
  }

  bool isCall() const {
    return kind_ == ExprKind::kCall;
  }

  bool isSpecialForm() const {
    return kind_ == ExprKind::kSpecialForm;
  }

  bool isAggregate() const {
    return kind_ == ExprKind::kAggregate;
  }

  bool isWindow() const {
    return kind_ == ExprKind::kWindow;
  }

  bool isLambda() const {
    return kind_ == ExprKind::kLambda;
  }

  bool isSubquery() const {
    return kind_ == ExprKind::kSubquery;
  }

  template <typename T>
  const T* asUnchecked() const {
    return dynamic_cast<const T*>(this);
  }

  virtual void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const = 0;

 private:
  const ExprKind kind_;
  const TypePtr type_;
  const std::vector<ExprPtr> inputs_;
};

/// Reference to an input column.
class InputReferenceExpr : public Expr {
 public:
  /// @param type Type of the referenced column.
  /// @param name Name of the referenced column. All names in the plan tree must
  /// be unique. Columns that are being passed through without modifications can
  /// keep their names. Two columns with the same name must represent the same
  /// data. Names cannot be empty.
  InputReferenceExpr(const TypePtr& type, const std::string& name)
      : Expr(ExprKind::kInputReference, type, {}), name_{name} {}

  const std::string& name() const {
    return name_;
  }

  void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const override;

 private:
  const std::string name_;
};

using InputReferenceExprPtr = std::shared_ptr<const InputReferenceExpr>;

/// Literal value.
class ConstantExpr : public Expr {
 public:
  ConstantExpr(const TypePtr& type, std::shared_ptr<const Variant> value)
      : Expr(ExprKind::kConstant, type, {}), value_{std::move(value)} {
    VELOX_USER_CHECK_NOT_NULL(value_);
    VELOX_USER_CHECK(
        value_->isTypeCompatible(type),
        "Constant value doesn't match its type: {} vs. {}",
        type->toString(),
        value_->inferType()->toString());
  }

  const std::shared_ptr<const Variant>& value() const {
    return value_;
  }

  bool isNull() const {
    return value_->isNull();
  }

  void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const override;

 private:
  const std::shared_ptr<const Variant> value_;
};

using ConstantExprPtr = std::shared_ptr<const ConstantExpr>;

/// Scalar function call.
class CallExpr : public Expr {
 public:
  /// @param type Return type of the call.
  /// @param name Name of the scalar function. Cannot be any of the special
  /// forms.
  /// @param inputs Zero or more inputs / arguments.
  CallExpr(
      const TypePtr& type,
      const std::string& name,
      const std::vector<ExprPtr>& inputs);

  template <typename... T>
  CallExpr(const TypePtr& type, const std::string& name, T... inputs)
      : CallExpr(
            type,
            name,
            std::vector<ExprPtr>{std::forward<ExprPtr>(inputs)...}) {}

  const std::string& name() const {
    return name_;
  }

  void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const override;

 private:
  const std::string name_;
};

using CallExprPtr = std::shared_ptr<const CallExpr>;

enum class SpecialForm {
  /// Logical AND. Requires two or more boolean inputs. Commutative. The result
  /// doesn’t depend on the order of inputs.
  ///
  /// AND(false, <anything>) => false
  /// AND(true, true) => true
  /// AND(true, NULL) => NULL
  /// AND(true, throws) => throws
  /// AND(NULL, throws) => throws
  ///
  /// Returns true only if all inputs evaluate to true.
  ///
  /// Returns false if at least one input evaluates to false even if some inputs
  /// return null or throw.
  ///
  /// Propagates the exception if an input throws and no other input returns
  /// false.
  ///
  /// Returns NULL if at least one input is NULL and no other input evaluates to
  /// false or throws.
  ///
  /// Stops evaluating inputs once the result is determined (e.g. an input
  /// evaluates to false).
  ///
  /// Doesn't guarantee the order in which inputs are evaluated.
  kAnd = 0,

  /// Logical OR. Requires two or more boolean inputs. Commutative. The result
  /// doesn’t depend on the order of inputs.
  ///
  /// AND(true, <anything>) => true
  /// AND(false, true) => false
  /// AND(false, NULL) => NULL
  /// AND(false, throws) => throws
  /// AND(NULL, throws) => throws
  ///
  /// Returns false only if all inputs evaluate to false.
  ///
  /// Returns true if at least one input evaluates to true even if some inputs
  /// return null or throw.
  ///
  /// Propagates the exception if an input throws and no other input returns
  /// true.
  ///
  /// Returns NULL if at least one input is NULL and no other input evaluates to
  /// true or throws.
  ///
  /// Stops evaluating inputs once the result is determined (e.g. an input
  /// evaluates to true).
  ///
  /// Doesn't guarantee the order in which inputs are evaluated.
  kOr = 1,

  /// Converts value of one type into another. Requires exactly one input. The
  /// input type is the source (from) type. The output type is the target (to)
  /// type.
  ///
  /// Redundant casts are not allowed. The “from" type must be different from
  /// the “to" type.
  ///
  /// The exact semantics including the set of supported conversions are
  /// implementation specific and differ between applications.
  kCast = 2,

  /// Like CAST, but returns NULL if conversion fails. Requires exactly one
  /// input.
  ///
  /// Unlike TRY(CAST(x)), suppresses conversion failures, but propagates
  /// failures from evaluating cast input.
  ///
  /// Example:
  ///
  ///   cast(‘foo’ as integer) => Cannot cast 'foo' to INTEGER
  ///   try_cast(‘foo’ as integer) => NULL
  ///
  ///   cast(10 / 0 as double) => Division by zero
  ///   try_cast(10 / 0 as double) => Division by zero
  ///   try(cast(10 / 0 as double)) => NULL
  kTryCast = 3,

  /// Returns the value of the input expression or NULL if input throws.
  /// Requires exactly one input.
  ///
  /// Example:
  ///
  ///   10 / 0 => Division by zero
  ///   try(10 / 0) => NULL
  kTry = 4,

  /// Returns the value of the struct field. Requires two inputs: an expression
  /// that returns a struct (ROW type) and a constant name or index of the
  /// field. Index must be a zero-based integer.
  kDereference = 5,

  /// Returns the first non-null value or NULL if all inputs are NULL. Requires
  /// two or more arguments. All arguments must be of the same type. Doesn’t
  /// evaluate arguments past the first non-NULL.
  kCoalesce = 6,

  /// IF is a special case of SWITCH.
  ///
  /// IF(condition, then)
  /// IF(condition, then, else)
  ///
  /// Takes 2 or 3 inputs: boolean condition, “then" clause, optional “else"
  /// clause. Returns the results of evaluating the "then" clause if the
  /// condition evaluates to true. Otherwise, returns the results of evaluating
  /// the "else" clause or NULL if "else" clause is not specified.
  ///
  /// The types of “then" and “else" clauses must be the same and match the
  /// output type of IF.
  ///
  /// Evaluates “then" expression only on rows that pass “condition". Evaluates
  /// “else" expression only on rows that do not pass “condition".
  kIf = 7,

  /// case
  ///   when condition then result
  ///   [when ...]
  ///   [else result]
  /// end
  ///
  /// Takes two or more inputs: condition1, then1, condition2, then2,.. else.
  ///
  /// Condition inputs must be boolean expressions. Else clause is optional.
  /// Types of all the then clauses and the else clause must be the same and
  /// match the output of SWITCH.
  ///
  /// Evaluates each boolean condition from left to right until one is true and
  /// returns the result of evaluating the corresponding then clause. If no
  /// conditions are true, returns the result of evaluating the else clause or
  /// NULL if the else clause is not specified.
  kSwitch = 8,

  kStar = 9,

  // TODO Add IN and EXISTS.
};

VELOX_DECLARE_ENUM_NAME(SpecialForm)

/// SpecialFormExpr is similar to CallExpr, but has different semantics.
/// CallExpr represents a function call. It is executed by evaluating all
/// arguments first, then evaluating the function. Special form may have short
/// circuit behavior where not all arguments are evaluated all the time.
/// Special form may produce a result even if some arguments raised an error
/// during evaluation. Each special form has its own special semantic, hence,
/// they are special and are not represented using CallExpr.
class SpecialFormExpr : public Expr {
 public:
  SpecialFormExpr(
      const TypePtr& type,
      SpecialForm form,
      const std::vector<ExprPtr>& inputs);

  template <typename... T>
  SpecialFormExpr(const TypePtr& type, SpecialForm form, T... inputs)
      : SpecialFormExpr(
            type,
            form,
            std::vector<ExprPtr>{std::forward<ExprPtr>(inputs)...}) {}

  SpecialForm form() const {
    return form_;
  }

  void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const override;

 private:
  const SpecialForm form_;
};

using SpecialFormExprPtr = std::shared_ptr<const SpecialFormExpr>;

class SortOrder {
 public:
  static const SortOrder kAscNullsFirst;
  static const SortOrder kAscNullsLast;
  static const SortOrder kDescNullsFirst;
  static const SortOrder kDescNullsLast;

  SortOrder(bool ascending, bool nullsFirst)
      : ascending_(ascending), nullsFirst_(nullsFirst) {}

  bool isAscending() const {
    return ascending_;
  }

  bool isNullsFirst() const {
    return nullsFirst_;
  }

  bool operator==(const SortOrder& other) const {
    return std::tie(ascending_, nullsFirst_) ==
        std::tie(other.ascending_, other.nullsFirst_);
  }

  bool operator!=(const SortOrder& other) const {
    return !(*this == other);
  }

  std::string toString() const {
    return fmt::format(
        "{} NULLS {}",
        (ascending_ ? "ASC" : "DESC"),
        (nullsFirst_ ? "FIRST" : "LAST"));
  }

 private:
  bool ascending_;
  bool nullsFirst_;
};

struct SortingField {
  ExprPtr expression;
  SortOrder order;
};

/// Aggregate function call. To be used in AggregateNode.
///
/// Examples:
///
///   sum(x)
///   sum(x * y)
///   sum(x) filter (where x > 10)
///   array_agg(x order by y)
///   array_agg(distinct x)
///
class AggregateExpr : public Expr {
 public:
  /// @param type Return type.
  /// @param name Name of the aggregate function.
  /// @param inputs Zero or more inputs / arguments. These may contain arbitrary
  /// trees of scalar expressions. Cannot include AggregateExpr or WindowExpr.
  /// @param filter Optional predicate. If specified, only rows that satisfy
  /// the predicate are passed to the aggregate function.
  /// @param ordering Optional sorting order. If specified, the input rows are
  /// sorted before passing to the aggregate function.
  /// @param distinct Whether to deduplicate input rows before passing to the
  /// aggregate function.
  AggregateExpr(
      const TypePtr& type,
      const std::string& name,
      const std::vector<ExprPtr>& inputs,
      const ExprPtr& filter = nullptr,
      const std::vector<SortingField>& ordering = {},
      bool distinct = false)
      : Expr(ExprKind::kAggregate, type, inputs),
        name_{name},
        filter_{filter},
        ordering_{ordering},
        distinct_{distinct} {
    VELOX_USER_CHECK(!name.empty());

    if (filter != nullptr) {
      VELOX_USER_CHECK_EQ(filter->typeKind(), TypeKind::BOOLEAN);
    }

    if (distinct) {
      VELOX_USER_CHECK_GT(inputs.size(), 0);
    }
  }

  const std::string& name() const {
    return name_;
  }

  const ExprPtr& filter() const {
    return filter_;
  }

  const std::vector<SortingField>& ordering() const {
    return ordering_;
  }

  bool isDistinct() const {
    return distinct_;
  }

  void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const override;

 private:
  const std::string name_;
  const ExprPtr filter_;
  const std::vector<SortingField> ordering_;
  const bool distinct_;
};

using AggregateExprPtr = std::shared_ptr<const AggregateExpr>;

// Represents a window function call. Can be used in ProjectNode.
// TODO Adapt documentation from
// https://prestodb.io/docs/current/functions/window.html
class WindowExpr : public Expr {
 public:
  enum class WindowType { kRange, kRows, kGroups };
  VELOX_DECLARE_EMBEDDED_ENUM_NAME(WindowType)

  enum class BoundType {
    kUnboundedPreceding,
    kPreceding,
    kCurrentRow,
    kFollowing,
    kUnboundedFollowing
  };
  VELOX_DECLARE_EMBEDDED_ENUM_NAME(BoundType)

  /// A sliding window of rows to be processed by the function for a given
  /// input row. A frame can be ROWS type, RANGE type or GROUPS type, and it
  /// runs from start to end.
  struct Frame {
    WindowType type;
    BoundType startType;
    ExprPtr startValue;
    BoundType endType;
    ExprPtr endValue;
  };

  /// @param name Name of a window or aggregate function.
  /// @param partitionKeys Optional keys to break up the input rows into
  /// separate partitions, over which the window function is independently
  /// evaluated.
  /// @param ordering Determines the order within a partition in which input
  /// rows are processed by the window function.
  /// @param frame The window frame within the current partition that
  /// determines what to include in the window.
  WindowExpr(
      const TypePtr& type,
      const std::string& name,
      const std::vector<ExprPtr>& inputs,
      const std::vector<ExprPtr>& partitionKeys,
      const std::vector<SortingField>& ordering,
      const Frame& frame,
      bool ignoreNulls)
      : Expr(ExprKind::kWindow, type, inputs),
        name_{name},
        partitionKeys_{partitionKeys},
        ordering_{ordering},
        frame_{frame},
        ignoreNulls_{ignoreNulls} {}

  const std::string& name() const {
    return name_;
  }

  const std::vector<ExprPtr>& partitionKeys() const {
    return partitionKeys_;
  }

  const std::vector<SortingField>& ordering() const {
    return ordering_;
  }

  const Frame& frame() const {
    return frame_;
  }

  bool ignoreNulls() const {
    return ignoreNulls_;
  }

  void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const override;

 private:
  const std::string name_;
  const std::vector<ExprPtr> partitionKeys_;
  const std::vector<SortingField> ordering_;
  const Frame frame_;
  const bool ignoreNulls_;
};

using WindowExprPtr = std::shared_ptr<const WindowExpr>;

/// Lambda expression used an argument of a lambda function.
/// Example:
///
///   filter(array, x -> x > 10)
///
/// This expression is represented as a CallExpr("filter",...) with 2 inputs:
/// an expression that produces an array and a LambdaExpr that represents x ->
/// x > 10.
///
/// Lambda expression consists of two parts: signature and body. Signature is
/// a list of names and types of the arguments (x). Body is the expression: x
/// > 10.
///
/// Lambda body may reference columns that are not part of the signature.
/// These are called captures. Any columns visible to enclosing CallExpr are
/// also visible to LambdaExpr.
class LambdaExpr : public Expr {
 public:
  LambdaExpr(const RowTypePtr& signature, const ExprPtr& body)
      : Expr(
            ExprKind::kLambda,
            std::make_shared<FunctionType>(
                std::vector<TypePtr>(signature->children()),
                body->type()),
            {}),
        signature_(signature),
        body_(body) {
    VELOX_USER_CHECK_GT(signature->size(), 0);
  }

  const RowTypePtr& signature() const {
    return signature_;
  }

  const ExprPtr& body() const {
    return body_;
  }

  void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const override;

 private:
  const RowTypePtr signature_;
  const ExprPtr body_;
};

using LambdaExprPtr = std::shared_ptr<const LambdaExpr>;

class LogicalPlanNode;
using LogicalPlanNodePtr = std::shared_ptr<const LogicalPlanNode>;

/// Scalar subquery that returns exactly one row and one column. Can be used
/// anywhere a scalar function call can be used.
class SubqueryExpr : public Expr {
 public:
  explicit SubqueryExpr(const LogicalPlanNodePtr& subquery);

  const LogicalPlanNodePtr& subquery() const {
    return subquery_;
  }

  void accept(const ExprVisitor& visitor, ExprVisitorContext& context)
      const override;

 private:
  const LogicalPlanNodePtr subquery_;
};

using SubqueryExprPtr = std::shared_ptr<const SubqueryExpr>;

} // namespace facebook::velox::logical_plan
