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

#include <optional>
#include <vector>
#include "axiom/sql/presto/ast/AstNode.h"

namespace axiom::sql::presto {

// Forward declarations
class Identifier;
class QualifiedName;

// Support Classes
class Property : public Node {
 public:
  Property(const std::shared_ptr<Identifier>& name, const ExpressionPtr& value)
      : Node(NodeType::kProperty), name_(name), value_(value) {}

  const std::shared_ptr<Identifier>& name() const {
    return name_;
  }

  const ExpressionPtr& value() const {
    return value_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::shared_ptr<Identifier> name_;
  ExpressionPtr value_;
};

class CallArgument : public Node {
 public:
  CallArgument(
      const std::shared_ptr<Identifier>& name,
      const ExpressionPtr& value)
      : Node(NodeType::kCallArgument), name_(name), value_(value) {}

  const std::shared_ptr<Identifier>& name() const {
    return name_;
  }

  const ExpressionPtr& value() const {
    return value_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::shared_ptr<Identifier> name_;
  ExpressionPtr value_;
};

// Window Classes
class FrameBound : public Node {
 public:
  enum class Type {
    kUnboundedPreceding,
    kPreceding,
    kCurrentRow,
    kFollowing,
    kUnboundedFollowing
  };

  FrameBound(Type type, std::optional<ExpressionPtr> value = std::nullopt)
      : Node(NodeType::kFrameBound), type_(type), value_(value) {}

  Type type() const {
    return type_;
  }

  const std::optional<ExpressionPtr>& value() const {
    return value_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  Type type_;
  std::optional<ExpressionPtr> value_;
};

class WindowFrame : public Node {
 public:
  enum class Type { kRange, kRows, kGroups };

  WindowFrame(
      Type type,
      const std::shared_ptr<FrameBound>& start,
      const std::shared_ptr<FrameBound>& end = nullptr)
      : Node(NodeType::kWindowFrame), type_(type), start_(start), end_(end) {}

  Type type() const {
    return type_;
  }

  const std::shared_ptr<FrameBound>& start() const {
    return start_;
  }

  const std::shared_ptr<FrameBound>& end() const {
    return end_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  Type type_;
  std::shared_ptr<FrameBound> start_;
  std::shared_ptr<FrameBound> end_;
};

class Window : public Node {
 public:
  Window(
      const std::vector<ExpressionPtr>& partitionBy,
      const std::shared_ptr<OrderBy>& orderBy = nullptr,
      const std::shared_ptr<WindowFrame>& frame = nullptr)
      : Node(NodeType::kWindow),
        partitionBy_(partitionBy),
        orderBy_(orderBy),
        frame_(frame) {}

  const std::vector<ExpressionPtr>& partitionBy() const {
    return partitionBy_;
  }

  const std::shared_ptr<OrderBy>& orderBy() const {
    return orderBy_;
  }

  const std::shared_ptr<WindowFrame>& frame() const {
    return frame_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::vector<ExpressionPtr> partitionBy_;
  std::shared_ptr<OrderBy> orderBy_;
  std::shared_ptr<WindowFrame> frame_;
};

// Sorting and Grouping
class SortItem : public Node {
 public:
  enum class Ordering { kAscending, kDescending };
  enum class NullOrdering { kFirst, kLast, kUndefined };

  SortItem(
      NodeLocation location,
      ExpressionPtr sortKey,
      Ordering ordering,
      NullOrdering nullOrdering)
      : Node(NodeType::kSortItem, location),
        sortKey_(sortKey),
        ordering_(ordering),
        nullOrdering_(nullOrdering) {}

  const ExpressionPtr& sortKey() const {
    return sortKey_;
  }

  Ordering ordering() const {
    return ordering_;
  }

  bool isAscending() const {
    return ordering_ == Ordering::kAscending;
  }

  NullOrdering nullOrdering() const {
    return nullOrdering_;
  }

  bool isNullsFirst() const {
    return nullOrdering_ == NullOrdering::kFirst;
  }

  void accept(AstVisitor* visitor) override;

 private:
  ExpressionPtr sortKey_;
  Ordering ordering_;
  NullOrdering nullOrdering_;
};

class OrderBy : public Node {
 public:
  explicit OrderBy(
      NodeLocation location,
      const std::vector<std::shared_ptr<SortItem>>& sortItems)
      : Node(NodeType::kOrderBy, location), sortItems_(sortItems) {}

  const std::vector<std::shared_ptr<SortItem>>& sortItems() const {
    return sortItems_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  std::vector<std::shared_ptr<SortItem>> sortItems_;
};

using OrderByPtr = std::shared_ptr<OrderBy>;

class Offset : public Node {
 public:
  explicit Offset(const std::string& offset)
      : Node(NodeType::kOffset), offset_(offset) {}

  const std::string& offset() const {
    return offset_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  std::string offset_;
};

using OffsetPtr = std::shared_ptr<Offset>;

// Grouping Elements
class GroupingElement : public Node {
 public:
  explicit GroupingElement(NodeType type) : Node(type) {}
};

using GroupingElementPtr = std::shared_ptr<GroupingElement>;

class SimpleGroupBy : public GroupingElement {
 public:
  explicit SimpleGroupBy(const std::vector<ExpressionPtr>& expressions)
      : GroupingElement(NodeType::kSimpleGroupBy), expressions_(expressions) {}

  const std::vector<ExpressionPtr>& expressions() const {
    return expressions_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  std::vector<ExpressionPtr> expressions_;
};

class GroupingSets : public GroupingElement {
 public:
  explicit GroupingSets(const std::vector<std::vector<ExpressionPtr>>& sets)
      : GroupingElement(NodeType::kGroupingSets), sets_(sets) {}

  const std::vector<std::vector<ExpressionPtr>>& sets() const {
    return sets_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  std::vector<std::vector<ExpressionPtr>> sets_;
};

class Cube : public GroupingElement {
 public:
  explicit Cube(const std::vector<ExpressionPtr>& expressions)
      : GroupingElement(NodeType::kCube), expressions_(expressions) {}

  const std::vector<ExpressionPtr>& expressions() const {
    return expressions_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  std::vector<ExpressionPtr> expressions_;
};

class Rollup : public GroupingElement {
 public:
  explicit Rollup(const std::vector<ExpressionPtr>& expressions)
      : GroupingElement(NodeType::kRollup), expressions_(expressions) {}

  const std::vector<ExpressionPtr>& expressions() const {
    return expressions_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  std::vector<ExpressionPtr> expressions_;
};

class GroupBy : public Node {
 public:
  GroupBy(
      bool distinct,
      const std::vector<GroupingElementPtr>& groupingElements)
      : Node(NodeType::kGroupBy),
        distinct_(distinct),
        groupingElements_(groupingElements) {}

  bool isDistinct() const {
    return distinct_;
  }
  const std::vector<GroupingElementPtr>& groupingElements() const {
    return groupingElements_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  bool distinct_;
  std::vector<GroupingElementPtr> groupingElements_;
};

using GroupByPtr = std::shared_ptr<GroupBy>;

// WITH clause
class WithQuery : public Node {
 public:
  WithQuery(
      const std::shared_ptr<Identifier>& name,
      const StatementPtr& query,
      std::optional<std::vector<std::shared_ptr<Identifier>>> columnNames =
          std::nullopt)
      : Node(NodeType::kWithQuery),
        name_(name),
        query_(query),
        columnNames_(columnNames) {}

  const std::shared_ptr<Identifier>& name() const {
    return name_;
  }

  const StatementPtr& query() const {
    return query_;
  }

  const std::optional<std::vector<std::shared_ptr<Identifier>>>& columnNames()
      const {
    return columnNames_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::shared_ptr<Identifier> name_;
  StatementPtr query_;
  std::optional<std::vector<std::shared_ptr<Identifier>>> columnNames_;
};

class With : public Node {
 public:
  With(bool recursive, const std::vector<std::shared_ptr<WithQuery>>& queries)
      : Node(NodeType::kWith), recursive_(recursive), queries_(queries) {}

  bool isRecursive() const {
    return recursive_;
  }
  const std::vector<std::shared_ptr<WithQuery>>& queries() const {
    return queries_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  bool recursive_;
  std::vector<std::shared_ptr<WithQuery>> queries_;
};

// Table Elements
class TableElement : public Node {
 public:
  explicit TableElement(NodeType type) : Node(type) {}
};

using TableElementPtr = std::shared_ptr<TableElement>;

class ColumnDefinition : public TableElement {
 public:
  ColumnDefinition(
      const std::shared_ptr<Identifier>& name,
      const std::string& type,
      bool nullable,
      const std::vector<std::shared_ptr<Property>>& properties,
      std::optional<std::string> comment = std::nullopt)
      : TableElement(NodeType::kColumnDefinition),
        name_(name),
        type_(type),
        nullable_(nullable),
        properties_(properties),
        comment_(comment) {}

  const std::shared_ptr<Identifier>& name() const {
    return name_;
  }

  const std::string& type() const {
    return type_;
  }

  bool isNullable() const {
    return nullable_;
  }

  const std::vector<std::shared_ptr<Property>>& properties() const {
    return properties_;
  }

  const std::optional<std::string>& comment() const {
    return comment_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::shared_ptr<Identifier> name_;
  std::string type_;
  bool nullable_;
  std::vector<std::shared_ptr<Property>> properties_;
  std::optional<std::string> comment_;
};

class LikeClause : public TableElement {
 public:
  enum class PropertiesOption { kIncluding, kExcluding };

  LikeClause(
      const std::shared_ptr<QualifiedName>& tableName,
      std::optional<PropertiesOption> propertiesOption = std::nullopt)
      : TableElement(NodeType::kLikeClause),
        tableName_(tableName),
        propertiesOption_(propertiesOption) {}

  const std::shared_ptr<QualifiedName>& tableName() const {
    return tableName_;
  }

  const std::optional<PropertiesOption>& propertiesOption() const {
    return propertiesOption_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::shared_ptr<QualifiedName> tableName_;
  std::optional<PropertiesOption> propertiesOption_;
};

class ConstraintSpecification : public TableElement {
 public:
  enum class ConstraintType { kUnique, kPrimaryKey };

  ConstraintSpecification(
      const std::shared_ptr<Identifier>& name,
      const std::vector<std::shared_ptr<Identifier>>& columns,
      ConstraintType type)
      : TableElement(NodeType::kConstraintSpecification),
        name_(name),
        columns_(columns),
        type_(type) {}

  const std::shared_ptr<Identifier>& name() const {
    return name_;
  }

  const std::vector<std::shared_ptr<Identifier>>& columns() const {
    return columns_;
  }

  ConstraintType constraintType() const {
    return type_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::shared_ptr<Identifier> name_;
  std::vector<std::shared_ptr<Identifier>> columns_;
  ConstraintType type_;
};

// Security and Transaction Support
class PrincipalSpecification : public Node {
 public:
  enum class Type { kUnspecified, kUser, kRole };

  PrincipalSpecification(Type type, const std::shared_ptr<Identifier>& name)
      : Node(NodeType::kPrincipalSpecification), type_(type), name_(name) {}

  Type type() const {
    return type_;
  }

  const std::shared_ptr<Identifier>& name() const {
    return name_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  Type type_;
  std::shared_ptr<Identifier> name_;
};

class GrantorSpecification : public Node {
 public:
  enum class Type { kPrincipal, kCurrentUser, kCurrentRole };

  GrantorSpecification(
      Type type,
      const std::shared_ptr<PrincipalSpecification>& principal = nullptr)
      : Node(NodeType::kGrantorSpecification),
        type_(type),
        principal_(principal) {}

  Type type() const {
    return type_;
  }

  const std::shared_ptr<PrincipalSpecification>& principal() const {
    return principal_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  Type type_;
  std::shared_ptr<PrincipalSpecification> principal_;
};

class Isolation : public Node {
 public:
  enum class Level {
    kSerializable,
    kRepeatableRead,
    kReadCommitted,
    kReadUncommitted
  };

  explicit Isolation(Level level) : Node(NodeType::kIsolation), level_(level) {}

  Level level() const {
    return level_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  Level level_;
};

class TransactionAccessMode : public Node {
 public:
  enum class Type { kReadOnly, kReadWrite };

  explicit TransactionAccessMode(Type type)
      : Node(NodeType::kTransactionAccessMode), type_(type) {}

  Type type() const {
    return type_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  Type type_;
};

class TransactionMode : public Node {
 public:
  explicit TransactionMode(NodeType type) : Node(type) {}
};

using TransactionModePtr = std::shared_ptr<TransactionMode>;

// Function and Routine Support
class SqlParameterDeclaration : public Node {
 public:
  SqlParameterDeclaration(
      const std::shared_ptr<Identifier>& name,
      const std::string& type)
      : Node(NodeType::kSqlParameterDeclaration), name_(name), type_(type) {}

  const std::shared_ptr<Identifier>& name() const {
    return name_;
  }

  const std::string& type() const {
    return type_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::shared_ptr<Identifier> name_;
  std::string type_;
};

class RoutineCharacteristics : public Node {
 public:
  enum class Determinism { kDeterministic, kNotDeterministic };
  enum class NullCallClause { kReturnsNullOnNullInput, kCalledOnNullInput };

  RoutineCharacteristics(
      std::optional<Determinism> determinism = std::nullopt,
      std::optional<NullCallClause> nullCallClause = std::nullopt)
      : Node(NodeType::kRoutineCharacteristics),
        determinism_(determinism),
        nullCallClause_(nullCallClause) {}

  const std::optional<Determinism>& determinism() const {
    return determinism_;
  }

  const std::optional<NullCallClause>& nullCallClause() const {
    return nullCallClause_;
  }

  void accept(AstVisitor* visitor) override;

 private:
  std::optional<Determinism> determinism_;
  std::optional<NullCallClause> nullCallClause_;
};

class RoutineBody : public Node {
 public:
  explicit RoutineBody(NodeType type) : Node(type) {}
};

using RoutineBodyPtr = std::shared_ptr<RoutineBody>;

class ExternalBodyReference : public RoutineBody {
 public:
  explicit ExternalBodyReference(const std::string& externalName)
      : RoutineBody(NodeType::kExternalBodyReference),
        externalName_(externalName) {}

  const std::string& externalName() const {
    return externalName_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  std::string externalName_;
};

class Return : public RoutineBody {
 public:
  explicit Return(ExpressionPtr expression)
      : RoutineBody(NodeType::kReturn), expression_(expression) {}

  const ExpressionPtr& expression() const {
    return expression_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  ExpressionPtr expression_;
};

// Explain Support
class ExplainOption : public Node {
 public:
  explicit ExplainOption(NodeType type) : Node(type) {}
};

using ExplainOptionPtr = std::shared_ptr<ExplainOption>;

class ExplainFormat : public ExplainOption {
 public:
  enum class Type { kText, kGraphviz, kJson };

  explicit ExplainFormat(Type type)
      : ExplainOption(NodeType::kExplainFormat), type_(type) {}

  Type type() const {
    return type_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  Type type_;
};

class ExplainType : public ExplainOption {
 public:
  enum class Type { kLogical, kDistributed, kValidate, kIo };

  explicit ExplainType(Type type)
      : ExplainOption(NodeType::kExplainType), type_(type) {}

  Type type() const {
    return type_;
  }
  void accept(AstVisitor* visitor) override;

 private:
  Type type_;
};

} // namespace axiom::sql::presto
