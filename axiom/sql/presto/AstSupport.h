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
class Identifier;
class QualifiedName;

// Support Classes
class Property : public Node {
public:
  Property(std::shared_ptr<Identifier> name, ExpressionPtr value)
      : Node(NodeType::kProperty), name_(name), value_(value) {}
  
  const std::shared_ptr<Identifier>& getName() const { return name_; }
  const ExpressionPtr& getValue() const { return value_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<Identifier> name_;
  ExpressionPtr value_;
};

class CallArgument : public Node {
public:
  CallArgument(std::optional<std::shared_ptr<Identifier>> name, ExpressionPtr value)
      : Node(NodeType::kCallArgument), name_(name), value_(value) {}
  
  const std::optional<std::shared_ptr<Identifier>>& getName() const { return name_; }
  const ExpressionPtr& getValue() const { return value_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::optional<std::shared_ptr<Identifier>> name_;
  ExpressionPtr value_;
};

// Window Classes
class FrameBound : public Node {
public:
  enum class Type { 
    UNBOUNDED_PRECEDING, PRECEDING, CURRENT_ROW, FOLLOWING, UNBOUNDED_FOLLOWING 
  };
  
  FrameBound(Type type, std::optional<ExpressionPtr> value = std::nullopt)
      : Node(NodeType::kFrameBound), type_(type), value_(value) {}
  
  Type getType() const { return type_; }
  const std::optional<ExpressionPtr>& getValue() const { return value_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Type type_;
  std::optional<ExpressionPtr> value_;
};

class WindowFrame : public Node {
public:
  enum class Type { RANGE, ROWS, GROUPS };
  
  WindowFrame(
      Type type,
      std::shared_ptr<FrameBound> start,
      std::optional<std::shared_ptr<FrameBound>> end = std::nullopt)
      : Node(NodeType::kWindowFrame), type_(type), start_(start), end_(end) {}
  
  Type getType() const { return type_; }
  const std::shared_ptr<FrameBound>& getStart() const { return start_; }
  const std::optional<std::shared_ptr<FrameBound>>& getEnd() const { return end_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Type type_;
  std::shared_ptr<FrameBound> start_;
  std::optional<std::shared_ptr<FrameBound>> end_;
};

class Window : public Node {
public:
  Window(
      const std::vector<ExpressionPtr>& partitionBy,
      std::optional<std::shared_ptr<OrderBy>> orderBy = std::nullopt,
      std::optional<std::shared_ptr<WindowFrame>> frame = std::nullopt)
      : Node(NodeType::kWindow), partitionBy_(partitionBy), orderBy_(orderBy), frame_(frame) {}
  
  const std::vector<ExpressionPtr>& getPartitionBy() const { return partitionBy_; }
  const std::optional<std::shared_ptr<OrderBy>>& getOrderBy() const { return orderBy_; }
  const std::optional<std::shared_ptr<WindowFrame>>& getFrame() const { return frame_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> partitionBy_;
  std::optional<std::shared_ptr<OrderBy>> orderBy_;
  std::optional<std::shared_ptr<WindowFrame>> frame_;
};

// Sorting and Grouping
class SortItem : public Node {
public:
  enum class Ordering { ASCENDING, DESCENDING };
  enum class NullOrdering { FIRST, LAST, UNDEFINED };
  
  SortItem(
      ExpressionPtr sortKey,
      Ordering ordering,
      NullOrdering nullOrdering)
      : Node(NodeType::kSortItem), sortKey_(sortKey), ordering_(ordering), nullOrdering_(nullOrdering) {}
  
  const ExpressionPtr& getSortKey() const { return sortKey_; }
  Ordering getOrdering() const { return ordering_; }
  NullOrdering getNullOrdering() const { return nullOrdering_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr sortKey_;
  Ordering ordering_;
  NullOrdering nullOrdering_;
};

class OrderBy : public Node {
public:
  explicit OrderBy(const std::vector<std::shared_ptr<SortItem>>& sortItems)
      : Node(NodeType::kOrderBy), sortItems_(sortItems) {}
  
  const std::vector<std::shared_ptr<SortItem>>& getSortItems() const { return sortItems_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<std::shared_ptr<SortItem>> sortItems_;
};

class Offset : public Node {
public:
  explicit Offset(const std::string& offset)
      : Node(NodeType::kOffset), offset_(offset) {}
  
  const std::string& getOffset() const { return offset_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string offset_;
};

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
  
  const std::vector<ExpressionPtr>& getExpressions() const { return expressions_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> expressions_;
};

class GroupingSets : public GroupingElement {
public:
  explicit GroupingSets(const std::vector<std::vector<ExpressionPtr>>& sets)
      : GroupingElement(NodeType::kGroupingSets), sets_(sets) {}
  
  const std::vector<std::vector<ExpressionPtr>>& getSets() const { return sets_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<std::vector<ExpressionPtr>> sets_;
};

class Cube : public GroupingElement {
public:
  explicit Cube(const std::vector<ExpressionPtr>& expressions)
      : GroupingElement(NodeType::kCube), expressions_(expressions) {}
  
  const std::vector<ExpressionPtr>& getExpressions() const { return expressions_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> expressions_;
};

class Rollup : public GroupingElement {
public:
  explicit Rollup(const std::vector<ExpressionPtr>& expressions)
      : GroupingElement(NodeType::kRollup), expressions_(expressions) {}
  
  const std::vector<ExpressionPtr>& getExpressions() const { return expressions_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> expressions_;
};

class GroupBy : public Node {
public:
  GroupBy(bool distinct, const std::vector<GroupingElementPtr>& groupingElements)
      : Node(NodeType::kGroupBy), distinct_(distinct), groupingElements_(groupingElements) {}
  
  bool isDistinct() const { return distinct_; }
  const std::vector<GroupingElementPtr>& getGroupingElements() const { return groupingElements_; }
  
  void accept(AstVisitor* visitor) override;

private:
  bool distinct_;
  std::vector<GroupingElementPtr> groupingElements_;
};

// WITH clause
class WithQuery : public Node {
public:
  WithQuery(std::shared_ptr<Identifier> name, StatementPtr query, std::optional<std::vector<std::shared_ptr<Identifier>>> columnNames = std::nullopt)
      : Node(NodeType::kWithQuery), name_(name), query_(query), columnNames_(columnNames) {}
  
  const std::shared_ptr<Identifier>& getName() const { return name_; }
  const StatementPtr& getQuery() const { return query_; }
  const std::optional<std::vector<std::shared_ptr<Identifier>>>& getColumnNames() const { return columnNames_; }
  
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
  
  bool isRecursive() const { return recursive_; }
  const std::vector<std::shared_ptr<WithQuery>>& getQueries() const { return queries_; }
  
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
      std::shared_ptr<Identifier> name,
      const std::string& type,
      bool nullable,
      const std::vector<std::shared_ptr<Property>>& properties,
      std::optional<std::string> comment = std::nullopt)
      : TableElement(NodeType::kColumnDefinition),
        name_(name), type_(type), nullable_(nullable), properties_(properties), comment_(comment) {}
  
  const std::shared_ptr<Identifier>& getName() const { return name_; }
  const std::string& getType() const { return type_; }
  bool isNullable() const { return nullable_; }
  const std::vector<std::shared_ptr<Property>>& getProperties() const { return properties_; }
  const std::optional<std::string>& getComment() const { return comment_; }
  
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
  enum class PropertiesOption { INCLUDING, EXCLUDING };
  
  LikeClause(std::shared_ptr<QualifiedName> tableName, std::optional<PropertiesOption> propertiesOption = std::nullopt)
      : TableElement(NodeType::kLikeClause), tableName_(tableName), propertiesOption_(propertiesOption) {}
  
  const std::shared_ptr<QualifiedName>& getTableName() const { return tableName_; }
  const std::optional<PropertiesOption>& getPropertiesOption() const { return propertiesOption_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> tableName_;
  std::optional<PropertiesOption> propertiesOption_;
};

class ConstraintSpecification : public TableElement {
public:
  enum class ConstraintType { UNIQUE, PRIMARY_KEY };
  
  ConstraintSpecification(
      std::optional<std::shared_ptr<Identifier>> name,
      const std::vector<std::shared_ptr<Identifier>>& columns,
      ConstraintType type)
      : TableElement(NodeType::kConstraintSpecification), name_(name), columns_(columns), type_(type) {}
  
  const std::optional<std::shared_ptr<Identifier>>& getName() const { return name_; }
  const std::vector<std::shared_ptr<Identifier>>& getColumns() const { return columns_; }
  ConstraintType getConstraintType() const { return type_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::optional<std::shared_ptr<Identifier>> name_;
  std::vector<std::shared_ptr<Identifier>> columns_;
  ConstraintType type_;
};

// Security and Transaction Support
class PrincipalSpecification : public Node {
public:
  enum class Type { UNSPECIFIED, USER, ROLE };
  
  PrincipalSpecification(Type type, std::shared_ptr<Identifier> name)
      : Node(NodeType::kPrincipalSpecification), type_(type), name_(name) {}
  
  Type getType() const { return type_; }
  const std::shared_ptr<Identifier>& getName() const { return name_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Type type_;
  std::shared_ptr<Identifier> name_;
};

class GrantorSpecification : public Node {
public:
  enum class Type { PRINCIPAL, CURRENT_USER, CURRENT_ROLE };
  
  GrantorSpecification(Type type, std::optional<std::shared_ptr<PrincipalSpecification>> principal = std::nullopt)
      : Node(NodeType::kGrantorSpecification), type_(type), principal_(principal) {}
  
  Type getType() const { return type_; }
  const std::optional<std::shared_ptr<PrincipalSpecification>>& getPrincipal() const { return principal_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Type type_;
  std::optional<std::shared_ptr<PrincipalSpecification>> principal_;
};

class Isolation : public Node {
public:
  enum class Level { 
    SERIALIZABLE, REPEATABLE_READ, READ_COMMITTED, READ_UNCOMMITTED 
  };
  
  explicit Isolation(Level level)
      : Node(NodeType::kIsolation), level_(level) {}
  
  Level getLevel() const { return level_; }
  void accept(AstVisitor* visitor) override;

private:
  Level level_;
};

class TransactionAccessMode : public Node {
public:
  enum class Type { READ_ONLY, READ_WRITE };
  
  explicit TransactionAccessMode(Type type)
      : Node(NodeType::kTransactionAccessMode), type_(type) {}
  
  Type getType() const { return type_; }
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
  SqlParameterDeclaration(std::shared_ptr<Identifier> name, const std::string& type)
      : Node(NodeType::kSqlParameterDeclaration), name_(name), type_(type) {}
  
  const std::shared_ptr<Identifier>& getName() const { return name_; }
  const std::string& getType() const { return type_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<Identifier> name_;
  std::string type_;
};

class RoutineCharacteristics : public Node {
public:
  enum class Determinism { DETERMINISTIC, NOT_DETERMINISTIC };
  enum class NullCallClause { RETURNS_NULL_ON_NULL_INPUT, CALLED_ON_NULL_INPUT };
  
  RoutineCharacteristics(
      std::optional<Determinism> determinism = std::nullopt,
      std::optional<NullCallClause> nullCallClause = std::nullopt)
      : Node(NodeType::kRoutineCharacteristics), determinism_(determinism), nullCallClause_(nullCallClause) {}
  
  const std::optional<Determinism>& getDeterminism() const { return determinism_; }
  const std::optional<NullCallClause>& getNullCallClause() const { return nullCallClause_; }
  
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
      : RoutineBody(NodeType::kExternalBodyReference), externalName_(externalName) {}
  
  const std::string& getExternalName() const { return externalName_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string externalName_;
};

class Return : public RoutineBody {
public:
  explicit Return(ExpressionPtr expression)
      : RoutineBody(NodeType::kReturn), expression_(expression) {}
  
  const ExpressionPtr& getExpression() const { return expression_; }
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
  enum class Type { TEXT, GRAPHVIZ, JSON };
  
  explicit ExplainFormat(Type type)
      : ExplainOption(NodeType::kExplainFormat), type_(type) {}
  
  Type getType() const { return type_; }
  void accept(AstVisitor* visitor) override;

private:
  Type type_;
};

class ExplainType : public ExplainOption {
public:
  enum class Type { LOGICAL, DISTRIBUTED, VALIDATE, IO };
  
  explicit ExplainType(Type type)
      : ExplainOption(NodeType::kExplainType), type_(type) {}
  
  Type getType() const { return type_; }
  void accept(AstVisitor* visitor) override;

private:
  Type type_;
};

} // namespace facebook::velox::sql