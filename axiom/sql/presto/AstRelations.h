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
class Identifier;
class Expression;
class Select;
class OrderBy;
class GroupBy;
class With;
class Window;
class WindowFrame;
class FrameBound;
class SortItem;
class GroupingElement;
class Offset;
class TableVersionExpression;

// Query Body base class
class QueryBody : public Relation {
public:
  explicit QueryBody(NodeType type) : Relation(type) {}
};

using QueryBodyPtr = std::shared_ptr<QueryBody>;

// Query Structure Classes
class Query : public Statement {
public:
  Query(
      std::optional<std::shared_ptr<With>> with,
      QueryBodyPtr queryBody,
      std::optional<std::shared_ptr<OrderBy>> orderBy = std::nullopt,
      std::optional<std::shared_ptr<Offset>> offset = std::nullopt,
      std::optional<std::string> limit = std::nullopt)
      : Statement(NodeType::kQuery),
        with_(with), queryBody_(queryBody), orderBy_(orderBy), offset_(offset), limit_(limit) {}
  
  const std::optional<std::shared_ptr<With>>& getWith() const { return with_; }
  const QueryBodyPtr& getQueryBody() const { return queryBody_; }
  const std::optional<std::shared_ptr<OrderBy>>& getOrderBy() const { return orderBy_; }
  const std::optional<std::shared_ptr<Offset>>& getOffset() const { return offset_; }
  const std::optional<std::string>& getLimit() const { return limit_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::optional<std::shared_ptr<With>> with_;
  QueryBodyPtr queryBody_;
  std::optional<std::shared_ptr<OrderBy>> orderBy_;
  std::optional<std::shared_ptr<Offset>> offset_;
  std::optional<std::string> limit_;
};

class QuerySpecification : public QueryBody {
public:
  QuerySpecification(
      std::shared_ptr<Select> select,
      std::optional<RelationPtr> from = std::nullopt,
      std::optional<ExpressionPtr> where = std::nullopt,
      std::optional<std::shared_ptr<GroupBy>> groupBy = std::nullopt,
      std::optional<ExpressionPtr> having = std::nullopt,
      std::optional<std::shared_ptr<Window>> window = std::nullopt)
      : QueryBody(NodeType::kQuerySpecification),
        select_(select), from_(from), where_(where), groupBy_(groupBy), having_(having), window_(window) {}
  
  const std::shared_ptr<Select>& getSelect() const { return select_; }
  const std::optional<RelationPtr>& getFrom() const { return from_; }
  const std::optional<ExpressionPtr>& getWhere() const { return where_; }
  const std::optional<std::shared_ptr<GroupBy>>& getGroupBy() const { return groupBy_; }
  const std::optional<ExpressionPtr>& getHaving() const { return having_; }
  const std::optional<std::shared_ptr<Window>>& getWindow() const { return window_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<Select> select_;
  std::optional<RelationPtr> from_;
  std::optional<ExpressionPtr> where_;
  std::optional<std::shared_ptr<GroupBy>> groupBy_;
  std::optional<ExpressionPtr> having_;
  std::optional<std::shared_ptr<Window>> window_;
};

// SELECT items
class SelectItem : public Node {
public:
  explicit SelectItem(NodeType type) : Node(type) {}
};

using SelectItemPtr = std::shared_ptr<SelectItem>;

class SingleColumn : public SelectItem {
public:
  SingleColumn(ExpressionPtr expression, std::optional<std::shared_ptr<Identifier>> alias = std::nullopt)
      : SelectItem(NodeType::kSingleColumn), expression_(expression), alias_(alias) {}
  
  const ExpressionPtr& getExpression() const { return expression_; }
  const std::optional<std::shared_ptr<Identifier>>& getAlias() const { return alias_; }
  
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr expression_;
  std::optional<std::shared_ptr<Identifier>> alias_;
};

class AllColumns : public SelectItem {
public:
  explicit AllColumns(std::optional<std::shared_ptr<QualifiedName>> prefix = std::nullopt)
      : SelectItem(NodeType::kAllColumns), prefix_(prefix) {}
  
  const std::optional<std::shared_ptr<QualifiedName>>& getPrefix() const { return prefix_; }
  void accept(AstVisitor* visitor) override;

private:
  std::optional<std::shared_ptr<QualifiedName>> prefix_;
};

class Select : public Node {
public:
  Select(bool distinct, const std::vector<SelectItemPtr>& selectItems)
      : Node(NodeType::kSelect), distinct_(distinct), selectItems_(selectItems) {}
  
  bool isDistinct() const { return distinct_; }
  const std::vector<SelectItemPtr>& getSelectItems() const { return selectItems_; }
  
  void accept(AstVisitor* visitor) override;

private:
  bool distinct_;
  std::vector<SelectItemPtr> selectItems_;
};

// Table Relations
class Table : public QueryBody {
public:
  Table(std::shared_ptr<QualifiedName> name, std::optional<std::shared_ptr<TableVersionExpression>> version = std::nullopt)
      : QueryBody(NodeType::kTable), name_(name), version_(version) {}
  
  const std::shared_ptr<QualifiedName>& getName() const { return name_; }
  const std::optional<std::shared_ptr<TableVersionExpression>>& getVersion() const { return version_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> name_;
  std::optional<std::shared_ptr<TableVersionExpression>> version_;
};

class AliasedRelation : public Relation {
public:
  AliasedRelation(
      RelationPtr relation,
      std::shared_ptr<Identifier> alias,
      std::optional<std::vector<std::shared_ptr<Identifier>>> columnNames = std::nullopt)
      : Relation(NodeType::kAliasedRelation), relation_(relation), alias_(alias), columnNames_(columnNames) {}
  
  const RelationPtr& getRelation() const { return relation_; }
  const std::shared_ptr<Identifier>& getAlias() const { return alias_; }
  const std::optional<std::vector<std::shared_ptr<Identifier>>>& getColumnNames() const { return columnNames_; }
  
  void accept(AstVisitor* visitor) override;

private:
  RelationPtr relation_;
  std::shared_ptr<Identifier> alias_;
  std::optional<std::vector<std::shared_ptr<Identifier>>> columnNames_;
};

class SampledRelation : public Relation {
public:
  enum class Type { BERNOULLI, SYSTEM };
  
  SampledRelation(RelationPtr relation, Type type, ExpressionPtr samplePercentage)
      : Relation(NodeType::kSampledRelation), relation_(relation), type_(type), samplePercentage_(samplePercentage) {}
  
  const RelationPtr& getRelation() const { return relation_; }
  Type getType() const { return type_; }
  const ExpressionPtr& getSamplePercentage() const { return samplePercentage_; }
  
  void accept(AstVisitor* visitor) override;

private:
  RelationPtr relation_;
  Type type_;
  ExpressionPtr samplePercentage_;
};

class TableSubquery : public QueryBody {
public:
  explicit TableSubquery(StatementPtr query)
      : QueryBody(NodeType::kTableSubquery), query_(query) {}
  
  const StatementPtr& getQuery() const { return query_; }
  void accept(AstVisitor* visitor) override;

private:
  StatementPtr query_;
};

class Lateral : public Relation {
public:
  explicit Lateral(RelationPtr query)
      : Relation(NodeType::kLateral), query_(query) {}
  
  const RelationPtr& getQuery() const { return query_; }
  void accept(AstVisitor* visitor) override;

private:
  RelationPtr query_;
};

class Unnest : public Relation {
public:
  Unnest(const std::vector<ExpressionPtr>& expressions, bool withOrdinality)
      : Relation(NodeType::kUnnest), expressions_(expressions), withOrdinality_(withOrdinality) {}
  
  const std::vector<ExpressionPtr>& getExpressions() const { return expressions_; }
  bool isWithOrdinality() const { return withOrdinality_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> expressions_;
  bool withOrdinality_;
};

class Values : public QueryBody {
public:
  explicit Values(const std::vector<ExpressionPtr>& rows)
      : QueryBody(NodeType::kValues), rows_(rows) {}
  
  const std::vector<ExpressionPtr>& getRows() const { return rows_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<ExpressionPtr> rows_;
};

// Join Classes
class JoinCriteria : public Node {
public:
  explicit JoinCriteria(NodeType type) : Node(type) {}
};

using JoinCriteriaPtr = std::shared_ptr<JoinCriteria>;

class JoinOn : public JoinCriteria {
public:
  explicit JoinOn(ExpressionPtr expression)
      : JoinCriteria(NodeType::kJoinOn), expression_(expression) {}
  
  const ExpressionPtr& getExpression() const { return expression_; }
  void accept(AstVisitor* visitor) override;

private:
  ExpressionPtr expression_;
};

class JoinUsing : public JoinCriteria {
public:
  explicit JoinUsing(const std::vector<std::shared_ptr<Identifier>>& columns)
      : JoinCriteria(NodeType::kJoinUsing), columns_(columns) {}
  
  const std::vector<std::shared_ptr<Identifier>>& getColumns() const { return columns_; }
  void accept(AstVisitor* visitor) override;

private:
  std::vector<std::shared_ptr<Identifier>> columns_;
};

class Join : public Relation {
public:
  enum class Type { CROSS, INNER, LEFT, RIGHT, FULL, IMPLICIT };
  
  Join(Type type, RelationPtr left, RelationPtr right, std::optional<JoinCriteriaPtr> criteria = std::nullopt)
      : Relation(NodeType::kJoin), type_(type), left_(left), right_(right), criteria_(criteria) {}
  
  Type getType() const { return type_; }
  const RelationPtr& getLeft() const { return left_; }
  const RelationPtr& getRight() const { return right_; }
  const std::optional<JoinCriteriaPtr>& getCriteria() const { return criteria_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Type type_;
  RelationPtr left_;
  RelationPtr right_;
  std::optional<JoinCriteriaPtr> criteria_;
};

class NaturalJoin : public Relation {
public:
  NaturalJoin(Join::Type type, RelationPtr left, RelationPtr right)
      : Relation(NodeType::kNaturalJoin), type_(type), left_(left), right_(right) {}
  
  Join::Type getType() const { return type_; }
  const RelationPtr& getLeft() const { return left_; }
  const RelationPtr& getRight() const { return right_; }
  
  void accept(AstVisitor* visitor) override;

private:
  Join::Type type_;
  RelationPtr left_;
  RelationPtr right_;
};

// Set Operations
class SetOperation : public QueryBody {
public:
  SetOperation(NodeType type, QueryBodyPtr left, QueryBodyPtr right, bool distinct)
      : QueryBody(type), left_(left), right_(right), distinct_(distinct) {}
  
  const QueryBodyPtr& getLeft() const { return left_; }
  const QueryBodyPtr& getRight() const { return right_; }
  bool isDistinct() const { return distinct_; }

protected:
  QueryBodyPtr left_;
  QueryBodyPtr right_;
  bool distinct_;
};

class Union : public SetOperation {
public:
  Union(QueryBodyPtr left, QueryBodyPtr right, bool distinct)
      : SetOperation(NodeType::kUnion, left, right, distinct) {}
  
  void accept(AstVisitor* visitor) override;
};

class Intersect : public SetOperation {
public:
  Intersect(QueryBodyPtr left, QueryBodyPtr right, bool distinct)
      : SetOperation(NodeType::kIntersect, left, right, distinct) {}
  
  void accept(AstVisitor* visitor) override;
};

class Except : public SetOperation {
public:
  Except(QueryBodyPtr left, QueryBodyPtr right, bool distinct)
      : SetOperation(NodeType::kExcept, left, right, distinct) {}
  
  void accept(AstVisitor* visitor) override;
};

} // namespace facebook::velox::sql