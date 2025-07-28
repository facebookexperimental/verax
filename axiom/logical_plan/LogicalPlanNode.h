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

#include "axiom/logical_plan/Expr.h"
#include "velox/common/Enums.h"
#include "velox/type/Variant.h"

namespace facebook::velox::logical_plan {

enum class NodeKind {
  kValues = 0,
  kTableScan = 1,
  kFilter = 2,
  kProject = 3,
  kAggregate = 4,
  kJoin = 5,
  kSort = 6,
  kLimit = 7,
  kSet = 8,
  kUnnest = 9,
};

VELOX_DECLARE_ENUM_NAME(NodeKind)

class LogicalPlanNode;
using LogicalPlanNodePtr = std::shared_ptr<const LogicalPlanNode>;

class PlanNodeVisitor;
class PlanNodeVisitorContext;

/// Base class for all logical plan nodes. Every plan node has a unique ID and
/// zero or more inputs. Leaf nodes like Scan and Values have no inputs. Joins
/// have two inputs. Union may have many inputs. Most other nodes have just
/// one input. Every plan node has an output schema (list of names and types
/// of output columns) expressed as a RowType.
class LogicalPlanNode {
 public:
  LogicalPlanNode(
      NodeKind kind,
      const std::string& id,
      const std::vector<LogicalPlanNodePtr>& inputs,
      const RowTypePtr& outputType)
      : kind_{kind}, id_{id}, inputs_{inputs}, outputType_{outputType} {
    VELOX_USER_CHECK_NOT_NULL(outputType);
    for (const auto& input : inputs_) {
      VELOX_USER_CHECK_NOT_NULL(input);
    }
  }

  virtual ~LogicalPlanNode() = default;

  NodeKind kind() const {
    return kind_;
  }

  template <typename T>
  const T* asUnchecked() const {
    return dynamic_cast<const T*>(this);
  }

  const std::string& id() const {
    return id_;
  }

  const std::vector<LogicalPlanNodePtr>& inputs() const {
    return inputs_;
  }

  /// Returns the only input. Throws if there are zero or more than one inputs.
  const LogicalPlanNodePtr& onlyInput() const {
    VELOX_USER_CHECK_EQ(1, inputs_.size());
    return inputs_.at(0);
  }

  /// Convenience getter for the input at the specified index. A shortcut for
  /// inputs().at(index).
  const LogicalPlanNodePtr& inputAt(size_t index) const {
    VELOX_USER_CHECK_LT(index, inputs_.size());
    return inputs_.at(index);
  }

  const RowTypePtr& outputType() const {
    return outputType_;
  }

  virtual void accept(
      const PlanNodeVisitor& visitor,
      PlanNodeVisitorContext& context) const = 0;

 private:
  const NodeKind kind_;
  const std::string id_;
  const std::vector<LogicalPlanNodePtr> inputs_;
  const RowTypePtr outputType_;
};

/// A table whose content is embedded in the plan.
class ValuesNode : public LogicalPlanNode {
 public:
  /// @param rowType Output schema. A list of column names and types. All names
  /// must be non-empty and unique.
  /// @param rows A list of rows. Each row is a list of values, one per column.
  /// The number, order and types of columns must match 'rowType'.
  ValuesNode(
      const std::string& id,
      const RowTypePtr& rowType,
      std::vector<Variant> rows);

  const std::vector<Variant>& rows() const {
    return rows_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  const std::vector<Variant> rows_;
};

using ValuesNodePtr = std::shared_ptr<const ValuesNode>;

/// A table accessible through a connector.
class TableScanNode : public LogicalPlanNode {
 public:
  /// @param id Unique ID of the plan node.
  /// @param outputType Output schema. A list of column names and types.
  /// @param connectorId ID of the connector to use to access the table.
  /// @param tableName Table name.
  /// @param columnNames A list of column names. Must align with 'outputType',
  /// which may expose columns under different names.
  TableScanNode(
      const std::string& id,
      const RowTypePtr& outputType,
      const std::string& connectorId,
      const std::string& tableName,
      const std::vector<std::string>& columnNames)
      : LogicalPlanNode(NodeKind::kTableScan, id, {}, outputType),
        connectorId_{connectorId},
        tableName_(tableName),
        columnNames_(columnNames) {
    VELOX_USER_CHECK_EQ(outputType->size(), columnNames.size());

    const auto numColumns = outputType->size();
    for (auto i = 0; i < numColumns; ++i) {
      VELOX_USER_CHECK(!outputType->nameOf(i).empty());
      VELOX_USER_CHECK(!columnNames.at(i).empty());
    }
  }

  const std::string& connectorId() const {
    return connectorId_;
  }

  const std::string& tableName() const {
    return tableName_;
  }

  const std::vector<std::string>& columnNames() const {
    return columnNames_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  const std::string connectorId_;
  const std::string tableName_;
  const std::vector<std::string> columnNames_;
};

using TableScanNodePtr = std::shared_ptr<const TableScanNode>;

/// Eliminates zero or more rows from the input based on a boolean expression.
/// The output schema is the same as the input schema. This node may reduce the
/// cardinality of the dataset, but it cannot increase it.
class FilterNode : public LogicalPlanNode {
 public:
  FilterNode(
      const std::string& id,
      const LogicalPlanNodePtr& input,
      const ExprPtr& predicate)
      : LogicalPlanNode(NodeKind::kFilter, id, {input}, input->outputType()),
        predicate_{predicate} {
    VELOX_USER_CHECK_NOT_NULL(predicate);
    VELOX_USER_CHECK_EQ(predicate->type()->kind(), TypeKind::BOOLEAN);
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

  const ExprPtr& predicate() const {
    return predicate_;
  }

 private:
  const ExprPtr predicate_;
};

using FilterNodePtr = std::shared_ptr<const FilterNode>;

/// Produces one or more columns using specified expressions. The output
/// schema matches the list of names and expressions. This not doesn't change
/// the cardinality of the dataset.
class ProjectNode : public LogicalPlanNode {
 public:
  /// @param names List of output column names. Names must be non-empty and
  /// unique.
  /// @param expressions List of expressions aligned with 'names'. Expressions
  /// may be scalar or window function calls. For any give expression, only
  /// the root expression can be a window function call. Scalar expressions
  /// over window function calls are not supported.
  ProjectNode(
      const std::string& id,
      const LogicalPlanNodePtr& input,
      const std::vector<std::string>& names,
      const std::vector<ExprPtr>& expressions)
      : LogicalPlanNode(
            NodeKind::kProject,
            id,
            {input},
            makeOutputType(names, expressions)),
        names_{names},
        expressions_{expressions} {}

  const std::vector<std::string>& names() const {
    return names_;
  }

  const std::vector<ExprPtr>& expressions() const {
    return expressions_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  static RowTypePtr makeOutputType(
      const std::vector<std::string>& names,
      const std::vector<ExprPtr>& expressions);

  const std::vector<std::string> names_;
  const std::vector<ExprPtr> expressions_;
};

using ProjectNodePtr = std::shared_ptr<const ProjectNode>;

/// Groups input data on one or more sets of grouping keys, calculating each
/// measure for each combination of the grouping keys. The order of columns in
/// the output schema is: grouping keys, followed by aggregations, optionally
/// followed by a grouping set index. This node may decrease the cardinality of
/// the dataset. Unless there are multiple grouping sets, this node cannot
/// increase the cardinality.
class AggregateNode : public LogicalPlanNode {
 public:
  using GroupingSet = std::vector<int32_t>;

  /// @param groupingKeys Zero or more grouping keys. If empty, at least one
  /// aggregate must be specified.
  /// @param groupingSets Zero of more grouping sets. Each set is a list of
  /// grouping keys specified by an index into 'groupingKeys' array.
  /// @param aggregates Zero or more aggregates. Must be aligned with
  /// 'aggregateNames'. If empty, at least one grouping key must be specified.
  /// Each aggregate is a call to an aggregate function. Each aggregate may
  /// specify (1) a filter to apply to the input rows; (2) a sorting order to
  /// apply to input rows; (2) whether to deduplicate input rows. Used to
  /// support agg(DISTINCT x ORDER BY y) FILTER (WHERE f(z)) SQL syntax.
  /// @param outputNames List of names of the output columns: one name for each
  /// grouping key, followed by one name for each aggregate, followed
  /// by the name of a column that contains grouping set index if 'groupingSets'
  /// is not empty. Names must be unique.
  AggregateNode(
      const std::string& id,
      const LogicalPlanNodePtr& input,
      const std::vector<ExprPtr>& groupingKeys,
      const std::vector<GroupingSet>& groupingSets,
      const std::vector<AggregateExprPtr>& aggregates,
      const std::vector<std::string>& outputNames)
      : LogicalPlanNode(
            NodeKind::kAggregate,
            id,
            {input},
            makeOutputType(
                groupingKeys,
                groupingSets,
                aggregates,
                outputNames)),
        groupingKeys_{groupingKeys},
        groupingSets_{groupingSets},
        aggregates_{aggregates},
        outputNames_{outputNames} {
    if (groupingKeys.empty()) {
      VELOX_USER_CHECK(
          !aggregates.empty(),
          "Aggregation node must specify at least one aggregate or grouping key");
    }

    if (aggregates.empty()) {
      VELOX_USER_CHECK(
          !groupingKeys.empty(),
          "Aggregation node must specify at least one aggregate or grouping key");
    }

    for (const auto& groupingSet : groupingSets) {
      for (const auto& key : groupingSet) {
        VELOX_USER_CHECK_LT(key, groupingKeys.size());
      }
    }
  }

  const std::vector<ExprPtr>& groupingKeys() const {
    return groupingKeys_;
  }

  const std::vector<GroupingSet>& groupingSets() const {
    return groupingSets_;
  }

  const std::vector<std::string>& outputNames() const {
    return outputNames_;
    ;
  }

  const std::vector<AggregateExprPtr>& aggregates() const {
    return aggregates_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  static RowTypePtr makeOutputType(
      const std::vector<ExprPtr>& groupingKeys,
      const std::vector<GroupingSet>& groupingSets,
      const std::vector<AggregateExprPtr>& aggregates,
      const std::vector<std::string>& outputNames);

  const std::vector<ExprPtr> groupingKeys_;
  const std::vector<GroupingSet> groupingSets_;
  const std::vector<AggregateExprPtr> aggregates_;
  const std::vector<std::string> outputNames_;
};

using AggregateNodePtr = std::shared_ptr<const AggregateNode>;

enum class JoinType {
  /// For each row on the left, find all matching rows on the right and return
  /// all combinations.
  kInner = 0,

  /// For each row on the left, find all matching rows on the right and return
  /// all combinations. In addition, return all rows from the left that have no
  /// match on the right with right-side columns filled with nulls.
  kLeft = 1,

  /// Opposite of kLeft. For each row on the right, find all matching rows on
  /// the left and return all combinations. In addition, return all rows from
  /// the
  /// right that have no match on the left with left-side columns filled with
  /// nulls.
  kRight = 2,

  /// A "union" of kLeft and kRight. For each row on the left, find all
  /// matching rows on the right and return all combinations. In addition,
  /// return all rows from the left that have no match on the right with
  /// right-side columns filled with nulls. Also, return
  /// all rows from the right that have no match on the left with left-side
  /// columns filled with nulls.
  kFull = 3,
};

VELOX_DECLARE_ENUM_NAME(JoinType)

/// Combines two separate inputs into a single output, based on a boolean join
/// condition. The output schema contains all columns from the left input
/// followed by all columns from the right input.
class JoinNode : public LogicalPlanNode {
 public:
  /// @param condition Optional join condition. If nullptr, the output is a
  /// cross product of the inputs.
  JoinNode(
      const std::string& id,
      const LogicalPlanNodePtr& left,
      const LogicalPlanNodePtr& right,
      JoinType joinType,
      const ExprPtr& condition)
      : LogicalPlanNode(
            NodeKind::kJoin,
            id,
            {left, right},
            makeOutputType(left, right)),
        joinType_{joinType},
        condition_{condition} {
    if (condition != nullptr) {
      VELOX_USER_CHECK_EQ(condition->typeKind(), TypeKind::BOOLEAN);
    }
  }

  const LogicalPlanNodePtr& left() const {
    return inputAt(0);
  }

  const LogicalPlanNodePtr& right() const {
    return inputAt(1);
  }

  JoinType joinType() const {
    return joinType_;
  }

  const ExprPtr& condition() const {
    return condition_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  static RowTypePtr makeOutputType(
      const LogicalPlanNodePtr& left,
      const LogicalPlanNodePtr& right);

  const JoinType joinType_;
  const ExprPtr condition_;
};

using JoinNodePtr = std::shared_ptr<const JoinNode>;

/// Sort rows based on one or more sort fields. The output schema for this node
/// matches the input. This node doesn't change the cardinality of the dataset.
class SortNode : public LogicalPlanNode {
 public:
  SortNode(
      const std::string& id,
      const LogicalPlanNodePtr& input,
      const std::vector<SortingField>& ordering)
      : LogicalPlanNode(NodeKind::kSort, id, {input}, input->outputType()),
        ordering_{ordering} {
    VELOX_USER_CHECK(!ordering.empty());
  }

  const std::vector<SortingField> ordering() const {
    return ordering_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  const std::vector<SortingField> ordering_;
};

using SortNodePtr = std::shared_ptr<const SortNode>;

/// Eliminates rows outside of the desired window. Returns up to 'count' rows
/// starting from 'offset'. The output schema of this node matches the input.
class LimitNode : public LogicalPlanNode {
 public:
  /// @param offset Zero-based index of the first row to return. Must be >= 0.
  /// @param count Maximum number of rows to return. Must be >= 0. If zero, the
  /// node produces empty dataset.
  LimitNode(
      const std::string& id,
      const LogicalPlanNodePtr& input,
      int64_t offset,
      int64_t count)
      : LogicalPlanNode(NodeKind::kLimit, id, {input}, input->outputType()),
        offset_{offset},
        count_{count} {
    VELOX_USER_CHECK_GE(offset, 0);
    VELOX_USER_CHECK_GE(count, 0);
  }

  int64_t offset() const {
    return offset_;
  }

  int64_t count() const {
    return count_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  const int64_t offset_;
  const int64_t count_;
};

using LimitNodePtr = std::shared_ptr<const LimitNode>;

enum class SetOperation {
  /// Returns all rows from all inputs after removing duplicates.
  kUnion = 0,

  /// Returns all rows from all inputs.
  kUnionAll = 1,

  /// Returns a subset of rows that are present in all inputs.
  kIntersect = 2,

  /// Returns a subset of rows in the first input that are not found in any
  /// other input.
  kExcept = 3,
};

VELOX_DECLARE_ENUM_NAME(SetOperation)

/// Set-level operation that supports combining datasets, possibly excluding
/// rows based on various types of row level matching.
///
/// All inputs must have compatible types. Number and types of columns must be
/// the same. Columns names being unique will be different. The output schema of
/// the Set node is the schema of the first input. Column names in the output of
/// the Set match column names in the first input.
///
/// Set operation must specify at least 2 inputs.
class SetNode : public LogicalPlanNode {
 public:
  SetNode(
      const std::string& id,
      const std::vector<LogicalPlanNodePtr>& inputs,
      SetOperation operation);

  SetOperation operation() const {
    return operation_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  const SetOperation operation_;
};

using SetNodePtr = std::shared_ptr<const SetNode>;

/// Used to expand an ARRAY or MAP into a relation. Arrays are expanded into a
/// single column, and maps are expanded into two columns (key, value). Can also
/// be used to expand multiple arrays and maps, in which case they are expanded
/// into multiple columns, with as many rows as the highest cardinality array or
/// map (the other columns are padded with nulls). Can optionally request to
/// produce an ordinality column.
///
/// If 'flattenArrayOfRows' is true, ARRAY(ROW) is expanded into multiple
/// columns, one per struct field.
///
/// Duplicates each input row by combining it with every row in the relation
/// produced by expanding the unnest expressions.
///
/// Rows with empty arrays or map do not appear in the output.
///
/// This operation typically increases the cardinality of the dataset. However,
/// it may also reduce the cardinality if there are many rows with empty arrays
/// and maps.
///
/// The output schema contains all columns from the input, followed by columns
/// in the relation produced by expanding the unnest expressions, followed by an
/// optional ordinality column of type BIGINT.
class UnnestNode : public LogicalPlanNode {
 public:
  UnnestNode(
      /// @param unnestExpressions One or more expressions that produce ARRAYs
      /// or MAPs.
      /// @param unnestedNames Names to use for expanded relations. Must align
      /// with 'unnestExpressions'. Each ARRAY requires one name. Each MAP
      /// requires two maps. If 'flattenArrayOfRows' is true, each ARRAY(ROW)
      /// requires as many names as there are fields in the ROW.
      /// @param ordinalityName Optional name for the ordinality output column.
      /// If not specified, ordinality column is not added. Ordinality is
      /// 1-based.
      const std::string& id,
      const LogicalPlanNodePtr& input,
      const std::vector<ExprPtr>& unnestExpressions,
      const std::vector<std::vector<std::string>>& unnestedNames,
      const std::optional<std::string>& ordinalityName,
      bool flattenArrayOfRows = false)
      : LogicalPlanNode(
            NodeKind::kUnnest,
            id,
            {input},
            makeOutputType(
                input,
                unnestExpressions,
                unnestedNames,
                ordinalityName,
                flattenArrayOfRows)),
        unnestExpressions_(unnestExpressions),
        unnestedNames_(unnestedNames),
        ordinalityName_{ordinalityName},
        flattenArrayOfRows_{flattenArrayOfRows} {
    if (ordinalityName.has_value()) {
      VELOX_USER_CHECK(
          !ordinalityName->empty(), "Ordinality column name must be not empty");
    }
  }

  const std::vector<ExprPtr>& unnestExpressions() const {
    return unnestExpressions_;
  }

  const std::vector<std::vector<std::string>>& unnestedNames() const {
    return unnestedNames_;
  }

  const std::optional<std::string>& ordinalityName() const {
    return ordinalityName_;
  }

  bool flattenArrayOfRows() const {
    return flattenArrayOfRows_;
  }

  void accept(const PlanNodeVisitor& visitor, PlanNodeVisitorContext& context)
      const override;

 private:
  static RowTypePtr makeOutputType(
      const LogicalPlanNodePtr& input,
      const std::vector<ExprPtr>& unnestExpressions,
      const std::vector<std::vector<std::string>>& unnestedNames,
      const std::optional<std::string>& ordinalityName,
      bool flattenArrayOfRows);

  const std::vector<ExprPtr> unnestExpressions_;
  const std::vector<std::vector<std::string>> unnestedNames_;
  const std::optional<std::string> ordinalityName_;
  const bool flattenArrayOfRows_;
};

using UnnestNodePtr = std::shared_ptr<const UnnestNode>;

} // namespace facebook::velox::logical_plan
