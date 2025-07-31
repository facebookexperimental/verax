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
#include "axiom/optimizer/Schema.h"
#include "velox/core/PlanNode.h"

/// Defines subclasses of PlanObject for describing the logical
/// structure of queries. These are the constraints that guide
/// generation of plan candidates. These are referenced from
/// candidates but stay immutable acrosss the candidate
/// generation. Sometimes new derived tables may be added for
/// representing constraints on partial plans but otherwise these stay
/// constant.
namespace facebook::velox::optimizer {

/// The join structure is described as a tree of derived tables with
/// base tables as leaves. Joins are described as join graph
/// edges. Edges describe direction for non-inner joins. Scalar and
/// existence subqueries are flattened into derived tables or base
/// tables. The join graph would represent select ... from t where
/// exists(x) or exists(y) as a derived table of three joined tables
/// where the edge from t to x and t to y is directed and qualified as
/// left semijoin. The semijoins project out one column, an existence
/// flag. The filter would be expresssed as a conjunct under the top
/// derived table with x-exists or y-exists.

/// A bit set that qualifies an Expr. Represents which functions/kinds
/// of functions are found inside the children of an Expr.
class FunctionSet {
 public:
  /// Indicates and aggregate function in the set.
  static constexpr uint64_t kAggregate = 1;

  /// Indicates a non-determinstic function
  static constexpr uint64_t kNondeterministic = 1UL << 1;

  FunctionSet() : set_(0) {}
  explicit FunctionSet(uint64_t set) : set_(set) {}

  /// True if 'item' is in 'this'.
  bool contains(int64_t item) const {
    return 0 != (set_ & item);
  }

  /// Unions 'this' and 'other' and returns the result.
  FunctionSet operator|(const FunctionSet& other) const {
    return FunctionSet(set_ | other.set_);
  }

  /// Unions 'this' and 'other' and returns the result.
  FunctionSet operator|(uint64_t other) const {
    return FunctionSet(set_ | other);
  }

 private:
  uint64_t set_;
};

/// Superclass for all expressions.
class Expr : public PlanObject {
 public:
  Expr(PlanType type, const Value& value) : PlanObject(type), value_(value) {}

  bool isExpr() const override {
    return true;
  }

  // Returns the single base or derived table 'this' depends on, nullptr if
  // 'this' depends on none or multiple tables.
  PlanObjectCP singleTable() const;

  /// Returns all tables 'this' depends on.
  PlanObjectSet allTables() const;

  /// True if '&other == this' or is recursively equal with column
  /// leaves either same or in same equivalence.
  bool sameOrEqual(const Expr& other) const;

  const PlanObjectSet& columns() const {
    return columns_;
  }

  const PlanObjectSet& subexpressions() const {
    return subexpressions_;
  }

  const Value& value() const {
    return value_;
  }

  /// True if 'this' contains any function from 'set'. See FunctionSet.
  virtual bool containsFunction(uint64_t /*set*/) const {
    return false;
  }

  virtual const FunctionSet& functions() const;

 protected:
  // The columns this depends on.
  PlanObjectSet columns_;

  // All expressions 'this' depends on.
  PlanObjectSet subexpressions_;

  // Type Constraints on the value of 'this'.
  Value value_;
};

/// If 'object' is an Expr, returns Expr::singleTable, else nullptr.
PlanObjectCP singleTable(PlanObjectCP object);

struct Equivalence;
using EquivalenceP = Equivalence*;

/// Represents a literal.
class Literal : public Expr {
 public:
  Literal(const Value& value, const velox::variant* literal)
      : Expr(PlanType::kLiteral, value), literal_(literal), vector_(nullptr) {}

  Literal(const Value& value, const BaseVector* vector)
      : Expr(PlanType::kLiteral, value), literal_{}, vector_(vector) {}

  const velox::variant& literal() const {
    return *literal_;
  }

  bool hasVector() const {
    return vector_ != nullptr;
  }

  const BaseVector* vector() const {
    return vector_;
  }

  std::string toString() const override;

 private:
  const velox::variant* const literal_;
  const BaseVector* const vector_;
};

/// Represents a column. A column is always defined by a relation, whether table
/// or derived table.
class Column : public Expr {
 public:
  Column(
      Name _name,
      PlanObjectP _relation,
      const Value& value,
      Name nameInTable = nullptr,
      ColumnCP topColumn = nullptr,
      PathCP path = nullptr);

  Name name() const {
    return name_;
  }

  PlanObjectCP relation() const {
    return relation_;
  }

  ColumnCP schemaColumn() const {
    return schemaColumn_;
  }

  /// Asserts that 'this' and 'other' are joined on equality. This has a
  /// transitive effect, so if a and b are previously asserted equal and c is
  /// asserted equal to b, a and c are also equal.
  void equals(ColumnCP other) const;

  std::string toString() const override;

  struct Equivalence* equivalence() const {
    return equivalence_;
  }

  ColumnCP topColumn() const {
    return topColumn_;
  }

  PathCP path() const {
    return path_;
  }

 private:
  // Last part of qualified name.
  Name name_;

  // The defining BaseTable or DerivedTable.
  PlanObjectP relation_;

  // Equivalence class. Lists all columns directly or indirectly asserted equal
  // to 'this'.
  mutable EquivalenceP equivalence_{nullptr};

  // If this is a column of a BaseTable, points to the corresponding
  // column in the SchemaTable. Used for matching with
  // ordering/partitioning columns in the SchemaTable.
  ColumnCP schemaColumn_{nullptr};

  // Containing top level column if 'this' is a subfield projected out as
  // column.
  ColumnCP topColumn_;

  // Path from 'topColumn'.
  PathCP path_;
};

template <typename T>
inline folly::Range<T*> toRange(const std::vector<T, QGAllocator<T>>& v) {
  return folly::Range<T const*>(v.data(), v.size());
}

class Field : public Expr {
 public:
  Field(const Type* type, ExprCP base, Name field)
      : Expr(PlanType::kField, Value(type, 1)), field_(field), base_(base) {
    columns_ = base->columns();
    subexpressions_ = base->subexpressions();
  }

  Field(const Type* type, ExprCP base, int32_t index)
      : Expr(PlanType::kField, Value(type, 1)),
        field_(nullptr),
        index_(index),
        base_(base) {
    columns_ = base->columns();
    subexpressions_ = base->subexpressions();
  }

  Name field() const {
    return field_;
  }

  int32_t index() const {
    return index_;
  }

  std::string toString() const override;

  ExprCP base() const {
    return base_;
  }

 private:
  Name field_;
  int32_t index_;
  ExprCP base_;
};

template <typename T, typename U>
inline CPSpan<T> toRangeCast(U v) {
  return CPSpan<T>(reinterpret_cast<const T* const*>(v.data()), v.size());
}

struct SubfieldSet {
  /// Id of an accessed column of complex type.
  std::vector<int32_t, QGAllocator<int32_t>> ids;

  // Set of subfield paths that are accessed for the corresponding 'column'.
  // empty means that all subfields are accessed.
  std::vector<BitSet, QGAllocator<BitSet>> subfields;

  std::optional<BitSet> findSubfields(int32_t id) const;
};

/// Describes where the args given to a lambda come from.
enum class LambdaArg : int8_t { kKey, kValue, kElement };

struct LambdaInfo {
  /// The ordinal of the lambda in the function's args
  int32_t ordinal;
  /// Getter applied to the collection given in corresponding 'argOrdinal' to
  /// get each argument of the lambda.
  std::vector<LambdaArg> lambdaArg;
  /// Argument giving the collection
  std::vector<int32_t> argOrdinal;
};

class Call;

struct ResultAccess;

/// Describes functions accepting lambdas and functions with special treatment
/// of subfields.
struct FunctionMetadata {
  bool processSubfields() const {
    return subfieldArg.has_value() || !fieldIndexForArg.empty() ||
        isArrayConstructor || isMapConstructor;
  }

  LambdaInfo* lambdaInfo(int32_t i) {
    for (auto j = 0; j < lambdas.size(); ++j) {
      if (lambdas[j].ordinal == i) {
        return &lambdas[j];
      }
    }
    return nullptr;
  }

  std::vector<LambdaInfo> lambdas;

  /// If accessing a subfield on the result means that the same subfield is
  /// required in an argument, this is the ordinal of the argument. This is 1
  /// for transform_values, which means that transform_values(f, map)[1] implies
  /// that key 1 is accessed in 'map'.
  std::optional<int32_t> subfieldArg;

  /// If true, then access of subscript 'i' in result means that argument 'i' is
  /// accessed.
  bool isArrayConstructor{false};

  /// If key 'k' in result is accessed, then the argument that corresponds to
  /// this key is accessed.
  bool isMapConstructor{false};

  /// If ordinal fieldIndexForArg_[i] is accessed, then argument argOrdinal_[i]
  /// is accessed.
  std::vector<int32_t> fieldIndexForArg;

  /// Ordinal of argument that produces the result subfield in the corresponding
  /// element of 'fieldIndexForArg_'.
  std::vector<int32_t> argOrdinal;

  /// bits of FunctionSet for the function.
  FunctionSet functionSet;

  /// Static fixed cost for processing one row. use 'costFunc' for non-constant
  /// cost.
  float cost{1};

  /// Function for evaluating the per-row cost when the cost depends on
  /// arguments and their stats.
  std::function<float(const Call*)> costFunc;

  /// Translates a set of paths into path, expression pairs if the complex type
  /// returning function is decomposable into per-path subexpressions. Suppose
  /// the function applies array sort to all arrays in a map. suppose it is used
  /// in [k1][0] and [k2][1]. This could return [k1] = array_sort(arg[k1]) and
  /// k2 = array_sort(arg[k2]. 'arg'  comes from 'call'.
  std::function<std::unordered_map<PathCP, core::TypedExprPtr>(
      const core::CallTypedExpr* call,
      std::vector<PathCP>& paths)>
      explode;

  std::function<std::unordered_map<PathCP, logical_plan::ExprPtr>(
      const logical_plan::CallExpr* call,
      std::vector<PathCP>& paths)>
      logicalExplode;
};

const FunctionMetadata* functionMetadata(Name name);

/// Represents a function call or a special form, any expression with
/// subexpressions.
class Call : public Expr {
 public:
  Call(
      PlanType type,
      Name name,
      const Value& value,
      ExprVector args,
      FunctionSet functions)
      : Expr(type, value),
        name_(name),
        args_(std::move(args)),
        functions_(functions),
        metadata_(functionMetadata(name_)) {
    for (auto arg : args_) {
      columns_.unionSet(arg->columns());
      subexpressions_.unionSet(arg->subexpressions());
      subexpressions_.add(arg);
    }
  }

  Call(Name name, Value value, ExprVector args, FunctionSet functions)
      : Call(PlanType::kCall, name, value, std::move(args), functions) {}

  Name name() const {
    return name_;
  }

  const FunctionSet& functions() const override {
    return functions_;
  }

  bool isFunction() const override {
    return true;
  }

  bool containsFunction(uint64_t set) const override {
    return functions_.contains(set);
  }

  const ExprVector& args() const {
    return args_;
  }

  ExprCP argAt(size_t index) const {
    return args_[index];
  }

  CPSpan<PlanObject> children() const override {
    return folly::Range<const PlanObject* const*>(
        reinterpret_cast<const PlanObject* const*>(args_.data()), args_.size());
  }

  std::string toString() const override;

  const FunctionMetadata* metadata() const {
    return metadata_;
  }

 private:
  // name of function.
  Name const name_;

  // Arguments.
  const ExprVector args_;

  // Set of functions used in 'this' and 'args'.
  const FunctionSet functions_;

  const FunctionMetadata* metadata_;
};

using CallCP = const Call*;

/// True if 'expr' is a call to function 'name'.
bool isCallExpr(ExprCP expr, Name name);

/// Represents a lambda. May occur as an immediate argument of selected
/// functions.
class Lambda : public Expr {
 public:
  Lambda(ColumnVector args, const Type* type, ExprCP body)
      : Expr(PlanType::kLambda, Value(type, 1)),
        args_(std::move(args)),
        body_(body) {}
  const ColumnVector& args() const {
    return args_;
  }

  ExprCP body() const {
    return body_;
  }

 private:
  ColumnVector args_;
  ExprCP body_;
};

/// Represens a set of transitively equal columns.
struct Equivalence {
  // Each element has a direct or implied equality edge to every other.
  ColumnVector columns;
};

/// Represents one side of a join. See Join below for the meaning of the
/// members.
struct JoinSide {
  PlanObjectCP table;
  const ExprVector& keys;
  float fanout;
  const bool isOptional;
  const bool isNonOptionalOfOuter;
  const bool isExists;
  const bool isNotExists;
  ColumnCP markColumn;
  const bool isUnique;

  /// Returns the join type to use if 'this' is the right side.
  velox::core::JoinType leftJoinType() const {
    if (isNotExists) {
      return velox::core::JoinType::kAnti;
    }
    if (isExists) {
      return velox::core::JoinType::kLeftSemiFilter;
    }
    if (isOptional) {
      return velox::core::JoinType::kLeft;
    }
    if (isNonOptionalOfOuter) {
      return velox::core::JoinType::kRight;
    }

    if (markColumn) {
      return velox::core::JoinType::kLeftSemiProject;
    }
    return velox::core::JoinType::kInner;
  }
};

/// Represents a possibly directional equality join edge.
/// 'rightTable' is always set. 'leftTable' is nullptr if 'leftKeys' come from
/// different tables. If so, 'this' must be non-inner and not full outer.
/// 'filter' is a list of post join conjuncts. This should be present only in
/// non-inner joins. For inner joins these are representable as freely
/// decomposable and reorderable conjuncts.
class JoinEdge {
 public:
  JoinEdge(
      PlanObjectCP leftTable,
      PlanObjectCP rightTable,
      ExprVector filter,
      bool leftOptional,
      bool rightOptional,
      bool rightExists,
      bool rightNotExists,
      ColumnCP markColumn = nullptr,
      bool directed = false)
      : leftTable_(leftTable),
        rightTable_(rightTable),
        filter_(std::move(filter)),
        leftOptional_(leftOptional),
        rightOptional_(rightOptional),
        rightExists_(rightExists),
        rightNotExists_(rightNotExists),
        markColumn_(markColumn),
        directed_(directed) {
    if (isInner()) {
      VELOX_CHECK(filter_.empty());
    }
  }

  PlanObjectCP leftTable() const {
    return leftTable_;
  }

  PlanObjectCP rightTable() const {
    return rightTable_;
  }

  const ExprVector& leftKeys() const {
    return leftKeys_;
  }

  const ExprVector& rightKeys() const {
    return rightKeys_;
  }

  float lrFanout() const {
    return lrFanout_;
  }

  float rlFanout() const {
    return rlFanout_;
  }

  bool leftOptional() const {
    return leftOptional_;
  }

  bool rightOptional() const {
    return rightOptional_;
  }

  void addEquality(ExprCP left, ExprCP right, bool update = false);

  /// True if inner join.
  bool isInner() const {
    return !leftOptional_ && !rightOptional_ && !rightExists_ &&
        !rightNotExists_;
  }

  // True if all tables referenced from 'leftKeys' must be placed before placing
  // this.
  bool isNonCommutative() const {
    // Inner and full outer joins are commutative.
    if (rightOptional_ && leftOptional_) {
      return false;
    }
    return !leftTable_ || rightOptional_ || leftOptional_ || rightExists_ ||
        rightNotExists_ || markColumn_ || directed_;
  }
  // Returns the join side info for 'table'. If 'other' is set, returns the
  // other side.
  const JoinSide sideOf(PlanObjectCP side, bool other = false) const;

  /// Returns the table on the otherside of 'table' and the number of rows in
  /// the returned table for one row in 'table'. If the join is not inner
  /// returns nullptr, 0.
  std::pair<PlanObjectCP, float> otherTable(PlanObjectCP table) const {
    return leftTable_ == table && !leftOptional_
        ? std::pair<PlanObjectCP, float>{rightTable_, lrFanout_}
        : rightTable_ == table && !rightOptional_ && !rightExists_
        ? std::pair<PlanObjectCP, float>{leftTable_, rlFanout_}
        : std::pair<PlanObjectCP, float>{nullptr, 0};
  }

  const ExprVector& filter() const {
    return filter_;
  }

  void setFanouts(float rl, float lr) {
    fanoutsFixed_ = true;
    lrFanout_ = lr;
    rlFanout_ = rl;
  }

  std::string toString() const;

  //// Fills in 'lrFanout' and 'rlFanout', 'leftUnique', 'rightUnique'.
  void guessFanout();

  // True if a hash join build can be broadcastable. Used when building on the
  // right. None of the right hash join variants is broadcastable.
  bool isBroadcastableType() const;

  /// Returns a key string for recording a join cardinality sample. The string
  /// is empty if not applicable. The bool is true if the key has right table
  /// before left.
  std::pair<std::string, bool> sampleKey() const;

 private:
  // Leading left side join keys.
  ExprVector leftKeys_;
  // Leading right side join keys, compared equals to 1:1 to 'leftKeys'.
  ExprVector rightKeys_;

  PlanObjectCP const leftTable_;
  PlanObjectCP const rightTable_;

  // 'rightKeys' select max 1 'leftTable' row.
  bool leftUnique_{false};

  // 'leftKeys' select max 1 'rightTable' row.
  bool rightUnique_{false};

  // number of right side rows selected for one row on the left.
  float lrFanout_{1};

  // Number of left side rows selected for one row on the right.
  float rlFanout_{1};

  // True if 'lrFanout_' and 'rlFanout_' are set by setFanouts.
  bool fanoutsFixed_{false};

  // Join condition for any non-equality  conditions for non-inner joins.
  const ExprVector filter_;

  // True if an unprobed right side row produces a result with right side
  // columns set and left side columns as null. Possible only be hash or
  // merge.
  const bool leftOptional_;

  // True if a right side miss produces a row with left side columns
  // and a null for right side columns (left outer join). A full outer
  // join has both left and right optional.
  const bool rightOptional_;

  // True if the right side is only checked for existence of a match. If
  // rightOptional is set, this can project out a null for misses.
  const bool rightExists_;

  // True if produces a result for left if no match on the right.
  const bool rightNotExists_;

  // Flag to set if right side has a match.
  const ColumnCP markColumn_;

  // If directed non-outer edge. For example unnest or inner dependent on
  // optional of outer.
  bool directed_;
};

using JoinEdgeP = JoinEdge*;

using JoinEdgeVector = std::vector<JoinEdgeP, QGAllocator<JoinEdgeP>>;

/// Adds 'element' to 'vector' if it is not in it.
template <typename V, typename E>
inline void pushBackUnique(V& vector, E& element) {
  if (std::find(vector.begin(), vector.end(), element) == vector.end()) {
    vector.push_back(element);
  }
}

/// Represents a reference to a table from a query. The There is one of these
/// for each occurrence of the schema table. A TableScan references one
/// baseTable but the same BaseTable can be referenced from many TableScans, for
/// example if accessing different indices in a secondary to primary key lookup.
struct BaseTable : public PlanObject {
  BaseTable() : PlanObject(PlanType::kTable) {}

  // Correlation name, distinguishes between uses of the same schema table.
  Name cname{nullptr};

  SchemaTableCP schemaTable{nullptr};

  /// All columns referenced from 'schemaTable' under this correlation name.
  /// Different indices may have to be combined in different TableScans to cover
  /// 'columns'.
  ColumnVector columns;

  // All joins where 'this' is an end point.
  JoinEdgeVector joinedBy;

  // Top level conjuncts on single columns and literals, column to the left.
  ExprVector columnFilters;

  // Multicolumn filters dependent on 'this' alone.
  ExprVector filter;

  // the fraction of base table rows selected by all filters involving this
  // table only.
  float filterSelectivity{1};

  SubfieldSet controlSubfields;
  SubfieldSet payloadSubfields;

  bool isTable() const override {
    return true;
  }

  void addJoinedBy(JoinEdgeP join) {
    pushBackUnique(joinedBy, join);
  }

  /// Adds 'expr' to 'filters' or 'columnFilters'.
  void addFilter(ExprCP expr);

  std::optional<int32_t> columnId(Name column) const;

  BitSet columnSubfields(int32_t id, bool payloadOnly, bool controlOnly) const;

  std::string toString() const override;
};

using BaseTableCP = const BaseTable*;

using TypeVector =
    std::vector<const velox::Type*, QGAllocator<const velox::Type*>>;

// Aggregate function. The aggregation and arguments are in the
// inherited Call. The Value pertains to the aggregation
// result or accumulator.
class Aggregate : public Call {
 public:
  Aggregate(
      Name name,
      const Value& value,
      ExprVector args,
      FunctionSet functions,
      bool isDistinct,
      ExprCP condition,
      bool isAccumulator,
      const velox::Type* intermediateType)
      : Call(
            PlanType::kAggregate,
            name,
            value,
            std::move(args),
            functions | FunctionSet::kAggregate),
        isDistinct_(isDistinct),
        condition_(condition),
        isAccumulator_(isAccumulator),
        intermediateType_(intermediateType) {
    for (auto& arg : this->args()) {
      rawInputType_.push_back(arg->value().type);
    }
    if (condition_) {
      columns_.unionSet(condition_->columns());
    }
  }

  ExprCP condition() const {
    return condition_;
  }

  bool isDistinct() const {
    return isDistinct_;
  }

  bool isAccumulator() const {
    return isAccumulator_;
  }

  const velox::Type* intermediateType() const {
    return intermediateType_;
  }

  const TypeVector rawInputType() const {
    return rawInputType_;
  }

 private:
  bool isDistinct_;
  ExprCP condition_;
  bool isAccumulator_;
  const velox::Type* intermediateType_;
  TypeVector rawInputType_;
};

using AggregateCP = const Aggregate*;

struct Aggregation;
using AggregationP = Aggregation*;

/// Wraps an Aggregation RelationOp. This gives the aggregation a PlanObject id
struct AggregationPlan : public PlanObject {
  AggregationPlan(AggregationP agg)
      : PlanObject(PlanType::kAggregate), aggregation(agg) {}

  AggregationP aggregation;
};

using AggregationPlanCP = const AggregationPlan*;

struct OrderBy;
using OrderByP = OrderBy*;

/// Represents a derived table, i.e. a select in a from clause. This is the
/// basic unit of planning. Derived tables can be merged and split apart from
/// other ones. Join types and orders are decided within each derived table. A
/// derived table is likewise a reorderable unit inside its parent derived
/// table. Joins can move between derived tables within limits, considering the
/// semantics of e.g. group by.
struct DerivedTable : public PlanObject {
  DerivedTable() : PlanObject(PlanType::kDerivedTable) {}

  // Distribution that gives partition, cardinality and
  // order/uniqueness for the dt alone. This is expressed in terms of
  // outside visible 'columns'. Actual uses of the dt in candidate
  // plans may be modified from this by e.g. importing restrictions
  // from enclosing query. Set for a non-top level dt.
  Distribution* distribution{nullptr};

  // Correlation name.
  Name cname{nullptr};

  // Columns projected out. Visible in the enclosing query.
  ColumnVector columns;

  // Exprs projected out.1:1 to 'columns'.
  ExprVector exprs;

  // References all joins where 'this' is an end point.
  JoinEdgeVector joinedBy;

  // All tables in from, either Table or DerivedTable. If Table, all
  // filters resolvable with the table alone are in single column filters or
  // 'filter' of BaseTable.
  std::vector<PlanObjectCP, QGAllocator<PlanObjectCP>> tables;

  // Repeats the contents of 'tables'. Used for membership check. A DerivedTable
  // can be a subset of another, for example when planning a join for a build
  // side. In this case joins that refer to tables not in 'tableSet' are not
  // considered.
  PlanObjectSet tableSet;

  // Single row tables from non-correlated scalar subqueries.
  PlanObjectSet singleRowDts;

  // Tables that are not to the right sides of non-commutative joins.
  PlanObjectSet startTables;

  // Joins between 'tables'.
  JoinEdgeVector joins;

  // Filters in where for that are not single table expressions and not join
  // filters of explicit joins and not equalities between columns of joined
  // tables.
  ExprVector conjuncts;

  // Number of fully processed leading elements of 'conjuncts'.
  int32_t numCanonicalConjuncts{0};

  // Set of reducing joined tables imported to reduce build size. Set if 'this'
  // represents a build side join.
  PlanObjectSet importedExistences;

  // The set of tables in import() '_tables' that are fully covered by this dt
  // and need not be considered outside of it. If 'firstTable' in import is a
  // group by dt, for example, some joins may be imported as reducing existences
  // but will still have to be considered by the enclosing query. Such tables
  // are not included in 'fullyImported' If 'firstTable' in import is a base
  // table, then 'fullyImported' is '_tables'.
  PlanObjectSet fullyImported;

  //
  // True if this dt is already a reducing join imported to a build side. Do not
  // try to further restrict this with probe side.
  bool noImportOfExists{false};
  // Postprocessing clauses, group by, having, order by, limit, offset.
  AggregationPlanCP aggregation{nullptr};
  ExprVector having;
  OrderByP orderBy{nullptr};
  int32_t limit{-1};
  int32_t offset{0};

  /// Adds an equijoin edge between 'left' and 'right'. The flags correspond to
  /// the like-named members in Join.
  void addJoinEquality(
      ExprCP left,
      ExprCP right,
      const ExprVector& filter,
      bool leftOptional,
      bool rightOptional,
      bool rightExists,
      bool rightNotExists);

  // after 'joins' is filled in, links tables to their direct and
  // equivalence-implied joins.
  void linkTablesToJoins();

  /// Completes 'joins' with edges implied by column equivalences.
  void addImpliedJoins();

  /// Extracts implied conjuncts and removes duplicates from
  /// 'conjuncts' and updates 'conjuncts'. Extracted conjuncts may
  /// allow extra pushdown or allow create join edges. May be called
  /// repeatedly, each e.g. after pushing down conjuncts from outer
  /// DTs.
  void expandConjuncts();

  /// Initializes 'this' to join 'tables' from 'super'. Adds the joins from
  /// 'existences' as semijoins to limit cardinality when making a hash join
  /// build side. Allows importing a reducing join from probe to build.
  /// 'firstTable' is the joined table that is restricted by the other tables in
  /// 'tables' and 'existences'. 'existsFanout' us the reduction from joining
  /// 'firstTable' with 'existences'.
  void import(
      const DerivedTable& super,
      PlanObjectCP firstTable,
      const PlanObjectSet& tables,
      const std::vector<PlanObjectSet>& existences,
      float existsFanout = 1);

  bool isTable() const override {
    return true;
  }

  //// True if 'table' is of 'this'.
  bool hasTable(PlanObjectCP table) const {
    return std::find(tables.begin(), tables.end(), table) != tables.end();
  }

  // True if 'join' exists in 'this'. Tables link to joins that may be
  // in different speculative candidate dts. So only consider joins
  // inside the current dt wen planning.
  bool hasJoin(JoinEdgeP join) const {
    return std::find(joins.begin(), joins.end(), join) != joins.end();
  }

  /// Fills in 'startTables_' to 'tables_' that are not to the right of
  /// non-commutative joins.
  void setStartTables();

  std::string toString() const override;
  void addJoinedBy(JoinEdgeP join) {
    pushBackUnique(joinedBy, join);
  }

  /// Moves suitable elements of 'conjuncts' into join edges or single
  /// table filters. May be called repeatedly if enclosing dt's add
  /// more conjuncts. May call itself recursively on component dts.
  void distributeConjuncts();

  /// memoizes plans for 'this' and fills in 'distribution_'. Needed
  /// before adding 'this' as a join side because join sides must have
  /// a cardinality guess.
  void makeInitialPlan();

 private:
  // Imports the joins in 'this' inside 'firstDt', which must be a
  // member of 'this'. The import is possible if the join is not
  // through aggregates in 'firstDt'. On return, all joins that can go
  // inside firstDt are imported below aggregation in
  // firstDt. 'firstDt' is not modified, its original contents are
  // copied in a new dt before the import.
  void importJoinsIntoFirstDt(const DerivedTable* firstDt);

  // Sets 'dt' to be the complete contents of 'this'.
  void flattenDt(const DerivedTable* dt);

  // Finds single row dts from non-correlated scalar subqueries.
  void findSingleRowDts();

  // Sets 'columns' and 'exprs'.
  void makeProjection(const ExprVector& exprs);
};

using DerivedTableP = DerivedTable*;

float tableCardinality(PlanObjectCP table);

/// Returns all distinct tables 'exprs' depend on.
PlanObjectSet allTables(CPSpan<Expr> exprs);

/// Fills 'leftKeys' and 'rightKeys's from 'conjuncts' so that
/// equalities with one side only depending on 'right' go to
/// 'rightKeys' and the other side not depending on 'right' goes to
/// 'leftKeys'. The left side may depend on more than one table. The
/// tables 'leftKeys' depend on are returned in 'allLeft'. The
/// conjuncts that are not equalities or have both sides depending
/// on right and something else are left in 'conjuncts'.
void extractNonInnerJoinEqualities(
    ExprVector& conjuncts,
    PlanObjectCP right,
    ExprVector& leftKeys,
    ExprVector& rightKeys,
    PlanObjectSet& allLeft);

/// Appends the string representation of 'exprs' to 'out'.
void exprsToString(const ExprVector& exprs, std::stringstream& out);

} // namespace facebook::velox::optimizer
