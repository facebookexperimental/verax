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

#include "axiom/logical_plan/LogicalPlanNode.h"
#include "axiom/optimizer/PlanObject.h"

namespace facebook::velox::optimizer {

struct Distribution;

class JoinEdge;
using JoinEdgeP = JoinEdge*;
using JoinEdgeVector = std::vector<JoinEdge*, QGAllocator<JoinEdge*>>;

struct AggregationPlan;
using AggregationPlanCP = const AggregationPlan*;

struct OrderBy;
using OrderByP = OrderBy*;

/// Represents a derived table, i.e. a SELECT in a FROM clause. This is the
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

  // Exprs projected out. 1:1 to 'columns'.
  ExprVector exprs;

  // References all joins where 'this' is an end point.
  JoinEdgeVector joinedBy;

  // All tables in FROM, either Table or DerivedTable. If Table, all
  // filters resolvable with the table alone are in single column filters or
  // 'filter' of BaseTable.
  std::vector<PlanObjectCP, QGAllocator<PlanObjectCP>> tables;

  // Repeats the contents of 'tables'. Used for membership check. A DerivedTable
  // can be a subset of another, for example when planning a join for a build
  // side. In this case joins that refer to tables not in 'tableSet' are not
  // considered.
  PlanObjectSet tableSet;

  // Set if this is a set operation. If set, 'children' has the operands.
  std::optional<logical_plan::SetOperation> setOp;

  /// Operands if 'this' is a set operation, e.g. union.
  std::vector<DerivedTable*, QGAllocator<DerivedTable*>> children;

  // Single row tables from non-correlated scalar subqueries.
  PlanObjectSet singleRowDts;

  // Tables that are not to the right sides of non-commutative joins.
  PlanObjectSet startTables;

  // Joins between 'tables'.
  JoinEdgeVector joins;

  // Filters in WHERE that are not single table expressions and not join
  // conditions of explicit joins and not equalities between columns of joined
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

  void addJoinedBy(JoinEdgeP join);

  /// Moves suitable elements of 'conjuncts' into join edges or single
  /// table filters. May be called repeatedly if enclosing dt's add
  /// more conjuncts. May call itself recursively on component dts.
  void distributeConjuncts();

  /// memoizes plans for 'this' and fills in 'distribution_'. Needed
  /// before adding 'this' as a join side because join sides must have
  /// a cardinality guess.
  void makeInitialPlan();

  std::string toString() const override;

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
using DerivedTableCP = const DerivedTable*;

} // namespace facebook::velox::optimizer
