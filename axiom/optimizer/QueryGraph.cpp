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

#include "axiom/optimizer/QueryGraph.h"
#include "axiom/optimizer/Plan.h"
#include "axiom/optimizer/PlanUtils.h"
#include "velox/expression/ScopedVarSetter.h"

namespace facebook::velox::optimizer {

void Column::equals(ColumnCP other) const {
  if (!equivalence_ && !other->equivalence_) {
    auto* equiv = make<Equivalence>();
    equiv->columns.push_back(this);
    equiv->columns.push_back(other);
    equivalence_ = equiv;
    other->equivalence_ = equiv;
    return;
  }
  if (!other->equivalence_) {
    other->equivalence_ = equivalence_;
    equivalence_->columns.push_back(other);
    return;
  }
  if (!equivalence_) {
    other->equals(this);
    return;
  }
  for (auto& column : other->equivalence_->columns) {
    equivalence_->columns.push_back(column);
    column->equivalence_ = equivalence_;
  }
}

std::string Column::toString() const {
  auto* opt = queryCtx()->optimization();
  if (!-opt->cnamesInExpr()) {
    return name_;
  }
  Name cname = !relation_ ? ""
      : relation_->type() == PlanType::kTable
      ? relation_->as<BaseTable>()->cname
      : relation_->type() == PlanType::kDerivedTable
      ? relation_->as<DerivedTable>()->cname
      : "--";

  // Map corre;correlation names to canonical if making keys for history pieces.
  auto canonical = opt->canonicalCnames();
  if (canonical) {
    auto it = canonical->find(cname);
    if (it != canonical->end()) {
      cname = it->second;
    }
  }
  return fmt::format("{}.{}", cname, name_);
}

std::string Literal::toString() const {
  std::stringstream out;
  if (vector_) {
    out << vector_->toString(0);
  } else {
    out << *literal_;
  }
  return out.str();
}

std::string Call::toString() const {
  std::stringstream out;
  out << name_ << "(";
  for (auto i = 0; i < args_.size(); ++i) {
    out << args_[i]->toString() << (i == args_.size() - 1 ? ")" : ", ");
  }
  return out.str();
}

std::string Field::toString() const {
  std::stringstream out;
  out << base_->toString() << ".";
  if (field_) {
    out << field_;
  } else {
    out << fmt::format("{}", index_);
  }
  return out.str();
}

std::string conjunctsToString(const ExprVector& conjuncts) {
  std::stringstream out;
  for (auto i = 0; i < conjuncts.size(); ++i) {
    out << conjuncts[i]->toString()
        << (i == conjuncts.size() - 1 ? "" : " and ");
  }
  return out.str();
}

std::optional<BitSet> SubfieldSet::findSubfields(int32_t id) const {
  for (auto i = 0; i < ids.size(); ++i) {
    if (ids[i] == id) {
      return subfields[i];
    }
  }
  return std::nullopt;
}

std::optional<int32_t> BaseTable::columnId(Name column) const {
  for (auto i = 0; i < columns.size(); ++i) {
    if (columns[i]->name() == column) {
      return columns[i]->id();
    }
  }
  return std::nullopt;
}

BitSet BaseTable::columnSubfields(
    int32_t id,
    bool controlOnly,
    bool payloadOnly) const {
  BitSet subfields;
  if (!controlOnly) {
    auto maybe = payloadSubfields.findSubfields(id);
    if (maybe.has_value()) {
      subfields = maybe.value();
    }
  }
  if (!payloadOnly) {
    auto maybe = controlSubfields.findSubfields(id);
    if (maybe.has_value()) {
      subfields.unionSet(maybe.value());
    }
  }
  Path::subfieldSkyline(subfields);
  return subfields;
}

std::string BaseTable::toString() const {
  std::stringstream out;
  out << "{" << PlanObject::toString();
  out << schemaTable->name << " " << cname << "}";
  return out.str();
}

const JoinSide JoinEdge::sideOf(PlanObjectCP side, bool other) const {
  if ((side == rightTable_ && !other) || (side == leftTable_ && other)) {
    return {
        rightTable_,
        rightKeys_,
        lrFanout_,
        rightOptional_,
        leftOptional_,
        rightExists_,
        rightNotExists_,
        markColumn_,
        rightUnique_};
  }
  return {
      leftTable_,
      leftKeys_,
      rlFanout_,
      leftOptional_,
      rightOptional_,
      false,
      false,
      nullptr,
      leftUnique_};
}

bool JoinEdge::isBroadcastableType() const {
  return !leftOptional_;
}

void JoinEdge::addEquality(ExprCP left, ExprCP right, bool update) {
  leftKeys_.push_back(left);
  rightKeys_.push_back(right);
  if (update) {
    guessFanout();
  }
}

std::pair<std::string, bool> JoinEdge::sampleKey() const {
  std::stringstream out;
  if (!leftTable_ || leftTable_->type() != PlanType::kTable ||
      rightTable_->type() != PlanType::kTable) {
    return std::make_pair("", false);
  }
  auto* opt = queryCtx()->optimization();
  ScopedVarSetter pref(&opt->cnamesInExpr(), false);
  std::vector<int32_t> indices(leftKeys_.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<std::string> leftString;
  for (auto& k : leftKeys_) {
    leftString.push_back(k->toString());
  }
  std::sort(indices.begin(), indices.end(), [&](int32_t l, int32_t r) {
    return leftString[l] < leftString[r];
  });
  auto left =
      fmt::format("{} ", leftTable_->as<BaseTable>()->schemaTable->name);
  auto right =
      fmt::format("{} ", rightTable_->as<BaseTable>()->schemaTable->name);
  for (auto i : indices) {
    left += leftKeys_[i]->toString() + " ";
    right += rightKeys_[i]->toString() + " ";
  }
  if (left < right) {
    return std::make_pair(left + " " + right, false);
  }
  return std::make_pair(right + " " + left, true);
}

std::string JoinEdge::toString() const {
  std::stringstream out;
  out << "<join "
      << (leftTable_ ? leftTable_->toString() : " multiple tables ");
  if (leftOptional_ && rightOptional_) {
    out << " full outr ";
  } else if (markColumn_) {
    out << " exists project ";
  } else if (rightOptional_) {
    out << " left";
  } else if (rightExists_) {
    out << " exists ";
  } else if (rightNotExists_) {
    out << " not exists ";
  } else if (leftOptional_) {
    out << "right";
  } else {
    out << " inner ";
  }
  out << rightTable_->toString();
  out << " on ";
  for (auto i = 0; i < leftKeys_.size(); ++i) {
    out << leftKeys_[i]->toString() << " = " << rightKeys_[i]->toString()
        << (i < leftKeys_.size() - 1 ? " and " : "");
  }
  if (!filter_.empty()) {
    out << " filter " << conjunctsToString(filter_);
  }
  out << ">";
  return out.str();
}

const FunctionSet& Expr::functions() const {
  static FunctionSet empty;
  return empty;
}

bool Expr::sameOrEqual(const Expr& other) const {
  if (this == &other) {
    return true;
  }
  if (type() != other.type()) {
    return false;
  }
  switch (type()) {
    case PlanType::kColumn:
      return as<Column>()->equivalence() &&
          as<Column>()->equivalence() == other.as<Column>()->equivalence();
    case PlanType::kAggregate: {
      auto a = reinterpret_cast<const Aggregate*>(this);
      auto b = reinterpret_cast<const Aggregate*>(&other);
      if (a->isDistinct() != b->isDistinct() ||
          a->isAccumulator() != b->isAccumulator() ||
          !(a->condition() == b->condition() ||
            (a->condition() && b->condition() &&
             a->condition()->sameOrEqual(*b->condition())))) {
        return false;
      }
    }
      // Fall through.
      FMT_FALLTHROUGH;
    case PlanType::kCall: {
      if (as<Call>()->name() != other.as<Call>()->name()) {
        return false;
      }
      auto numArgs = as<Call>()->args().size();
      if (numArgs != other.as<Call>()->args().size()) {
        return false;
      }
      for (auto i = 0; i < numArgs; ++i) {
        if (as<Call>()->args()[i]->sameOrEqual(*other.as<Call>()->args()[i])) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

PlanObjectCP singleTable(PlanObjectCP object) {
  if (isExprType(object->type())) {
    return object->as<Expr>()->singleTable();
  }
  return nullptr;
}

PlanObjectCP Expr::singleTable() const {
  if (type() == PlanType::kColumn) {
    return as<Column>()->relation();
  }
  PlanObjectCP table = nullptr;
  bool multiple = false;
  columns_.forEach([&](PlanObjectCP object) {
    VELOX_CHECK(object->type() == PlanType::kColumn);
    if (!table) {
      table = object->template as<Column>()->relation();
    } else if (table != object->as<Column>()->relation()) {
      multiple = true;
    }
  });
  return multiple ? nullptr : table;
}

PlanObjectSet Expr::allTables() const {
  PlanObjectSet set;
  columns_.forEach(
      [&](PlanObjectCP object) { set.add(object->as<Column>()->relation()); });
  return set;
}

PlanObjectSet allTables(CPSpan<Expr> exprs) {
  PlanObjectSet all;
  for (auto expr : exprs) {
    auto set = expr->allTables();
    all.unionSet(set);
  }
  return all;
}

Column::Column(
    Name name,
    PlanObjectP relation,
    const Value& value,
    Name nameInTable,
    ColumnCP top,
    PathCP path)
    : Expr(PlanType::kColumn, value),
      name_(name),
      relation_(relation),
      topColumn_(top),
      path_(path) {
  columns_.add(this);
  subexpressions_.add(this);
  if (relation_ && relation_->type() == PlanType::kTable) {
    if (topColumn_) {
      schemaColumn_ = topColumn_->schemaColumn_;
    } else {
      schemaColumn_ = relation->as<BaseTable>()->schemaTable->findColumn(
          nameInTable ? nameInTable : name_);
      VELOX_CHECK(schemaColumn_);
    }
  }
}

void DerivedTable::addJoinEquality(
    ExprCP left,
    ExprCP right,
    const ExprVector& filter,
    bool leftOptional,
    bool rightOptional,
    bool rightExists,
    bool rightNotExists) {
  auto leftTable = singleTable(left);
  auto rightTable = singleTable(right);
  for (auto& join : joins) {
    if (join->leftTable() == leftTable && join->rightTable() == rightTable) {
      join->addEquality(left, right);
      return;
    } else if (
        join->rightTable() == leftTable && join->leftTable() == rightTable) {
      join->addEquality(right, left);
      return;
    }
  }
  auto* join = make<JoinEdge>(
      leftTable,
      rightTable,
      filter,
      leftOptional,
      rightOptional,
      rightExists,
      rightNotExists);
  join->addEquality(left, right);
  joins.push_back(join);
}

using EdgeSet = std::unordered_set<std::pair<int32_t, int32_t>>;

bool hasEdge(const EdgeSet& edges, int32_t id1, int32_t id2) {
  if (id1 == id2) {
    return true;
  }
  auto it = edges.find(
      id1 > id2 ? std::pair<int32_t, int32_t>(id2, id1)
                : std::pair<int32_t, int32_t>(id1, id2));
  return it != edges.end();
}

void addEdge(EdgeSet& edges, int32_t id1, int32_t id2) {
  if (id1 > id2) {
    edges.insert(std::pair<int32_t, int32_t>(id2, id1));
  } else {
    edges.insert(std::pair<int32_t, int32_t>(id1, id2));
  }
}

void fillJoins(
    PlanObjectCP column,
    const Equivalence& equivalence,
    EdgeSet& edges,
    DerivedTableP dt) {
  for (auto& other : equivalence.columns) {
    if (!hasEdge(edges, column->id(), other->id())) {
      addEdge(edges, column->id(), other->id());
      dt->addJoinEquality(
          column->as<Column>(),
          other->as<Column>(),
          {},
          false,
          false,
          false,
          false);
    }
  }
}

void DerivedTable::addImpliedJoins() {
  EdgeSet edges;
  for (auto& join : joins) {
    if (join->isInner()) {
      for (auto i = 0; i < join->leftKeys().size(); ++i) {
        if (join->leftKeys()[i]->type() == PlanType::kColumn &&
            join->rightKeys()[i]->type() == PlanType::kColumn) {
          addEdge(edges, join->leftKeys()[i]->id(), join->rightKeys()[i]->id());
        }
      }
    }
  }
  // The loop appends to 'joins', so loop over a copy.
  JoinEdgeVector joinsCopy = joins;
  for (auto& join : joinsCopy) {
    if (join->isInner()) {
      for (auto i = 0; i < join->leftKeys().size(); ++i) {
        if (join->leftKeys()[i]->type() == PlanType::kColumn &&
            join->rightKeys()[i]->type() == PlanType::kColumn) {
          auto leftEq = join->leftKeys()[i]->as<Column>()->equivalence();
          auto rightEq = join->rightKeys()[i]->as<Column>()->equivalence();
          if (rightEq && leftEq) {
            for (auto& left : leftEq->columns) {
              fillJoins(left, *rightEq, edges, this);
            }
          } else if (leftEq) {
            fillJoins(join->rightKeys()[i], *leftEq, edges, this);
          } else if (rightEq) {
            fillJoins(join->leftKeys()[i], *rightEq, edges, this);
          }
        }
      }
    }
  }
}

void DerivedTable::setStartTables() {
  findSingleRowDts();
  startTables = tableSet;
  startTables.except(singleRowDts);
  for (auto join : joins) {
    if (join->isNonCommutative()) {
      startTables.erase(join->rightTable());
    }
  }
}

bool isSingleRowDt(PlanObjectCP object) {
  if (object->type() == PlanType::kDerivedTable) {
    auto dt = object->as<DerivedTable>();
    return dt->limit == 1 ||
        (dt->aggregation && dt->aggregation->aggregation->grouping.empty());
  }
  return false;
}

void DerivedTable::findSingleRowDts() {
  auto tablesCopy = tableSet;
  int32_t numSingle = 0;
  for (auto& join : joins) {
    tablesCopy.erase(join->rightTable());
    for (auto& key : join->leftKeys()) {
      tablesCopy.except(key->allTables());
    }
    for (auto& filter : join->filter()) {
      tablesCopy.except(filter->allTables());
    }
  }
  tablesCopy.forEach([&](PlanObjectCP object) {
    if (isSingleRowDt(object)) {
      ++numSingle;
      singleRowDts.add(object);
    }
  });
  // if everything is a single row dt, then process tese as cross products and
  // not as placed with filters.
  if (numSingle == tables.size()) {
    singleRowDts = PlanObjectSet();
  }
}

void DerivedTable::linkTablesToJoins() {
  setStartTables();

  // All tables directly mentioned by a join link to the join. A non-inner
  // that depends on multiple left tables has no leftTable but is still linked
  // from all the tables it depends on.
  for (auto join : joins) {
    PlanObjectSet tables;
    for (auto key : join->leftKeys()) {
      tables.unionSet(key->allTables());
    }
    for (auto key : join->rightKeys()) {
      tables.unionSet(key->allTables());
    }
    if (!join->filter().empty()) {
      for (auto& conjunct : join->filter()) {
        tables.unionSet(conjunct->allTables());
      }
    }
    tables.forEachMutable([&](PlanObjectP table) {
      if (table->type() == PlanType::kTable) {
        table->as<BaseTable>()->addJoinedBy(join);
      } else {
        VELOX_CHECK(table->type() == PlanType::kDerivedTable);
        table->as<DerivedTable>()->addJoinedBy(join);
      }
    });
  }
}

// Returns a right exists (semijoin) with 'table' on the left and one of
// 'tables' on the right.
JoinEdgeP makeExists(PlanObjectCP table, const PlanObjectSet& tables) {
  for (auto join : joinedBy(table)) {
    if (join->leftTable() == table) {
      if (!tables.contains(join->rightTable())) {
        continue;
      }
      auto* exists = new (queryCtx()->allocate(sizeof(JoinEdge)))
          JoinEdge(table, join->rightTable(), {}, false, false, true, false);
      for (auto i = 0; i < join->leftKeys().size(); ++i) {
        exists->addEquality(join->leftKeys()[i], join->rightKeys()[i]);
      }
      return exists;
    }
    if (join->rightTable() == table) {
      if (!join->leftTable() || !tables.contains(join->leftTable())) {
        continue;
      }

      auto* exists = make<JoinEdge>(
          table, join->leftTable(), ExprVector{}, false, false, true, false);
      for (auto i = 0; i < join->leftKeys().size(); ++i) {
        exists->addEquality(join->rightKeys()[i], join->leftKeys()[i]);
      }
      return exists;
    }
  }
  VELOX_UNREACHABLE("No join to make an exists build side restriction");
}

std::pair<DerivedTableP, JoinEdgeP> makeExistsDtAndJoin(
    const DerivedTable& super,
    PlanObjectCP firstTable,
    float existsFanout,
    PlanObjectVector& existsTables,
    JoinEdgeP existsJoin) {
  auto firstExistsTable = existsJoin->rightKeys()[0]->singleTable();
  VELOX_CHECK(firstExistsTable);
  MemoKey existsDtKey;
  existsDtKey.firstTable = firstExistsTable;
  for (auto& column : existsJoin->rightKeys()) {
    existsDtKey.columns.unionColumns(column);
  }
  auto optimization = queryCtx()->optimization();
  existsDtKey.tables.unionObjects(existsTables);
  auto it = optimization->existenceDts().find(existsDtKey);
  DerivedTableP existsDt;
  if (it == optimization->existenceDts().end()) {
    auto* newDt = make<DerivedTable>();
    existsDt = newDt;
    existsDt->cname = queryCtx()->optimization()->newCName("edt");
    existsDt->import(super, firstExistsTable, existsDtKey.tables, {});
    for (auto& k : existsJoin->rightKeys()) {
      auto* existsColumn = make<Column>(
          toName(fmt::format("{}.{}", existsDt->cname, k->toString())),
          existsDt,
          k->value());
      existsDt->columns.push_back(existsColumn);
      existsDt->exprs.push_back(k);
    }
    existsDt->noImportOfExists = true;
    existsDt->makeInitialPlan();
    optimization->existenceDts()[existsDtKey] = existsDt;
  } else {
    existsDt = it->second;
  }
  auto* joinWithDt = make<JoinEdge>(
      firstTable, existsDt, ExprVector{}, false, false, true, false);
  joinWithDt->setFanouts(existsFanout, 1);
  for (auto i = 0; i < existsJoin->leftKeys().size(); ++i) {
    joinWithDt->addEquality(existsJoin->leftKeys()[i], existsDt->columns[i]);
  }
  return std::make_pair(existsDt, joinWithDt);
}

void DerivedTable::import(
    const DerivedTable& super,
    PlanObjectCP firstTable,
    const PlanObjectSet& _tables,
    const std::vector<PlanObjectSet>& existences,
    float existsFanout) {
  tableSet = _tables;
  _tables.forEach([&](auto table) { tables.push_back(table); });
  for (auto join : super.joins) {
    if (_tables.contains(join->rightTable()) && join->leftTable() &&
        _tables.contains(join->leftTable())) {
      joins.push_back(join);
    }
  }
  for (auto& exists : existences) {
    // We filter the derived table by importing reducing semijoins.
    // These are based on joins on the outer query but become
    // existences so as not to change cardinality. The reducing join
    // is against one or more tables. If more than one table, the join
    // of these tables goes into its own derived table which is joined
    // with exists to the main table(s) in the 'this'.
    importedExistences.unionSet(exists);
    PlanObjectVector existsTables;
    exists.forEach([&](auto object) { existsTables.push_back(object); });
    auto existsJoin = makeExists(firstTable, exists);
    if (existsTables.size() > 1) {
      // There is a join on the right of exists. Needs its own dt.
      auto [existsDt, joinWithDt] = makeExistsDtAndJoin(
          super, firstTable, existsFanout, existsTables, existsJoin);
      joins.push_back(joinWithDt);
      tables.push_back(existsDt);
      tableSet.add(existsDt);
      noImportOfExists = true;
    } else {
      joins.push_back(existsJoin);
      assert(!existsTables.empty());
      tables.push_back(existsTables[0]);
      tableSet.add(existsTables[0]);
      noImportOfExists = true;
    }
  }
  if (firstTable->type() == PlanType::kDerivedTable) {
    importJoinsIntoFirstDt(firstTable->as<DerivedTable>());
  } else {
    fullyImported = _tables;
  }
  linkTablesToJoins();
}

// Returns a copy of 'expr,, replacing instances of columns in 'outer' with the
// corresponding expression from 'inner'
ExprCP
importExpr(ExprCP expr, const ColumnVector& outer, const ExprVector& inner) {
  if (!expr) {
    return nullptr;
  }
  switch (expr->type()) {
    case PlanType::kColumn:
      for (auto i = 0; i < inner.size(); ++i) {
        if (outer[i] == expr) {
          return inner[i];
        }
      }
      return expr;
    case PlanType::kLiteral:
      return expr;
    case PlanType::kCall:
    case PlanType::kAggregate: {
      auto children = expr->children();
      std::vector<ExprCP> newChildren(children.size());
      FunctionSet functions;
      bool anyChange = false;
      for (auto i = 0; i < children.size(); ++i) {
        newChildren[i] = importExpr(children[i]->as<Expr>(), outer, inner);
        anyChange |= newChildren[i] != children[i];
        if (newChildren[i]->isFunction()) {
          functions = functions | newChildren[i]->as<Call>()->functions();
        }
      }
      ExprCP newCondition = nullptr;
      if (expr->type() == PlanType::kAggregate) {
        newCondition =
            importExpr(expr->as<Aggregate>()->condition(), outer, inner);
        anyChange |= newCondition != expr->as<Aggregate>()->condition();

        if (newCondition && newCondition->isFunction()) {
          functions = functions | newCondition->as<Call>()->functions();
        }
      }
      if (!anyChange) {
        return expr;
      }
      ExprVector childVector;
      childVector.insert(
          childVector.begin(), newChildren.begin(), newChildren.end());
      if (expr->type() == PlanType::kCall) {
        auto call = expr->as<Call>();
        auto* copy = make<Call>(
            call->name(), call->value(), std::move(childVector), functions);
        return copy;
      } else if (expr->type() == PlanType::kAggregate) {
        auto aggregate = expr->as<Aggregate>();
        auto* copy = make<Aggregate>(
            aggregate->name(),
            aggregate->value(),
            std::move(childVector),
            functions,
            aggregate->isDistinct(),
            newCondition,
            aggregate->isAccumulator(),
            aggregate->intermediateType());
        return copy;
      }
    }
      FMT_FALLTHROUGH;
    default:
      VELOX_UNREACHABLE();
  }
}

PlanObjectCP otherSide(JoinEdgeP join, PlanObjectCP side) {
  if (side == join->leftTable()) {
    return join->rightTable();
  } else if (join->rightTable() == side) {
    return join->leftTable();
  }
  return nullptr;
}

bool isProjected(PlanObjectCP table, const PlanObjectSet& columns) {
  bool projected = false;
  columns.forEach([&](PlanObjectCP column) {
    projected |= column->as<Column>()->relation() == table;
  });
  return projected;
}

// True if 'join'  has max 1 match for a row of 'side'.
bool isUnique(JoinEdgeP join, PlanObjectCP side) {
  return join->sideOf(side, true).isUnique;
}

// Returns a join partner of 'startin 'joins' ' where the partner is
// not in 'visited' Sets 'isFullyImported' to false if the partner is
// not guaranteed n:1 reducing or has columns that are projected out.
PlanObjectCP nextJoin(
    PlanObjectCP start,
    const JoinEdgeVector& joins,
    const PlanObjectSet& columns,
    const PlanObjectSet& visited,
    bool& fullyImported) {
  for (auto& join : joins) {
    auto other = otherSide(join, start);
    if (!other) {
      continue;
    }
    if (visited.contains(other)) {
      continue;
    }
    if (!isUnique(join, other) || isProjected(other, columns)) {
      fullyImported = false;
    }
    return other;
  }
  return nullptr;
}

void joinChain(
    PlanObjectCP start,
    const JoinEdgeVector& joins,
    const PlanObjectSet& columns,
    PlanObjectSet visited,
    bool& fullyImported,
    std::vector<PlanObjectCP>& path) {
  auto next = nextJoin(start, joins, columns, visited, fullyImported);
  if (!next) {
    return;
  }
  visited.add(next);
  path.push_back(next);
  joinChain(next, joins, columns, visited, fullyImported, path);
}

JoinEdgeP importedJoin(
    JoinEdgeP join,
    PlanObjectCP other,
    ExprCP innerKey,
    bool fullyImported) {
  auto left = singleTable(innerKey);
  VELOX_CHECK(left);
  auto otherKey = join->sideOf(other).keys[0];
  auto* newJoin = make<JoinEdge>(
      left, other, ExprVector{}, false, false, !fullyImported, false);
  newJoin->addEquality(innerKey, otherKey);
  return newJoin;
}

JoinEdgeP importedDtJoin(
    JoinEdgeP join,
    DerivedTableP dt,
    ExprCP innerKey,
    bool fullyImported) {
  auto left = singleTable(innerKey);
  VELOX_CHECK(left);
  auto otherKey = dt->columns[0];
  auto* newJoin = make<JoinEdge>(
      left, dt, ExprVector{}, false, false, !fullyImported, false);
  newJoin->addEquality(innerKey, otherKey);
  return newJoin;
}

template <typename V, typename E>
void eraseFirst(V& set, E element) {
  auto it = std::find(set.begin(), set.end(), element);
  if (it != set.end()) {
    set.erase(it);
  } else {
    LOG(INFO) << "suspect erase";
  }
}

void DerivedTable::makeProjection(const ExprVector& exprs) {
  auto optimization = queryCtx()->optimization();
  for (auto& expr : exprs) {
    auto* column =
        make<Column>(optimization->newCName("ec"), this, expr->value());
    columns.push_back(column);
    this->exprs.push_back(expr);
  }
}

void DerivedTable::importJoinsIntoFirstDt(const DerivedTable* firstDt) {
  if (tables.size() == 1 && tables[0]->type() == PlanType::kDerivedTable) {
    flattenDt(tables[0]->as<DerivedTable>());
    return;
  }
  auto initialTables = tables;
  if (firstDt->limit != -1 || firstDt->orderBy) {
    // tables can't be imported but are marked as used so not tried again.
    for (auto i = 1; i < tables.size(); ++i) {
      importedExistences.add(tables[i]);
    }
    return;
  }
  auto& outer = firstDt->columns;
  auto& inner = firstDt->exprs;
  PlanObjectSet projected;
  for (auto& expr : exprs) {
    projected.unionColumns(expr);
  }

  auto* newFirst = make<DerivedTable>(*firstDt->as<DerivedTable>());
  newFirst->cname = firstDt->as<DerivedTable>()->cname;
  for (auto& join : joins) {
    auto other = otherSide(join, firstDt);
    if (!other) {
      continue;
    }
    if (!tableSet.contains(other)) {
      // Already placed in some previous join chain.
      continue;
    }
    auto side = join->sideOf(firstDt);
    if (side.keys.size() > 1 || !join->filter().empty()) {
      continue;
    }
    auto innerKey = importExpr(side.keys[0], outer, inner);
    assert(innerKey);
    if (innerKey->containsFunction(FunctionSet::kAggregate)) {
      // If the join key is an aggregate, the join can't be moved below the agg.
      continue;
    }
    auto otherSide = join->sideOf(firstDt, true);
    PlanObjectSet visited;
    visited.add(firstDt);
    visited.add(other);
    std::vector<PlanObjectCP> path;
    bool fullyImported = otherSide.isUnique;
    joinChain(other, joins, projected, visited, fullyImported, path);
    if (path.empty()) {
      if (other->type() == PlanType::kDerivedTable) {
        const_cast<PlanObject*>(other)->as<DerivedTable>()->makeInitialPlan();
      }

      newFirst->tables.push_back(other);
      newFirst->tableSet.add(other);
      newFirst->joins.push_back(
          importedJoin(join, other, innerKey, fullyImported));
      if (fullyImported) {
        newFirst->fullyImported.add(other);
      }
    } else {
      auto* chainDt = make<DerivedTable>();
      PlanObjectSet chainSet;
      chainSet.add(other);
      if (fullyImported) {
        newFirst->fullyImported.add(other);
      }
      for (auto& object : path) {
        chainSet.add(object);
        if (fullyImported) {
          newFirst->fullyImported.add(object);
        }
      }
      chainDt->makeProjection(otherSide.keys);
      chainDt->import(*this, other, chainSet, {}, 1);
      chainDt->makeInitialPlan();
      newFirst->tables.push_back(chainDt);
      newFirst->tableSet.add(chainDt);
      newFirst->joins.push_back(
          importedDtJoin(join, chainDt, innerKey, fullyImported));
    }
    eraseFirst(tables, other);
    tableSet.erase(other);
    for (auto& table : path) {
      eraseFirst(tables, table);
      tableSet.erase(table);
    }
  }

  VELOX_CHECK_EQ(tables.size(), 1);
  for (auto i = 0; i < initialTables.size(); ++i) {
    if (!newFirst->fullyImported.contains(initialTables[i])) {
      newFirst->importedExistences.add(initialTables[i]);
    }
  }
  tables[0] = newFirst;
  flattenDt(newFirst);
}

void DerivedTable::flattenDt(const DerivedTable* dt) {
  tables = dt->tables;
  cname = dt->cname;
  tableSet = dt->tableSet;
  joins = dt->joins;
  columns = dt->columns;
  exprs = dt->exprs;
  fullyImported = dt->fullyImported;
  importedExistences.unionSet(dt->importedExistences);
  aggregation = dt->aggregation;
  having = dt->having;
}

void BaseTable::addFilter(ExprCP expr) {
  const auto& columns = expr->columns();
  bool isMultiColumn = false;
  bool isSingleColumn = false;
  columns.forEach([&](PlanObjectCP object) {
    if (!isMultiColumn) {
      if (isSingleColumn) {
        isMultiColumn = true;
      } else {
        isSingleColumn = true;
      }
    };
  });
  if (isSingleColumn) {
    columnFilters.push_back(expr);
    filterUpdated(this);
    return;
  }
  filter.push_back(expr);
  filterUpdated(this);
}

// Finds a JoinEdge between tables[0] and tables[1]. Sets tables[0] to the left
// and [1] to the right table of the found join. Returns the JoinEdge. If
// 'create' is true and no edge is found, makes a new edge with tables[0] as
// left and [1] as right.
JoinEdgeP
findJoin(DerivedTableP dt, std::vector<PlanObjectP>& tables, bool create) {
  for (auto& join : dt->joins) {
    if (join->leftTable() == tables[0] && join->rightTable() == tables[1]) {
      return join;
    }
    if (join->leftTable() == tables[1] && join->rightTable() == tables[0]) {
      std::swap(tables[0], tables[1]);
      return join;
    }
  }
  if (create) {
    auto* join = make<JoinEdge>(
        tables[0], tables[1], ExprVector{}, false, false, false, false);
    dt->joins.push_back(join);
    return join;
  }
  return nullptr;
}

// True if 'expr' is of the form a = b where a depends on one of ''tables' and b
// on the other. If true, returns the side depending on tables[0] in 'left' and
// the other in 'right'.
bool isJoinEquality(
    ExprCP expr,
    std::vector<PlanObjectP>& tables,
    ExprCP& left,
    ExprCP& right) {
  if (expr->type() == PlanType::kCall) {
    auto call = expr->as<Call>();
    if (call->name() == toName("eq")) {
      left = call->args()[0];
      right = call->args()[1];
      auto leftTable = singleTable(left);
      auto rightTable = singleTable(right);
      if (!leftTable || !rightTable) {
        return false;
      }
      if (leftTable == tables[1]) {
        std::swap(left, right);
      }
      return true;
    }
  }
  return false;
}

void extractNonInnerJoinEqualities(
    ExprVector& conjuncts,
    PlanObjectCP right,
    ExprVector& leftKeys,
    ExprVector& rightKeys,
    PlanObjectSet& allLeft) {
  for (auto i = 0; i < conjuncts.size(); ++i) {
    auto* eq = toName("eq");
    auto conjunct = conjuncts[i];
    if (isCallExpr(conjunct, eq)) {
      auto eq = conjunct->as<Call>();
      auto leftTables = eq->args()[0]->allTables();
      auto rightTables = eq->args()[1]->allTables();
      if (rightTables.size() == 1 && rightTables.contains(right) &&
          !leftTables.contains(right)) {
        allLeft.unionSet(leftTables);
        leftKeys.push_back(eq->args()[0]);
        rightKeys.push_back(eq->args()[1]);
        conjuncts.erase(conjuncts.begin() + i);
        --i;
      } else if (
          leftTables.size() == 1 && leftTables.contains(right) &&
          !rightTables.contains(right)) {
        allLeft.unionSet(rightTables);
        leftKeys.push_back(eq->args()[1]);
        rightKeys.push_back(eq->args()[0]);
        conjuncts.erase(conjuncts.begin() + i);
        --i;
      }
    }
  }
}

void DerivedTable::distributeConjuncts() {
  std::vector<DerivedTableP> changedDts;
  if (!having.empty()) {
    VELOX_CHECK(aggregation && aggregation->aggregation);
    auto op = aggregation->aggregation;
    PlanObjectSet grouping;
    for (auto expr : op->grouping) {
      grouping.unionSet(expr->columns());
    }
    for (auto i = 0; i < having.size(); ++i) {
      // No pushdown of non-deterministic.
      if (having[i]->containsFunction(FunctionSet::kNondeterministic)) {
        continue;
      }
      // having that refers to no aggregates goes below the aggregation.
      if (having[i]->columns().isSubset(grouping)) {
        conjuncts.push_back(having[i]);
        having.erase(having.begin() + i);
        --i;
      }
    }
  }
  expandConjuncts();
  for (auto i = 0; i < conjuncts.size(); ++i) {
    // No pushdown of non-deterministic.
    if (conjuncts[i]->containsFunction(FunctionSet::kNondeterministic)) {
      continue;
    }
    PlanObjectSet tableSet = conjuncts[i]->allTables();
    std::vector<PlanObjectP> tables;
    tableSet.forEachMutable([&](auto table) { tables.push_back(table); });
    if (tables.size() == 1) {
      if (tables[0] == this) {
        continue; // the conjunct depends on containing dt, like grouping or
                  // existence flags. Leave in place.
      } else if (tables[0]->type() == PlanType::kDerivedTable) {
        // Translate the column names and add the condition to the conjuncts in
        // the dt.
        auto innerDt = tables[0]->as<DerivedTable>();
        auto imported =
            importExpr(conjuncts[i], innerDt->columns, innerDt->exprs);
        if (innerDt->aggregation) {
          innerDt->having.push_back(imported);
        } else {
          innerDt->conjuncts.push_back(imported);
        }
        if (std::find(changedDts.begin(), changedDts.end(), innerDt) ==
            changedDts.end()) {
          changedDts.push_back(innerDt);
        }
        conjuncts.erase(conjuncts.begin() + i);
        --numCanonicalConjuncts;
        --i;
        continue;
      } else {
        VELOX_CHECK(tables[0]->type() == PlanType::kTable);
        tables[0]->as<BaseTable>()->addFilter(conjuncts[i]);
      }
      conjuncts.erase(conjuncts.begin() + i);
      --numCanonicalConjuncts;
      --i;
      continue;
    }
    if (tables.size() == 2) {
      ExprCP left = nullptr;
      ExprCP right = nullptr;
      // expr depends on 2 tables. If it is left = right or right = left and
      // there is no edge or the edge is inner, add the equality. For other
      // cases, leave the conjunct in place, to be evaluated when its
      // dependences are known.
      if (isJoinEquality(conjuncts[i], tables, left, right)) {
        auto join = findJoin(this, tables, true);
        if (join->isInner()) {
          if (left->type() == PlanType::kColumn &&
              right->type() == PlanType::kColumn) {
            left->as<Column>()->equals(right->as<Column>());
          }
          if (join->leftTable() == tables[0]) {
            join->addEquality(left, right);
          } else {
            join->addEquality(right, left);
          }
          conjuncts.erase(conjuncts.begin() + i);
          --numCanonicalConjuncts;
          --i;
        }
      }
    }
  }
  // Remake initial plan for changedDTs. Calls distributeConjuncts
  // recursively for further pushdown of pushed down items. Replans
  // on returning edge of recursion, so everybody's initial plan is
  // up to date after all pushdowns.
  for (auto* changed : changedDts) {
    changed->makeInitialPlan();
  }
}

void DerivedTable::makeInitialPlan() {
  auto optimization = queryCtx()->optimization();
  MemoKey key;
  key.firstTable = this;
  key.tables.add(this);
  for (auto& column : columns) {
    key.columns.add(column);
  }
  distributeConjuncts();
  addImpliedJoins();
  linkTablesToJoins();
  for (auto& join : joins) {
    join->guessFanout();
  }
  setStartTables();
  PlanState state(*optimization, this);
  for (auto expr : exprs) {
    state.targetColumns.unionColumns(expr);
  }

  optimization->makeJoins(nullptr, state);
  Distribution emptyDistribution;
  bool needsShuffle;
  auto plan = state.plans.best(emptyDistribution, needsShuffle)->op;
  auto& distribution = plan->distribution();
  ExprVector partition = distribution.partition;
  ExprVector order = distribution.order;
  auto orderType = distribution.orderType;
  replace(partition, exprs, columns.data());
  replace(order, exprs, columns.data());
  auto* dtDist = make<Distribution>(
      distribution.distributionType,
      distribution.cardinality,
      partition,
      order,
      orderType);
  this->distribution = dtDist;
  optimization->memo()[key] = std::move(state.plans);
}

std::string DerivedTable::toString() const {
  std::stringstream out;
  out << "{dt " << cname << " from ";
  for (auto& table : tables) {
    out << table->toString() << " ";
  }
  out << " where ";
  for (auto& join : joins) {
    out << join->toString();
  }
  if (!conjuncts.empty()) {
    out << " where " << conjunctsToString(conjuncts);
  }
  out << "}";
  return out.str();
}

float tableCardinality(PlanObjectCP table) {
  if (table->type() == PlanType::kTable) {
    return table->as<BaseTable>()
        ->schemaTable->columnGroups[0]
        ->distribution()
        .cardinality;
  }
  VELOX_CHECK(table->type() == PlanType::kDerivedTable);
  return table->as<DerivedTable>()->distribution->cardinality;
}

void JoinEdge::guessFanout() {
  if (fanoutsFixed_) {
    return;
  }
  auto* opt = queryCtx()->optimization();
  auto samplePair = opt->history().sampleJoin(this);
  auto left = joinCardinality(leftTable_, toRangeCast<Column>(leftKeys_));
  auto right = joinCardinality(rightTable_, toRangeCast<Column>(rightKeys_));
  leftUnique_ = left.unique;
  rightUnique_ = right.unique;
  if (samplePair.first == 0 && samplePair.second == 0) {
    lrFanout_ = right.joinCardinality * baseSelectivity(rightTable_);
    rlFanout_ = left.joinCardinality * baseSelectivity(leftTable_);
  } else {
    lrFanout_ = samplePair.second * baseSelectivity(rightTable_);
    rlFanout_ = samplePair.first * baseSelectivity(leftTable_);
  }
  // If one side is unique, the other side is a pk to fk join, with fanout =
  // fk-table-card / pk-table-card.
  if (rightUnique_) {
    lrFanout_ = baseSelectivity(rightTable_);
    rlFanout_ = tableCardinality(leftTable_) / tableCardinality(rightTable_) *
        baseSelectivity(leftTable_);
  }
  if (leftUnique_) {
    rlFanout_ = baseSelectivity(leftTable_);
    lrFanout_ = tableCardinality(rightTable_) / tableCardinality(leftTable_) *
        baseSelectivity(rightTable_);
  }
}

void exprsToString(const ExprVector& exprs, std::stringstream& out) {
  int32_t size = exprs.size();
  for (auto i = 0; i < size; ++i) {
    out << exprs[i]->toString() << (i < size - 1 ? ", " : "");
  }
}

} // namespace facebook::velox::optimizer
