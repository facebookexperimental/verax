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

#include "axiom/optimizer/Plan.h"
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "axiom/logical_plan/PlanBuilder.h"
#include "axiom/optimizer/VeloxHistory.h"
#include "axiom/optimizer/connectors/tests/TestConnector.h"
#include "axiom/optimizer/tests/ParquetTpchTest.h"
#include "axiom/optimizer/tests/PlanMatcher.h"
#include "axiom/optimizer/tests/QueryTestBase.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

DEFINE_int32(num_repeats, 1, "Number of repeats for optimization timing");

DECLARE_int32(optimizer_trace);
DECLARE_int32(num_workers);
DECLARE_string(history_save_path);

namespace lp = facebook::velox::logical_plan;

namespace facebook::velox::optimizer {
namespace {

class PlanTest : public virtual test::ParquetTpchTest,
                 public virtual test::QueryTestBase {
 protected:
  static constexpr auto kTestConnectorId = "test";

  static void SetUpTestCase() {
    ParquetTpchTest::SetUpTestCase();
    LocalRunnerTestBase::testDataPath_ = FLAGS_data_path;
    LocalRunnerTestBase::localFileFormat_ = "parquet";
    connector::unregisterConnector(exec::test::kHiveConnectorId);
    connector::unregisterConnectorFactory("hive");
    LocalRunnerTestBase::SetUpTestCase();
  }

  static void TearDownTestCase() {
    if (!FLAGS_history_save_path.empty()) {
      suiteHistory().saveToFile(FLAGS_history_save_path);
    }
    LocalRunnerTestBase::TearDownTestCase();
    ParquetTpchTest::TearDownTestCase();
  }

  void SetUp() override {
    ParquetTpchTest::SetUp();
    QueryTestBase::SetUp();
    allocator_ = std::make_unique<HashStringAllocator>(pool_.get());
    context_ = std::make_unique<QueryGraphContext>(*allocator_);
    queryCtx() = context_.get();

    testConnector_ =
        std::make_shared<connector::TestConnector>(kTestConnectorId);
    connector::registerConnector(testConnector_);
  }

  void TearDown() override {
    context_.reset();
    queryCtx() = nullptr;
    allocator_.reset();
    ParquetTpchTest::TearDown();
    QueryTestBase::TearDown();
    connector::unregisterConnector(kTestConnectorId);
  }

  void checkSame(
      const lp::LogicalPlanNodePtr& planNode,
      core::PlanNodePtr referencePlan) {
    VELOX_CHECK_NOT_NULL(referencePlan);

    auto fragmentedPlan = planVelox(planNode);
    ASSERT_TRUE(fragmentedPlan.plan != nullptr);

    optimizer::test::TestResult referenceResult;
    assertSame(referencePlan, fragmentedPlan, &referenceResult);

    auto numWorkers = FLAGS_num_workers;
    if (numWorkers != 1) {
      gflags::FlagSaver saver;
      FLAGS_num_workers = 1;

      auto singlePlan = planVelox(planNode);
      ASSERT_TRUE(singlePlan.plan != nullptr);

      auto singleResult = runFragmentedPlan(singlePlan);
      exec::test::assertEqualResults(
          referenceResult.results, singleResult.results);
    }
  }

  core::PlanNodePtr toSingleNodePlan(
      const lp::LogicalPlanNodePtr& logicalPlan,
      const std::shared_ptr<connector::Connector>& defaultConnector = nullptr) {
    gflags::FlagSaver saver;
    FLAGS_num_workers = 1;

    schema_ = std::make_shared<velox::optimizer::SchemaResolver>(
        defaultConnector == nullptr ? testConnector_ : defaultConnector, "");

    auto plan = planVelox(logicalPlan).plan;

    EXPECT_EQ(1, plan->fragments().size());
    return plan->fragments().at(0).fragment.planNode;
  }

  std::unique_ptr<HashStringAllocator> allocator_;
  std::unique_ptr<QueryGraphContext> context_;
  std::shared_ptr<connector::TestConnector> testConnector_;
};

auto gte(const std::string& name, int64_t n) {
  return common::test::singleSubfieldFilter(name, exec::greaterThanOrEqual(n));
}

auto lte(const std::string& name, int64_t n) {
  return common::test::singleSubfieldFilter(name, exec::lessThanOrEqual(n));
}

auto between(const std::string& name, int64_t min, int64_t max) {
  return common::test::singleSubfieldFilter(name, exec::between(min, max));
}

auto gt(const std::string& name, double d) {
  return common::test::singleSubfieldFilter(name, exec::greaterThanDouble(d));
}

auto lt(const std::string& name, double d) {
  return common::test::singleSubfieldFilter(name, exec::lessThanDouble(d));
}

TEST_F(PlanTest, queryGraph) {
  TypePtr row1 = ROW({{"c1", ROW({{"c1a", INTEGER()}})}, {"c2", DOUBLE()}});
  TypePtr row2 = row1 =
      ROW({{"c1", ROW({{"c1a", INTEGER()}})}, {"c2", DOUBLE()}});
  TypePtr largeRow = ROW(
      {{"c1", ROW({{"c1a", INTEGER()}})},
       {"c2", DOUBLE()},
       {"m1", MAP(INTEGER(), ARRAY(INTEGER()))}});
  TypePtr differentNames =
      ROW({{"different", ROW({{"c1a", INTEGER()}})}, {"c2", DOUBLE()}});

  auto* dedupRow1 = toType(row1);
  auto* dedupRow2 = toType(row2);
  auto* dedupLargeRow = toType(largeRow);
  auto* dedupDifferentNames = toType(differentNames);

  // dedupped complex types make a copy.
  EXPECT_NE(row1.get(), dedupRow1);

  // Identical types get equal pointers.
  EXPECT_EQ(dedupRow1, dedupRow2);

  // Different names differentiate types.
  EXPECT_NE(dedupDifferentNames, dedupRow1);

  // Shared complex substructure makes equal pointers.
  EXPECT_EQ(dedupRow1->childAt(0).get(), dedupLargeRow->childAt(0).get());

  // Identical child types with different names get equal pointers.
  EXPECT_EQ(dedupRow1->childAt(0).get(), dedupDifferentNames->childAt(0).get());

  auto* path = make<Path>()
                   ->subscript("field")
                   ->subscript(123)
                   ->field("f1")
                   ->cardinality();
  auto interned = queryCtx()->toPath(path);
  EXPECT_EQ(interned, path);
  auto* path2 = make<Path>()
                    ->subscript("field")
                    ->subscript(123)
                    ->field("f1")
                    ->cardinality();
  auto interned2 = queryCtx()->toPath(path2);
  EXPECT_EQ(interned2, interned);
}

TEST_F(PlanTest, agg) {
  testConnector_->addTable(
      "numbers", ROW({"a", "b", "c"}, {BIGINT(), DOUBLE(), VARCHAR()}));

  auto logicalPlan = lp::PlanBuilder()
                         .tableScan(kTestConnectorId, "numbers", {"a", "b"})
                         .aggregate({"a"}, {"sum(b)"})
                         .build();

  auto plan = toSingleNodePlan(logicalPlan);

  auto matcher = core::PlanMatcherBuilder()
                     .tableScan()
                     .partialAggregation()
                     .localPartition()
                     .finalAggregation()
                     .project()
                     .build();

  ASSERT_TRUE(matcher->match(plan));
}

// Verify that optimizer can handle connectors that do not support filter
// pushdown.
TEST_F(PlanTest, rejectedFilters) {
  testConnector_->addTable(
      "numbers", ROW({"a", "b", "c"}, {BIGINT(), DOUBLE(), VARCHAR()}));

  auto logicalPlan = lp::PlanBuilder()
                         .tableScan(kTestConnectorId, "numbers", {"a", "b"})
                         .filter("a > 10")
                         .map({"a + 2"})
                         .build();

  auto plan = toSingleNodePlan(logicalPlan);

  auto matcher =
      core::PlanMatcherBuilder().tableScan().filter().project().build();

  ASSERT_TRUE(matcher->match(plan));
}

TEST_F(PlanTest, filterToJoinEdge) {
  auto nationType = ROW({"n_regionkey"}, {BIGINT()});
  auto regionType = ROW({"r_regionkey"}, {BIGINT()});

  const auto connectorId = exec::test::kHiveConnectorId;
  const auto connector = connector::getConnector(connectorId);

  lp::PlanBuilder::Context context;
  auto logicalPlan = lp::PlanBuilder(context)
                         .tableScan(connectorId, "nation", nationType->names())
                         .crossJoin(lp::PlanBuilder(context).tableScan(
                             connectorId, "region", regionType->names()))
                         .filter("n_regionkey + 1 = r_regionkey + 1")
                         .build();

  {
    auto plan = toSingleNodePlan(logicalPlan, connector);
    auto matcher = core::PlanMatcherBuilder()
                       .tableScan("region")
                       .project()
                       .hashJoin(core::PlanMatcherBuilder()
                                     .tableScan("nation")
                                     .project()
                                     .build())
                       .project()
                       .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto referencePlan = exec::test::PlanBuilder(planNodeIdGenerator)
                           .tableScan("nation", nationType)
                           .nestedLoopJoin(
                               exec::test::PlanBuilder(planNodeIdGenerator)
                                   .tableScan("region", regionType)
                                   .planNode(),
                               {"n_regionkey", "r_regionkey"},
                               core::JoinType::kInner)
                           .filter("n_regionkey + 1 = r_regionkey + 1")
                           .planNode();

  checkSame(logicalPlan, referencePlan);

  // Add non-deterministic always-true filters to both branches of the join and
  // after the join.
  logicalPlan =
      lp::PlanBuilder(context)
          .tableScan(connectorId, "nation", nationType->names())
          .filter("rand() < 2.0")
          .crossJoin(lp::PlanBuilder(context)
                         .tableScan(connectorId, "region", regionType->names())
                         .filter("rand() < 3.0"))
          .filter("n_regionkey + 1 = r_regionkey + 1 and rand() < 4.0")
          .build();

  {
    auto plan = toSingleNodePlan(logicalPlan, connector);
    auto matcher =
        core::PlanMatcherBuilder()
            .tableScan("nation")
            // TODO Why is this filter not pushed down into scan?
            .filter("rand() < 2.0")
            // TODO Fix this plan. There should be only one project node.
            .project()
            .project()
            .project()
            .hashJoin(core::PlanMatcherBuilder()
                          .tableScan("region")
                          .filter("rand() < 3.0")
                          // TODO Fix this plan. There should be only one
                          // project node.
                          .project()
                          .project()
                          .project()
                          .build())
            .filter("rand() < 4.0")
            // TODO Fix this plan. There should be only one project node.
            .project()
            .project()
            .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  checkSame(logicalPlan, referencePlan);
}

TEST_F(PlanTest, filterImport) {
  auto ordersType = ROW({"o_custkey", "o_totalprice"}, {BIGINT(), DOUBLE()});

  const auto connectorId = exec::test::kHiveConnectorId;
  const auto connector = connector::getConnector(connectorId);

  auto logicalPlan = lp::PlanBuilder()
                         .tableScan(connectorId, "orders", ordersType->names())
                         .aggregate({"o_custkey"}, {"sum(o_totalprice) as a0"})
                         .filter("o_custkey < 100 and a0 > 200.0")
                         .build();

  {
    auto plan = toSingleNodePlan(logicalPlan, connector);
    auto matcher = core::PlanMatcherBuilder()
                       .tableScan("orders")
                       .partialAggregation()
                       .localPartition()
                       .finalAggregation()
                       .filter("gt(\"dt1.a0\",200)")
                       .project()
                       .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  auto referencePlan =
      exec::test::PlanBuilder()
          .tableScan("orders", ordersType)
          .singleAggregation({"o_custkey"}, {"sum(o_totalprice)"})
          .filter("o_custkey < 100 and a0 > 200.0")
          .planNode();

  checkSame(logicalPlan, referencePlan);
}

TEST_F(PlanTest, filterBreakup) {
  const char* filterText =
      "        (\n"
      "                l_partkey = p_partkey\n"
      "                and p_brand = 'Brand#12'\n"
      "                and p_container like 'SM%'\n"
      "                and l_quantity >= 1.0 and l_quantity <= 1.0 + 10.0\n"
      "                and p_size between 1::int and 5::int\n"
      "                and l_shipmode in ('AIR', 'AIR REG')\n"
      "                and l_shipinstruct = 'DELIVER IN PERSON'\n"
      "        )\n"
      "        or\n"
      "        (\n"
      "                p_partkey = l_partkey\n"
      "                and p_brand = 'Brand#23'\n"
      "                and p_container like 'MED%'\n"
      "                and l_quantity >= 10.0 and l_quantity <= 10.0 + 10.0\n"
      "                and p_size between 1::int and 10::int\n"
      "                and l_shipmode in ('AIR', 'AIR REG')\n"
      "                and l_shipinstruct = 'DELIVER IN PERSON'\n"
      "        )\n"
      "        or\n"
      "        (\n"
      "                p_partkey = l_partkey\n"
      "                and p_brand = 'Brand#34'\n"
      "                and p_container like 'LG%'\n"
      "                and l_quantity >= 20.0 and l_quantity <= 20.0 + 10.0\n"
      "                and p_size between 1::int and 15::int\n"
      "                and l_shipmode in ('AIR', 'AIR REG')\n"
      "                and l_shipinstruct = 'DELIVER IN PERSON'\n"
      "        )\n";

  auto lineitemType = ROW(
      {{"l_partkey", BIGINT()},
       {"l_shipmode", VARCHAR()},
       {"l_shipinstruct", VARCHAR()},
       {"l_extendedprice", DOUBLE()},
       {"l_discount", DOUBLE()},
       {"l_quantity", DOUBLE()}});

  auto partType = ROW(
      {{"p_partkey", BIGINT()},
       {"p_brand", VARCHAR()},
       {"p_container", VARCHAR()},
       {"p_size", INTEGER()}});

  const auto connectorId = exec::test::kHiveConnectorId;
  const auto connector = connector::getConnector(connectorId);

  lp::PlanBuilder::Context context;
  auto logicalPlan =
      lp::PlanBuilder(context)
          .tableScan(connectorId, "lineitem", lineitemType->names())
          .crossJoin(lp::PlanBuilder(context).tableScan(
              connectorId, "part", partType->names()))
          .filter(filterText)
          .project({"l_extendedprice * (1.0 - l_discount) as part_revenue"})
          .aggregate({}, {"sum(part_revenue)"})
          .build();

  {
    // Expect the per table filters to be extracted from the OR.
    // TODO Verify remaining filters.
    auto lineitemFilters =
        common::test::SubfieldFiltersBuilder()
            .add("l_shipinstruct", exec::equal("DELIVER IN PERSON"))
            .add(
                "l_shipmode",
                exec::in(std::vector<std::string>{"AIR", "AIR REG"}))
            .build();

    auto plan = toSingleNodePlan(logicalPlan, connector);
    auto matcher =
        core::PlanMatcherBuilder()
            .hiveScan(
                "lineitem",
                std::move(lineitemFilters),
                // TODO Fix this plan. Compact the filter to between(1, 30) and
                // push down as subfield filter.
                "\"or\"(l_quantity >= 20.0 AND l_quantity <= 30.0, "
                "   \"or\"(l_quantity >= 1.0 AND l_quantity <= 11.0, "
                "          l_quantity >= 10.0 AND l_quantity <= 20.0))")
            .hashJoin(
                core::PlanMatcherBuilder()
                    .hiveScan(
                        "part",
                        {},
                        "\"or\"(\"and\"(p_size between 1 and 15, (p_brand = 'Brand#34' AND p_container LIKE 'LG%')), "
                        "   \"or\"(\"and\"(p_size between 1 and 5, (p_brand = 'Brand#12' AND p_container LIKE 'SM%')), "
                        "          \"and\"(p_size between 1 and 10, (p_brand = 'Brand#23' AND p_container LIKE 'MED%'))))")
                    .build())
            .filter()
            .project()
            .partialAggregation()
            .localPartition()
            .finalAggregation()
            .project()
            .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  auto referenceBuilder = std::make_unique<exec::test::TpchQueryBuilder>(
      dwio::common::FileFormat::PARQUET);
  referenceBuilder->initialize(FLAGS_data_path);

  auto referencePlan = referenceBuilder->getQueryPlan(19).plan;

  checkSame(logicalPlan, referencePlan);
}

TEST_F(PlanTest, unionAll) {
  auto nationType =
      ROW({"n_nationkey", "n_regionkey", "n_name", "n_comment"},
          {BIGINT(), BIGINT(), VARCHAR(), VARCHAR()});

  const auto connectorId = exec::test::kHiveConnectorId;
  const auto connector = connector::getConnector(connectorId);

  const std::vector<std::string>& names = nationType->names();

  lp::PlanBuilder::Context ctx;
  auto t1 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "nation", names)
                .filter("n_nationkey < 11");
  auto t2 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "nation", names)
                .filter("n_nationkey > 13");

  auto logicalPlan = t1.unionAll(t2)
                         .project({"n_regionkey + 1 as rk"})
                         .filter("rk % 3 = 1")
                         .build();

  {
    auto plan = toSingleNodePlan(logicalPlan, connector);
    auto matcher =
        core::PlanMatcherBuilder()
            .hiveScan(
                "nation", lte("n_nationkey", 10), "1 = (n_regionkey + 1) % 3")
            .project()
            .localPartition(core::PlanMatcherBuilder()
                                .hiveScan(
                                    "nation",
                                    gte("n_nationkey", 14),
                                    "1 = (n_regionkey + 1) % 3")
                                .project()
                                .build())
            .project()
            .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  // Skip distributed run. Problem with local exchange source with
  // multiple inputs.
  gflags::FlagSaver saver;
  FLAGS_num_workers = 1;

  auto referencePlan = exec::test::PlanBuilder(pool_.get())
                           .tableScan("nation", nationType)
                           .filter("n_nationkey < 11 or n_nationkey > 13")
                           .project({"n_regionkey + 1 as rk"})
                           .filter("rk % 3 = 1")
                           .planNode();

  checkSame(logicalPlan, referencePlan);
}

TEST_F(PlanTest, unionJoin) {
  auto partType = ROW({"p_partkey", "p_retailprice"}, {BIGINT(), DOUBLE()});
  auto partSuppType = ROW({"ps_partkey", "ps_availqty"}, {BIGINT(), INTEGER()});

  const auto connectorId = exec::test::kHiveConnectorId;
  const auto connector = connector::getConnector(connectorId);

  lp::PlanBuilder::Context ctx;
  auto ps1 =
      lp::PlanBuilder(ctx)
          .tableScan(connectorId, "partsupp", {"ps_partkey", "ps_availqty"})
          .filter("ps_availqty < 1000::int")
          .project({"ps_partkey"});

  auto ps2 =
      lp::PlanBuilder(ctx)
          .tableScan(connectorId, "partsupp", {"ps_partkey", "ps_availqty"})
          .filter("ps_availqty > 2000::int")
          .project({"ps_partkey"});

  auto ps3 =
      lp::PlanBuilder(ctx)
          .tableScan(connectorId, "partsupp", {"ps_partkey", "ps_availqty"})
          .filter("ps_availqty between 1200::int and 1400::int")
          .project({"ps_partkey"});

  // The shape of the partsupp union is ps1 union all (ps2 union all ps3). We
  // verify that a stack of multiple set ops works.
  auto psu2 = ps2.unionAll(ps3);

  auto p1 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "part", {"p_partkey", "p_retailprice"})
                .filter("p_retailprice < 1100.0");

  auto p2 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "part", {"p_partkey", "p_retailprice"})
                .filter("p_retailprice > 1200.0");

  auto logicalPlan =
      ps1.unionAll(psu2)
          .join(p1.unionAll(p2), "ps_partkey = p_partkey", lp::JoinType::kInner)
          .aggregate({}, {"sum(1)"})
          .build();

  {
    auto plan = toSingleNodePlan(logicalPlan, connector);
    auto matcher =
        core::PlanMatcherBuilder()
            .hiveScan("partsupp", lte("ps_availqty", 999))
            .project()
            .localPartition(
                core::PlanMatcherBuilder()
                    .hiveScan("partsupp", gte("ps_availqty", 2001))
                    .project()
                    .localPartition(
                        core::PlanMatcherBuilder()
                            .hiveScan(
                                "partsupp", between("ps_availqty", 1200, 1400))
                            .project()
                            .build())
                    .build())
            .hashJoin(
                core::PlanMatcherBuilder()
                    .hiveScan("part", lt("p_retailprice", 1100.0))
                    .project()
                    .localPartition(
                        core::PlanMatcherBuilder()
                            .hiveScan("part", gt("p_retailprice", 1200.0))
                            .project()
                            .build())
                    .build(),
                core::JoinType::kInner)
            .project()
            .partialAggregation()
            .localPartition()
            .finalAggregation()
            .project()
            .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  // Skip distributed run. Problem with local exchange source with
  // multiple inputs.
  gflags::FlagSaver saver;
  FLAGS_num_workers = 1;

  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto referencePlan =
      exec::test::PlanBuilder(idGenerator)
          .tableScan("partsupp", partSuppType)
          .filter(
              "ps_availqty < 1000::int or ps_availqty > 2000::int "
              "or ps_availqty between 1200::int and 1400::int")
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              exec::test::PlanBuilder(idGenerator)
                  .tableScan("part", partType)
                  .filter("p_retailprice < 1100.0 or p_retailprice > 1200.0")
                  .planNode(),
              "",
              {"p_partkey"})
          .project({"p_partkey"})
          .localPartition({})
          .singleAggregation({}, {"sum(1)"})
          .planNode();

  checkSame(logicalPlan, referencePlan);
}

TEST_F(PlanTest, intersect) {
  auto nationType =
      ROW({"n_nationkey", "n_regionkey", "n_name", "n_comment"},
          {BIGINT(), BIGINT(), VARCHAR(), VARCHAR()});

  const auto connectorId = exec::test::kHiveConnectorId;
  const auto connector = connector::getConnector(connectorId);

  const std::vector<std::string>& names = nationType->names();

  lp::PlanBuilder::Context ctx;
  auto t1 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "nation", names)
                .filter("n_nationkey < 21")
                .project({"n_nationkey", "n_regionkey"});
  auto t2 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "nation", names)
                .filter("n_nationkey > 11")
                .project({"n_nationkey", "n_regionkey"});
  auto t3 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "nation", names)
                .filter("n_nationkey > 12")
                .project({"n_nationkey", "n_regionkey"});

  auto logicalPlan =
      lp::PlanBuilder(ctx)
          .setOperation(lp::SetOperation::kIntersect, {t1, t2, t3})
          .project({"n_regionkey + 1 as rk"})
          .filter("rk % 3 = 1")
          .build();

  {
    auto plan = toSingleNodePlan(logicalPlan, connector);
    auto matcher = core::PlanMatcherBuilder()
                       // TODO Fix this plan to push down (n_regionkey + 1) % 3
                       // = 1 to all branches of 'intersect'.
                       .hiveScan("nation", gte("n_nationkey", 13))
                       .project()
                       .hashJoin(
                           core::PlanMatcherBuilder()
                               .hiveScan("nation", gte("n_nationkey", 12))
                               .project()
                               .hashJoin(
                                   core::PlanMatcherBuilder()
                                       .hiveScan(
                                           "nation",
                                           lte("n_nationkey", 20),
                                           "1 = (n_regionkey + 1) % 3")
                                       .project()
                                       .build(),
                                   core::JoinType::kRightSemiFilter)
                               .build(),
                           core::JoinType::kRightSemiFilter)
                       .project()
                       .partialAggregation()
                       .localPartition()
                       .finalAggregation()
                       .project()
                       .project()
                       .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  auto referencePlan = exec::test::PlanBuilder(pool_.get())
                           .tableScan("nation", nationType)
                           .filter("n_nationkey > 12 and n_nationkey < 21")
                           .project({"n_regionkey + 1 as rk"})
                           .filter("rk % 3 = 1")
                           .planNode();

  checkSame(logicalPlan, referencePlan);
}

TEST_F(PlanTest, except) {
  auto nationType =
      ROW({"n_nationkey", "n_regionkey", "n_name", "n_comment"},
          {BIGINT(), BIGINT(), VARCHAR(), VARCHAR()});

  const auto connectorId = exec::test::kHiveConnectorId;
  const auto connector = connector::getConnector(connectorId);

  const std::vector<std::string>& names = nationType->names();

  lp::PlanBuilder::Context ctx;
  auto t1 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "nation", names)
                .filter("n_nationkey < 21")
                .project({"n_nationkey", "n_regionkey"});
  auto t2 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "nation", names)
                .filter("n_nationkey > 16")
                .project({"n_nationkey", "n_regionkey"});
  auto t3 = lp::PlanBuilder(ctx)
                .tableScan(connectorId, "nation", names)
                .filter("n_nationkey <= 5")
                .project({"n_nationkey", "n_regionkey"});

  auto logicalPlan = lp::PlanBuilder(ctx)
                         .setOperation(lp::SetOperation::kExcept, {t1, t2, t3})
                         .project({"n_nationkey", "n_regionkey + 1 as rk"})
                         .filter("rk % 3 = 1")
                         .build();

  {
    auto plan = toSingleNodePlan(logicalPlan, connector);
    auto matcher =
        core::PlanMatcherBuilder()
            .hiveScan(
                "nation", lte("n_nationkey", 20), "1 = (n_regionkey + 1) % 3")
            .project()
            .hashJoin(
                core::PlanMatcherBuilder()
                    // TODO Fix this plan to push down (n_regionkey + 1) % 3 = 1
                    // to all branches of 'except'.
                    .hiveScan("nation", gte("n_nationkey", 17))
                    .project()
                    .build(),
                core::JoinType::kAnti)
            .hashJoin(
                core::PlanMatcherBuilder()
                    .hiveScan("nation", lte("n_nationkey", 5))
                    .project()
                    .build(),
                core::JoinType::kAnti)
            .project()
            .partialAggregation()
            .localPartition()
            .finalAggregation()
            .project()
            .project()
            .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  auto referencePlan = exec::test::PlanBuilder(pool_.get())
                           .tableScan("nation", nationType)
                           .filter("n_nationkey > 5 and n_nationkey <= 16")
                           .project({"n_nationkey", "n_regionkey + 1 as rk"})
                           .filter("rk % 3 = 1")
                           .planNode();

  checkSame(logicalPlan, referencePlan);
}

} // namespace
} // namespace facebook::velox::optimizer

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
