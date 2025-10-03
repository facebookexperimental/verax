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

#include <gtest/gtest.h>
#include "axiom/connectors/tests/TestConnector.h"
#include "axiom/logical_plan/PlanBuilder.h"
#include "axiom/optimizer/Optimization.h"
#include "axiom/optimizer/tests/PlanMatcher.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

namespace facebook::axiom::optimizer {
namespace {

using namespace facebook::velox;
namespace lp = facebook::axiom::logical_plan;

class AggregationPlanTest : public testing::Test {
 protected:
  static constexpr auto kTestConnectorId = "test";

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});

    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
  }

  void SetUp() override {
    testConnector_ =
        std::make_shared<connector::TestConnector>(kTestConnectorId);
    velox::connector::registerConnector(testConnector_);

    rootPool_ = memory::memoryManager()->addRootPool("root");
    optimizerPool_ = rootPool_->addLeafChild("optimizer");
  }

  void TearDown() override {
    velox::connector::unregisterConnector(kTestConnectorId);
  }

  velox::core::PlanNodePtr planVelox(const lp::LogicalPlanNodePtr& plan) {
    auto distributedPlan = Optimization::toVeloxPlan(
                               *plan,
                               *optimizerPool_,
                               {}, // optimizerOptions
                               {.numWorkers = 1, .numDrivers = 1})
                               .plan;

    VELOX_CHECK_EQ(1, distributedPlan->fragments().size());
    return distributedPlan->fragments().at(0).fragment.planNode;
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> optimizerPool_;
  std::shared_ptr<connector::TestConnector> testConnector_;
};

TEST_F(AggregationPlanTest, dedupGroupingKeysAndAggregates) {
  testConnector_->createTable(
      "numbers", ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), DOUBLE()}));

  {
    auto logicalPlan = lp::PlanBuilder{}
                           .tableScan(kTestConnectorId, "numbers")
                           .project({"a + b as x", "a + b as y", "c"})
                           .aggregate({"x", "y"}, {"count(1)", "count(1)"})
                           .build();

    auto plan = planVelox(logicalPlan);

    auto matcher = core::PlanMatcherBuilder()
                       .tableScan()
                       .project({"a + b"})
                       .singleAggregation({"x"}, {"count(1)"})
                       .project({"x", "x", "count", "count"})
                       .build();

    ASSERT_TRUE(matcher->match(plan));
  }
}

TEST_F(AggregationPlanTest, duplicatesBetweenGroupAndAggregate) {
  testConnector_->createTable("t", ROW({"a", "b"}, {BIGINT(), BIGINT()}));

  auto logicalPlan = lp::PlanBuilder{}
                         .tableScan(kTestConnectorId, "t")
                         .project({"a + b AS ab1", "a + b AS ab2"})
                         .aggregate({"ab1", "ab2"}, {"count(ab2) AS c1"})
                         .project({"ab1 AS x", "ab2 AS y", "c1 AS z"})
                         .build();

  auto plan = planVelox(logicalPlan);

  auto matcher = core::PlanMatcherBuilder()
                     .tableScan()
                     .project({"plus(a, b)"})
                     .singleAggregation({"ab1"}, {"count(ab1)"})
                     .project({"ab1", "ab1", "c1"})
                     .build();

  ASSERT_TRUE(matcher->match(plan));
}

TEST_F(AggregationPlanTest, dedupMask) {
  testConnector_->createTable("t", ROW({"a", "b"}, {INTEGER()}));

  auto logicalPlan = lp::PlanBuilder(/*enableCoersions=*/true)
                         .tableScan(kTestConnectorId, "t")
                         .aggregate(
                             {},
                             {"sum(a) FILTER (WHERE b > 0) AS sum1",
                              "sum(a) FILTER (WHERE b < 0) AS sum2",
                              "sum(a) FILTER (WHERE b > 0) AS sum3"})
                         .build();

  auto plan = planVelox(logicalPlan);

  auto matcher = core::PlanMatcherBuilder()
                     .tableScan()
                     .project(
                         {"cast(b as BIGINT) > 0 as mask1",
                          "a",
                          "cast(b as BIGINT) < 0 as mask2"})
                     .singleAggregation(
                         {},
                         {
                             "sum(a) FILTER (WHERE mask1)",
                             "sum(a) FILTER (WHERE mask2)",
                         })
                     .project({"sum1", "sum2", "sum1"})
                     .build();

  ASSERT_TRUE(matcher->match(plan));
}

} // namespace
} // namespace facebook::axiom::optimizer
