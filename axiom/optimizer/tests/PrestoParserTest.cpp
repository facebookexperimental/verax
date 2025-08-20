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

#include "axiom/optimizer/tests/PrestoParser.h"
#include <gtest/gtest.h>
#include "axiom/logical_plan/PlanBuilder.h"
#include "axiom/logical_plan/PlanPrinter.h"
#include "axiom/optimizer/connectors/tpch/TpchConnectorMetadata.h"
#include "axiom/optimizer/tests/LogicalPlanMatcher.h"
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

namespace lp = facebook::velox::logical_plan;

namespace facebook::velox::optimizer::test {
namespace {

class PrestoParserTest : public testing::Test {
 public:
  static constexpr const char* kTpchConnectorId = "tpch";

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManagerOptions{});

    auto emptyConfig = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>());

    velox::connector::tpch::registerTpchConnectorMetadataFactory(
        std::make_unique<
            velox::connector::tpch::TpchConnectorMetadataFactoryImpl>());

    connector::tpch::TpchConnectorFactory tpchConnectorFactory;
    auto tpchConnector =
        tpchConnectorFactory.newConnector(kTpchConnectorId, emptyConfig);
    connector::registerConnector(std::move(tpchConnector));

    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
  }

  static void TearDownTestCase() {
    connector::unregisterConnector(kTpchConnectorId);
  }

  memory::MemoryPool* pool() {
    return pool_.get();
  }

  void testSql(const std::string& sql, lp::LogicalPlanMatcherBuilder& matcher) {
    SCOPED_TRACE(sql);
    test::PrestoParser parser(kTpchConnectorId, pool());

    auto statement = parser.parse(sql);
    ASSERT_TRUE(statement->isSelect());

    auto logicalPlan = statement->asUnchecked<test::SelectStatement>()->plan();
    ASSERT_TRUE(matcher.build()->match(logicalPlan));
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild("leaf")};
};

TEST_F(PrestoParserTest, selectStar) {
  auto matcher = lp::LogicalPlanMatcherBuilder().tableScan();
  testSql("SELECT * FROM nation", matcher);
}

TEST_F(PrestoParserTest, countStar) {
  auto matcher = lp::LogicalPlanMatcherBuilder().tableScan().aggregate();

  testSql("SELECT count(*) FROM nation", matcher);
  testSql("SELECT count(1) FROM nation", matcher);
}

TEST_F(PrestoParserTest, simpleGroupBy) {
  {
    auto matcher = lp::LogicalPlanMatcherBuilder().tableScan().aggregate();

    testSql("SELECT n_name, count(1) FROM nation GROUP BY 1", matcher);
    testSql("SELECT n_name, count(1) FROM nation GROUP BY n_name", matcher);
  }

  {
    auto matcher =
        lp::LogicalPlanMatcherBuilder().tableScan().aggregate().project();
    testSql(
        "SELECT count(1) FROM nation GROUP BY n_name, n_regionkey", matcher);
  }
}

TEST_F(PrestoParserTest, distinct) {
  {
    auto matcher =
        lp::LogicalPlanMatcherBuilder().tableScan().project().aggregate();
    testSql("SELECT DISTINCT n_regionkey FROM nation", matcher);
    testSql("SELECT DISTINCT n_regionkey, length(n_name) FROM nation", matcher);
  }

  {
    auto matcher = lp::LogicalPlanMatcherBuilder()
                       .tableScan()
                       .aggregate()
                       .project()
                       .aggregate();
    testSql(
        "SELECT DISTINCT count(1) FROM nation GROUP BY n_regionkey", matcher);
  }

  {
    auto matcher = lp::LogicalPlanMatcherBuilder().tableScan().aggregate();
    testSql("SELECT DISTINCT * FROM nation", matcher);
  }
}

TEST_F(PrestoParserTest, groupingKeyExpr) {
  {
    auto matcher =
        lp::LogicalPlanMatcherBuilder().tableScan().aggregate().project();

    testSql(
        "SELECT n_name, count(1), length(n_name) FROM nation GROUP BY 1",
        matcher);
  }

  {
    auto matcher =
        lp::LogicalPlanMatcherBuilder().tableScan().project().aggregate();
    testSql(
        "SELECT substr(n_name, 1, 2), count(1) FROM nation GROUP BY 1",
        matcher);
  }
}

TEST_F(PrestoParserTest, join) {
  {
    auto matcher = lp::LogicalPlanMatcherBuilder().tableScan().join(
        lp::LogicalPlanMatcherBuilder().tableScan().build());

    testSql("SELECT * FROM nation, region", matcher);

    testSql(
        "SELECT * FROM nation LEFT JOIN region ON n_regionkey = r_regionkey",
        matcher);

    testSql(
        "SELECT * FROM nation FULL OUTER JOIN region ON n_regionkey = r_regionkey",
        matcher);
  }

  {
    auto matcher =
        lp::LogicalPlanMatcherBuilder()
            .tableScan()
            .join(lp::LogicalPlanMatcherBuilder().tableScan().build())
            .filter();

    testSql(
        "SELECT * FROM nation, region WHERE n_regionkey = r_regionkey",
        matcher);
  }

  {
    auto matcher =
        lp::LogicalPlanMatcherBuilder()
            .tableScan()
            .join(lp::LogicalPlanMatcherBuilder().tableScan().build())
            .filter()
            .project();

    testSql(
        "SELECT n_name, r_name FROM nation, region WHERE n_regionkey = r_regionkey",
        matcher);
  }
}

TEST_F(PrestoParserTest, everything) {
  auto matcher = lp::LogicalPlanMatcherBuilder()
                     .tableScan()
                     .join(lp::LogicalPlanMatcherBuilder().tableScan().build())
                     .filter()
                     .aggregate()
                     .sort();

  testSql(
      "SELECT r_name, count(*) FROM nation, region "
      "WHERE n_regionkey = r_regionkey "
      "GROUP BY 1 "
      "ORDER BY 2 DESC",
      matcher);
}

} // namespace
} // namespace facebook::velox::optimizer::test
