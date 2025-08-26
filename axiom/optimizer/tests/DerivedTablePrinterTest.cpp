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
#include "axiom/optimizer/DerivedTablePrinter.h"
#include <gtest/gtest.h>
#include "axiom/optimizer/Plan.h"
#include "axiom/optimizer/VeloxHistory.h"
#include "axiom/optimizer/connectors/tests/TestConnector.h"
#include "axiom/optimizer/tests/PrestoParser.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/tpch/gen/TpchGen.h"

namespace lp = facebook::velox::logical_plan;

namespace facebook::velox::optimizer {
namespace {

class DerivedTablePrinterTest : public testing::Test {
 public:
  static constexpr auto kTestConnectorId = "test";

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});

    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
  }

  void SetUp() override {
    rootPool_ = memory::memoryManager()->addRootPool("root");
    pool_ = rootPool_->addLeafChild("optimizer");

    testConnector_ =
        std::make_shared<connector::TestConnector>(kTestConnectorId);
    connector::registerConnector(testConnector_);

    for (const auto& table : tpch::tables) {
      const auto tableName = tpch::toTableName(table);
      const auto tableSchema = tpch::getTableSchema(table);

      testConnector_->createTable(std::string(tableName), tableSchema);
    }
  }

  void TearDown() override {
    connector::unregisterConnector(kTestConnectorId);
  }

  std::string printQueryGraph(const std::string& sql) {
    test::PrestoParser parser(kTestConnectorId, pool_.get());
    auto statement = parser.parse(sql);

    VELOX_CHECK(statement->isSelect());
    auto logicalPlan = statement->asUnchecked<test::SelectStatement>()->plan();

    auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
    auto context = std::make_unique<QueryGraphContext>(*allocator);
    queryCtx() = context.get();
    SCOPE_EXIT {
      queryCtx() = nullptr;
    };

    auto veloxQueryCtx = core::QueryCtx::create();
    exec::SimpleExpressionEvaluator evaluator(veloxQueryCtx.get(), pool_.get());

    velox::optimizer::VeloxHistory history;
    velox::optimizer::SchemaResolver schemaResolver;

    Locus locus(kTestConnectorId, testConnector_.get());
    Schema schema("test", &schemaResolver, &locus);
    Optimization opt(
        *logicalPlan,
        schema,
        history,
        veloxQueryCtx,
        evaluator,
        {.sampleJoins = false});

    const auto& root = opt.queryGraph();

    return DerivedTablePrinter::toText(root);
  }

  void testPrinter(const std::string& sql) {
    LOG(ERROR) << sql << std::endl;
    LOG(ERROR) << std::endl << printQueryGraph(sql);
  }

  static std::string readSqlFromFile(const std::string& filePath) {
    auto path = velox::test::getDataFilePath("axiom/optimizer/tests", filePath);
    std::ifstream inputFile(path, std::ifstream::binary);

    VELOX_CHECK(inputFile, "Failed to open SQL file: {}", path);

    // Find out file size.
    auto begin = inputFile.tellg();
    inputFile.seekg(0, std::ios::end);
    auto end = inputFile.tellg();

    const auto fileSize = end - begin;
    VELOX_CHECK_GT(fileSize, 0, "SQL file is empty: {}", path);

    // Read the file.
    std::string sql;
    sql.resize(fileSize);

    inputFile.seekg(begin);
    inputFile.read(sql.data(), fileSize);
    inputFile.close();

    return sql;
  }

  void testTpch(int32_t query) {
    auto sql = readSqlFromFile(fmt::format("tpch.queries/q{}.sql", query));
    testPrinter(sql);
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;

  std::shared_ptr<connector::TestConnector> testConnector_;
};

TEST_F(DerivedTablePrinterTest, basic) {
  testPrinter("SELECT count(1) FROM nation");
  testPrinter("SELECT * FROM nation WHERE n_name like 'A%' LIMIT 2");
  testPrinter("SELECT count(1) FROM nation GROUP BY n_regionkey");
  testPrinter(
      "SELECT r_name, count(1) FROM nation, region "
      "WHERE n_regionkey = r_regionkey "
      "GROUP BY 1 "
      "ORDER BY 2");
}

TEST_F(DerivedTablePrinterTest, unnest) {
  testPrinter("SELECT * FROM nation, UNNEST(array[n_nationkey, n_regionkey])");
}

TEST_F(DerivedTablePrinterTest, tpch01) {
  testTpch(1);
}

TEST_F(DerivedTablePrinterTest, tpch03) {
  testTpch(3);
}

} // namespace
} // namespace facebook::velox::optimizer
