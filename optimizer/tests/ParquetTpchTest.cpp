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

#include "optimizer/tests/ParquetTpchTest.h" //@manual
#include "velox/connectors/hive/HiveConnector.h"

DEFINE_string(
    data_path,
    "",
    "Path to TPCH data directory. If empty, the test creates a temp directory and deletes it on exit");
DEFINE_bool(create_dataset, true, "Creates the TPCH tables");
DEFINE_double(tpch_scale, 0.01, "Scale factor");

namespace facebook::velox::optimizer::test {
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

std::shared_ptr<exec::test::DuckDbQueryRunner> ParquetTpchTest::duckDb_;
std::string ParquetTpchTest::createPath_;
std::string ParquetTpchTest::path_;
std::shared_ptr<exec::test::TempDirectoryPath> ParquetTpchTest::tempDirectory_;
std::shared_ptr<exec::test::TpchQueryBuilder> ParquetTpchTest::tpchBuilder_;

//  static
void ParquetTpchTest::SetUpTestCase() {
  memory::MemoryManager::testingSetInstance(memory::MemoryManagerOptions{});

  duckDb_ = std::make_shared<DuckDbQueryRunner>();
  if (FLAGS_data_path.empty()) {
    tempDirectory_ = TempDirectoryPath::create();
    createPath_ = tempDirectory_->getPath();
    path_ = createPath_;
    FLAGS_data_path = createPath_;
  } else if (FLAGS_create_dataset) {
    VELOX_CHECK(!FLAGS_data_path.empty());
    createPath_ = FLAGS_data_path;
    path_ = createPath_;
  }
  tpchBuilder_ =
      std::make_shared<TpchQueryBuilder>(dwio::common::FileFormat::PARQUET);

  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();

  parse::registerTypeResolver();
  filesystems::registerLocalFileSystem();
  dwio::common::registerFileSinks();

  parquet::registerParquetReaderFactory();
  parquet::registerParquetWriterFactory();

  connector::registerConnectorFactory(
      std::make_shared<connector::hive::HiveConnectorFactory>());
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(
              kHiveConnectorId,
              std::make_shared<config::ConfigBase>(
                  std::unordered_map<std::string, std::string>()));
  connector::registerConnector(hiveConnector);

  connector::registerConnectorFactory(
      std::make_shared<connector::tpch::TpchConnectorFactory>());
  auto tpchConnector =
      connector::getConnectorFactory(
          connector::tpch::TpchConnectorFactory::kTpchConnectorName)
          ->newConnector(
              kTpchConnectorId,
              std::make_shared<config::ConfigBase>(
                  std::unordered_map<std::string, std::string>()));
  connector::registerConnector(tpchConnector);

  if (!createPath_.empty()) {
    saveTpchTablesAsParquet();
  }
  tpchBuilder_->initialize(path_);
}

//  static
void ParquetTpchTest::TearDownTestCase() {
  connector::unregisterConnectorFactory(
      connector::hive::HiveConnectorFactory::kHiveConnectorName);
  connector::unregisterConnectorFactory(
      connector::tpch::TpchConnectorFactory::kTpchConnectorName);
  connector::unregisterConnector(kHiveConnectorId);
  connector::unregisterConnector(kTpchConnectorId);
  parquet::unregisterParquetReaderFactory();
  parquet::unregisterParquetWriterFactory();
}

void ParquetTpchTest::saveTpchTablesAsParquet() {
  std::shared_ptr<memory::MemoryPool> rootPool{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool{rootPool->addLeafChild("leaf")};

  for (const auto& table : tpch::tables) {
    auto tableName = toTableName(table);
    auto tableDirectory = fmt::format("{}/{}", createPath_, tableName);
    auto tableSchema = tpch::getTableSchema(table);
    auto columnNames = tableSchema->names();
    int32_t numSplits = 1;
    if (tableName != "nation" && tableName != "region" &&
        FLAGS_tpch_scale > 1) {
      numSplits = std::min<int32_t>(FLAGS_tpch_scale, 200);
    }
    common::CompressionKind compression =
        common::CompressionKind::CompressionKind_SNAPPY;
    auto builder = PlanBuilder().tpchTableScan(
        table, std::move(columnNames), FLAGS_tpch_scale);

    auto writer = PlanBuilder::TableWriterBuilder(builder)
                      .outputDirectoryPath(tableDirectory)
                      .fileFormat(dwio::common::FileFormat::PARQUET)
                      .compressionKind(compression);
    auto plan = writer.endTableWriter().planNode();

    std::vector<exec::Split> splits;
    for (auto nthSplit = 0; nthSplit < numSplits; ++nthSplit) {
      splits.push_back(
          exec::Split(std::make_shared<connector::tpch::TpchConnectorSplit>(
              kTpchConnectorId, numSplits, nthSplit)));
    }
    int32_t numDrivers =
        std::min<int32_t>(numSplits, std::thread::hardware_concurrency());
    LOG(INFO) << "Creating dataset SF=" << FLAGS_tpch_scale
              << " numSplits=" << numSplits << " numDrivers=" << numDrivers
              << " hw concurrency=" << std::thread::hardware_concurrency();
    auto rows = AssertQueryBuilder(plan)
                    .splits(std::move(splits))
                    .maxDrivers(numDrivers)
                    .copyResults(pool.get());
  }
}

std::shared_ptr<Task> ParquetTpchTest::assertQuery(
    const TpchPlan& tpchPlan,
    const std::string& duckQuery,
    const std::optional<std::vector<uint32_t>>& sortingKeys) const {
  bool noMoreSplits = false;
  constexpr int kNumSplits = 10;
  constexpr int kNumDrivers = 4;
  auto addSplits = [&](TaskCursor* taskCursor) {
    auto& task = taskCursor->task();
    if (!noMoreSplits) {
      for (const auto& entry : tpchPlan.dataFiles) {
        for (const auto& path : entry.second) {
          auto const splits = HiveConnectorTestBase::makeHiveConnectorSplits(
              path, kNumSplits, tpchPlan.dataFileFormat);
          for (const auto& split : splits) {
            task->addSplit(entry.first, Split(split));
          }
        }
        task->noMoreSplits(entry.first);
      }
    }
    noMoreSplits = true;
  };
  CursorParameters params;
  params.maxDrivers = kNumDrivers;
  params.planNode = tpchPlan.plan;
  return exec::test::assertQuery(
      params, addSplits, duckQuery, *duckDb_, sortingKeys);
}

} // namespace facebook::velox::optimizer::test
