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

#include "optimizer/tests/QueryTestBase.h" //@manual

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/Exchange.h"

#include "optimizer/Plan.h" //@manual
#include "optimizer/SchemaResolver.h" //@manual
#include "optimizer/VeloxHistory.h" //@manual
#include "optimizer/connectors/ConnectorSplitSource.h" //@manual
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/expression/Expr.h"
#include "velox/serializers/PrestoSerializer.h"

DECLARE_string(data_path);

DEFINE_int32(optimizer_trace, 0, "Optimizer trace level");

DEFINE_bool(print_plan, false, "Print optimizer results");

DEFINE_int32(num_drivers, 4, "Number of drivers");
DEFINE_int32(num_workers, 4, "Number of in-process workers");

DEFINE_string(data_format, "parquet", "Data format");

namespace facebook::velox::optimizer::test {
using namespace facebook::velox::exec;

void QueryTestBase::SetUp() {
  exec::test::LocalRunnerTestBase::SetUp();
  connector_ = connector::getConnector(exec::test::kHiveConnectorId);
  rootPool_ = memory::memoryManager()->addRootPool("velox_sql");
  optimizerPool_ = rootPool_->addLeafChild("optimizer");
  schemaPool_ = rootPool_->addLeafChild("schema");

  parquet::registerParquetReaderFactory();
  dwrf::registerDwrfReaderFactory();
  exec::ExchangeSource::registerFactory(exec::test::createLocalExchangeSource);
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(VectorSerde::Kind::kPresto)) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs;
  auto copy = hiveConfig_;
  connectorConfigs[exec::test::kHiveConnectorId] =
      std::make_shared<config::ConfigBase>(std::move(copy));

  schemaQueryCtx_ = core::QueryCtx::create(
      driverExecutor_.get(),
      core::QueryConfig(config_),
      std::move(connectorConfigs),
      cache::AsyncDataCache::getInstance(),
      rootPool_->shared_from_this(),
      nullptr,
      "schema");
  common::SpillConfig spillConfig;
  common::PrefixSortConfig prefixSortConfig;

  schemaRootPool_ = rootPool_->addAggregateChild("schemaRoot");
  connectorQueryCtx_ = std::make_shared<connector::ConnectorQueryCtx>(
      schemaPool_.get(),
      schemaRootPool_.get(),
      schemaQueryCtx_->connectorSessionProperties(exec::test::kHiveConnectorId),
      &spillConfig,
      prefixSortConfig,
      std::make_unique<exec::SimpleExpressionEvaluator>(
          schemaQueryCtx_.get(), schemaPool_.get()),
      schemaQueryCtx_->cache(),
      "scan_for_schema",
      "schema",
      "N/a",
      0,
      schemaQueryCtx_->queryConfig().sessionTimezone());

  schema_ = std::make_shared<facebook::velox::optimizer::SchemaResolver>(
      connector_, "");
}

void QueryTestBase::TearDown() {
  queryCtx_.reset();
  connector::unregisterConnector(exec::test::kHiveConnectorId);
  connector_.reset();
  optimizerPool_.reset();
  schema_.reset();
  schemaQueryCtx_.reset();
  schemaPool_.reset();
  schemaRootPool_.reset();
  connectorQueryCtx_.reset();
  rootPool_.reset();
  LocalRunnerTestBase::TearDown();
}

void QueryTestBase::tablesCreated() {
  auto metadata = dynamic_cast<connector::hive::LocalHiveConnectorMetadata*>(
      connector_->metadata());
  VELOX_CHECK_NOT_NULL(metadata);
  metadata->reinitialize();
  planner_ = std::make_unique<core::DuckDbQueryPlanner>(optimizerPool_.get());
  auto& tables = metadata->tables();
  for (auto& pair : tables) {
    planner_->registerTable(pair.first, pair.second->rowType());
  }
  planner_->registerTableScan([this](
                                  const std::string& id,
                                  const std::string& name,
                                  const RowTypePtr& rowType,
                                  const std::vector<std::string>& columnNames) {
    return toTableScan(id, name, rowType, columnNames);
  });
  history_ = std::make_unique<facebook::velox::optimizer::VeloxHistory>();
}

core::PlanNodePtr QueryTestBase::toTableScan(
    const std::string& id,
    const std::string& name,
    const RowTypePtr& rowType,
    const std::vector<std::string>& columnNames) {
  using namespace connector::hive;
  auto handle = std::make_shared<HiveTableHandle>(
      exec::test::kHiveConnectorId,
      name,
      true,
      common::SubfieldFilters{},
      nullptr);
  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      assignments;

  auto table = connector_->metadata()->findTable(name);
  for (auto i = 0; i < rowType->size(); ++i) {
    auto projectedName = rowType->nameOf(i);
    auto& columnName = columnNames[i];
    VELOX_CHECK(
        table->columnMap().find(columnName) != table->columnMap().end(),
        "No column {} in {}",
        columnName,
        name);
    assignments[projectedName] = std::make_shared<HiveColumnHandle>(
        columnName,
        HiveColumnHandle::ColumnType::kRegular,
        rowType->childAt(i),
        rowType->childAt(i));
  }
  return std::make_shared<core::TableScanNode>(
      id, rowType, handle, assignments);
}

std::shared_ptr<runner::LocalRunner> QueryTestBase::runSql(
    const std::string& sql,
    std::vector<RowVectorPtr>* resultVector,
    std::string* planString,
    std::string* errorString,
    std::vector<exec::TaskStats>* statsReturn) {
  std::shared_ptr<runner::LocalRunner> runner;
  auto fragmentedPlan = planSql(sql, planString, errorString);
  if (!fragmentedPlan) {
    return nullptr;
  }
  try {
    runner = std::make_shared<runner::LocalRunner>(
        fragmentedPlan,
        queryCtx_,
        std::make_shared<connector::ConnectorSplitSourceFactory>());
    std::vector<RowVectorPtr> results;
    while (auto rows = runner->next()) {
      results.push_back(rows);
    }
    if (resultVector) {
      *resultVector = results;
    }
    auto stats = runner->stats();
    if (statsReturn) {
      *statsReturn = stats;
    }
    auto& fragments = fragmentedPlan->fragments();
    history_->recordVeloxExecution(nullptr, fragments, stats);
  } catch (const std::exception& e) {
    std::cerr << "Query terminated with: " << e.what() << std::endl;
    if (errorString) {
      *errorString = fmt::format("Runtime error: {}", e.what());
    }
    waitForCompletion(runner);
    return nullptr;
  }
  waitForCompletion(runner);
  return runner;
}

runner::MultiFragmentPlanPtr QueryTestBase::planSql(
    const std::string& sql,
    std::string* planString,
    std::string* errorString) {
  core::PlanNodePtr plan;
  try {
    plan = planner_->plan(sql);
  } catch (std::exception& e) {
    std::cerr << "parse error: " << e.what() << std::endl;
    if (errorString) {
      *errorString = fmt::format("Parse error: {}", e.what());
    }
    return nullptr;
  }
  return planVelox(plan, planString, errorString);
}

runner::MultiFragmentPlanPtr QueryTestBase::planVelox(
    const core::PlanNodePtr& plan,
    std::string* planString,
    std::string* errorString) {
  ++queryCounter_;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs;
  auto copy = hiveConfig_;
  connectorConfigs[exec::test::kHiveConnectorId] =
      std::make_shared<config::ConfigBase>(std::move(copy));
  queryCtx_ = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(config_),
      std::move(connectorConfigs),
      cache::AsyncDataCache::getInstance(),
      rootPool_->shared_from_this(),
      spillExecutor_.get(),
      fmt::format("query_{}", queryCounter_));

  // The default Locus for planning is the system and data of 'connector_'.
  optimizer::Locus locus(connector_->connectorId().c_str(), connector_.get());
  facebook::velox::optimizer::Optimization::PlanCostMap estimates;
  runner::MultiFragmentPlan::Options opts;
  opts.numWorkers = FLAGS_num_workers;
  opts.numDrivers = FLAGS_num_drivers;
  auto allocator = std::make_unique<HashStringAllocator>(optimizerPool_.get());
  auto context =
      std::make_unique<facebook::velox::optimizer::QueryGraphContext>(
          *allocator);
  facebook::velox::optimizer::queryCtx() = context.get();
  exec::SimpleExpressionEvaluator evaluator(
      queryCtx_.get(), optimizerPool_.get());
  runner::MultiFragmentPlanPtr fragmentedPlan;
  try {
    facebook::velox::optimizer::Schema veraxSchema(
        "test", schema_.get(), &locus);
    facebook::velox::optimizer::Optimization opt(
        *plan, veraxSchema, *history_, evaluator, FLAGS_optimizer_trace);
    auto best = opt.bestPlan();
    if (planString) {
      *planString = best->op->toString(true, false);
    }
    fragmentedPlan = opt.toVeloxPlan(best->op, opts);
  } catch (const std::exception& e) {
    facebook::velox::optimizer::queryCtx() = nullptr;
    std::cerr << "optimizer error: " << e.what() << std::endl;
    if (errorString) {
      *errorString = fmt::format("optimizer error: {}", e.what());
    }
    return nullptr;
  }
  facebook::velox::optimizer::queryCtx() = nullptr;
  return fragmentedPlan;
}

void QueryTestBase::waitForCompletion(
    const std::shared_ptr<runner::LocalRunner>& runner) {
  if (runner) {
    try {
      runner->waitForCompletion(50000);
    } catch (const std::exception& /*ignore*/) {
    }
  }
}

std::string QueryTestBase::veloxString(const std::string& sql) {
  auto plan = planSql(sql);
  VELOX_CHECK_NOT_NULL(plan);
  return veloxString(plan);
}

std::string QueryTestBase::veloxString(
    const runner::MultiFragmentPlanPtr& plan) {
  std::stringstream out;
  for (auto i = 0; i < plan->fragments().size(); ++i) {
    auto& fragment = plan->fragments()[i];
    out << "Fragment " << i << ":\n";
    auto* fragmentRoot = fragment.fragment.planNode.get();
    auto planNodeDetails = [&](const core::PlanNodeId& planNodeId,
                               const std::string& indentation,
                               std::stringstream& stream) {
      auto node = core::PlanNode::findFirstNode(
          fragmentRoot, [&](auto* node) { return node->id() == planNodeId; });
      if (!node) {
        return;
      }
      if (auto* scan = dynamic_cast<const core::TableScanNode*>(node)) {
        stream << std::endl;
        for (auto& pair : scan->assignments()) {
          auto* hiveColumn =
              dynamic_cast<const connector::hive::HiveColumnHandle*>(
                  pair.second.get());
          if (!hiveColumn) {
            continue;
          }
          stream << indentation << pair.first << " = " << hiveColumn->toString()
                 << std::endl;
        }
      }
    };

    out << fragment.fragment.planNode->toString(true, true, planNodeDetails)
        << std::endl;
  }
  out << std::endl;
  return out.str();
}

void QueryTestBase::expectRegexp(
    std::string& text,
    const std::string regexp,
    bool expect) {
  std::istringstream iss(text);
  std::string line;
  bool found = false;
  for (; std::getline(iss, line);) {
    if (RE2::PartialMatch(line, regexp)) {
      found = true;
      break;
    }
  }
  if (found != expect) {
    FAIL() << "Expected " << (expect == false ? " no " : "") << regexp << " in "
           << text;
  }
}

} // namespace facebook::velox::optimizer::test
