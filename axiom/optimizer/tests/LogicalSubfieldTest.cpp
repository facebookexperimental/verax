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

#include "axiom/logical_plan/PlanBuilder.h"
#include "axiom/optimizer/FunctionRegistry.h"
#include "axiom/optimizer/tests/FeatureGen.h"
#include "axiom/optimizer/tests/Genies.h"
#include "axiom/optimizer/tests/PlanMatcher.h"
#include "axiom/optimizer/tests/QueryTestBase.h"
#include "axiom/optimizer/tests/utils/DfFunctions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorMaker.h"

DEFINE_string(subfield_data_path, "", "Data directory for subfield test data");

DECLARE_int32(num_workers);

using namespace facebook::velox::optimizer::test;
using namespace facebook::velox::exec::test;

namespace lp = facebook::velox::logical_plan;

namespace facebook::velox::optimizer {

class LogicalSubfieldTest : public QueryTestBase,
                            public testing::WithParamInterface<int32_t> {
 protected:
  static void SetUpTestCase() {
    testDataPath_ = FLAGS_subfield_data_path;
    LocalRunnerTestBase::localFileFormat_ = "dwrf";
    LocalRunnerTestBase::SetUpTestCase();
    registerDfFunctions();
    registerRowUdfs();
  }

  static void TearDownTestCase() {
    LocalRunnerTestBase::TearDownTestCase();
  }

  void SetUp() override {
    QueryTestBase::SetUp();
    switch (GetParam()) {
      case 1:
        optimizerOptions_ = OptimizerOptions();
        break;
      case 2:
        optimizerOptions_ = OptimizerOptions{.pushdownSubfields = true};
        break;
      case 3:
        optimizerOptions_ = OptimizerOptions{.pushdownSubfields = true};
        optimizerOptions_.mapAsStruct["features"] = {
            "float_features", "id_list_features", "id_score_list_features"};
        break;
      default:
        FAIL();
        break;
    }
  }

  void TearDown() override {
    QueryTestBase::TearDown();
  }

  void declareGenies() {
    auto genieType = makeGenieType();
    std::vector<TypePtr> genieArgs = genieType->children();
    planner().registerScalarFunction("genie", genieArgs, genieType);
    planner().registerScalarFunction("exploding_genie", genieArgs, genieType);
    registerGenieUdfs();

    auto metadata = std::make_unique<FunctionMetadata>();
    metadata->fieldIndexForArg = {1, 2, 3};
    metadata->argOrdinal = {1, 2, 3};

    auto* registry = FunctionRegistry::instance();
    registry->registerFunction(
        "genie", std::make_unique<FunctionMetadata>(*metadata));

    auto explodingMetadata = std::make_unique<FunctionMetadata>(*metadata);
    explodingMetadata->logicalExplode = logicalExplodeGenie;
    registry->registerFunction("exploding_genie", std::move(explodingMetadata));
  }

  static std::unordered_map<PathCP, lp::ExprPtr> logicalExplodeGenie(
      const lp::CallExpr* call,
      std::vector<PathCP>& paths) {
    // This function understands paths like [1][cc], [2][cc],
    // .__3[cc] where __x is an ordinal field reference and cc is an integer
    // constant. If there is an empty path or a path with just one step, this
    // returns empty, meaning nothing is exploded. If the paths are longer, e.g.
    // idslf[11][1], then the trailing part is ignored. The returned map will
    // have the expression for each distinct path that begins with one of [1],
    // [2], [3] followed by an integer subscript.
    std::unordered_map<PathCP, lp::ExprPtr> result;
    for (auto& path : paths) {
      auto& steps = path->steps();
      if (steps.size() < 2) {
        return {};
      }

      std::vector<Step> prefixSteps = {steps[0], steps[1]};
      auto prefixPath = toPath(std::move(prefixSteps));
      if (result.count(prefixPath)) {
        // There already is an expression for this path.
        continue;
      }
      VELOX_CHECK(steps.front().kind == StepKind::kField);
      auto nth = steps.front().id;
      VELOX_CHECK_LE(nth, 3);
      auto args = call->inputs();

      // Here, for the sake of example, we make every odd key return identity.
      if (steps[1].id % 2 == 1) {
        result[prefixPath] = stepToLogicalPlanGetter(steps[1], args[nth]);
        continue;
      }

      // For changed float_features, we add the feature id to the value.
      if (nth == 1) {
        result[prefixPath] = std::make_shared<lp::CallExpr>(
            REAL(),
            "plus",
            std::vector<lp::ExprPtr>{
                stepToLogicalPlanGetter(steps[1], args[nth]),
                std::make_shared<lp::ConstantExpr>(
                    REAL(),
                    std::make_shared<variant>(
                        static_cast<float>(steps[1].id)))});
        continue;
      }

      // For changed id list features, we do array_distinct on the list.
      if (nth == 2) {
        result[prefixPath] = std::make_shared<lp::CallExpr>(
            ARRAY(BIGINT()),
            "array_distinct",
            std::vector<lp::ExprPtr>{
                stepToLogicalPlanGetter(steps[1], args[nth])});
        continue;
      }

      // Access to idslf. Identity.
      result[prefixPath] = stepToLogicalPlanGetter(steps[1], args[nth]);
    }
    return result;
  }

  std::vector<RowVectorPtr> extractAndIncrementIdList(
      const std::vector<RowVectorPtr>& vectors,
      int32_t key) {
    std::vector<RowVectorPtr> result;
    velox::test::VectorMaker vectorMaker(pool());

    for (auto& row : vectors) {
      const auto* map = row->childAt("id_list_features")->as<MapVector>();
      const auto* keys = map->mapKeys()->as<FlatVector<int32_t>>();
      const auto* values = map->mapValues()->as<ArrayVector>();

      auto ids =
          BaseVector::create<ArrayVector>(ARRAY(BIGINT()), row->size(), pool());
      auto* elements = ids->elements()->as<FlatVector<int64_t>>();
      for (auto i = 0; i < row->size(); ++i) {
        bool found = false;
        const auto mapOffset = map->offsetAt(i);
        const auto mapSize = map->sizeAt(i);
        for (auto k = mapOffset; k < mapOffset + mapSize; ++k) {
          if (keys->valueAt(k) == key) {
            ids->copy(values, i, k, 1);

            const auto arrayOffset = ids->offsetAt(i);
            const auto arraySize = ids->sizeAt(i);
            for (auto e = arrayOffset; e < arrayOffset + arraySize; ++e) {
              elements->set(e, elements->valueAt(e) + 1);
            }
            found = true;
            break;
          }
        }
        if (!found) {
          ids->setNull(i, true);
        }
      }
      result.push_back(vectorMaker.rowVector({ids}));
    }

    return result;
  }

  void testParallelExpr(FeatureOptions& opts, const RowTypePtr& rowType) {
    core::PlanNodePtr veloxPlan;
    // No randoms in test expr, different runs must come out the same.
    opts.randomPct = 0;

    {
      std::vector<std::string> names;
      std::vector<core::TypedExprPtr> exprs;

      opts.rng.seed(1);
      makeExprs(opts, names, exprs);

      auto builder = PlanBuilder()
                         .tableScan("features", rowType)
                         .addNode([&](std::string id, auto node) {
                           return std::make_shared<core::ProjectNode>(
                               id, std::move(names), std::move(exprs), node);
                         });
      veloxPlan = builder.planNode();
    }

    std::vector<std::string> names;
    std::vector<lp::ExprPtr> exprs;

    opts.rng.seed(1);
    makeLogicalExprs(opts, names, exprs);

    lp::PlanBuilder::Context ctx;
    auto logicalPlan = std::make_shared<lp::ProjectNode>(
        ctx.planNodeIdGenerator->next(),
        lp::PlanBuilder(ctx)
            .tableScan(kHiveConnectorId, "features", rowType->names())
            .build(),
        std::move(names),
        std::move(exprs));

    optimizerOptions_.parallelProjectWidth = 8;
    auto fragmentedPlan = planVelox(logicalPlan);

    auto* parallelProject = core::PlanNode::findFirstNode(
        extractPlanNode(fragmentedPlan).get(), [](const core::PlanNode* node) {
          return dynamic_cast<const core::ParallelProjectNode*>(node) !=
              nullptr;
        });

    ASSERT_TRUE(parallelProject != nullptr);

    assertSame(veloxPlan, fragmentedPlan);
  }

  std::string subfield(const std::string& first, const std::string& rest = "")
      const {
    return GetParam() == 3 ? fmt::format(".{}{}", first, rest)
                           : fmt::format("[{}]{}", first, rest);
  };

  void testMakeRowFromMap() {
    lp::PlanBuilder::Context ctx(getQueryCtx(), resolveDfFunction);
    auto logicalPlan =
        lp::PlanBuilder(ctx)
            .tableScan(
                exec::test::kHiveConnectorId,
                "features",
                {"float_features", "id_list_features"})
            .unionAll(lp::PlanBuilder(ctx).tableScan(
                exec::test::kHiveConnectorId,
                "features",
                {"float_features", "id_list_features"}))
            .project(
                {"make_row_from_map(float_features, array[10010, 10020, 10030], array['f1', 'f2', 'f3']) as r"})
            .project(
                {"make_named_row('f1b', r.f1 + 1::REAL, 'f2b', r.f2 + 2::REAL) as named"})
            .filter("named.f1b < 10000::REAL")
            .project({"make_named_row('rf2', named.f2b * 2::REAL) as fin"})
            .build();

    const auto plan = toSingleNodePlan(logicalPlan);

    verifyRequiredSubfields(
        plan, {{"float_features", {subfield("10010"), subfield("10020")}}});

    auto matcher =
        core::PlanMatcherBuilder()
            .hiveScan("features", {}, "float_features[10010] + 1 < 10000")
            .project()
            .localPartition(
                core::PlanMatcherBuilder()
                    .hiveScan(
                        "features", {}, "float_features[10010] + 1 < 10000")
                    .project()
                    .build())
            .project()
            .build();

    ASSERT_TRUE(matcher->match(plan));
  }

  core::PlanNodePtr toSingleNodePlan(
      const lp::LogicalPlanNodePtr& logicalPlan) {
    gflags::FlagSaver saver;
    FLAGS_num_workers = 1;

    auto plan = planVelox(logicalPlan).plan;

    EXPECT_EQ(1, plan->fragments().size());
    return plan->fragments().at(0).fragment.planNode;
  }

  void createTable(
      const std::string& name,
      const std ::vector<RowVectorPtr>& vectors,
      const std::shared_ptr<dwrf::Config>& config =
          std::make_shared<dwrf::Config>()) {
    auto fs = filesystems::getFileSystem(testDataPath_, {});
    fs->mkdir(fmt::format("{}/{}", testDataPath_, name));

    const auto filePath =
        fmt::format("{}/{}/{}.dwrf", testDataPath_, name, name);
    writeToFile(filePath, vectors, config);
    tablesCreated();
  }

  // TODO Move to PlanMatcher.
  static void verifyRequiredSubfields(
      const core::PlanNodePtr& plan,
      const std::unordered_map<std::string, std::vector<std::string>>&
          expectedSubfields) {
    auto* scanNode = core::PlanNode::findFirstNode(
        plan.get(), [](const core::PlanNode* node) {
          auto scan = dynamic_cast<const core::TableScanNode*>(node);
          return scan != nullptr;
        });

    ASSERT_TRUE(scanNode != nullptr);

    SCOPED_TRACE(scanNode->toString(true, true));

    const auto& assignments =
        dynamic_cast<const core::TableScanNode*>(scanNode)->assignments();
    ASSERT_EQ(assignments.size(), expectedSubfields.size());

    for (const auto& [_, handle] : assignments) {
      auto hiveHandle =
          dynamic_cast<const connector::hive::HiveColumnHandle*>(handle.get());
      ASSERT_TRUE(hiveHandle != nullptr);

      const auto& name = hiveHandle->name();
      const auto& subfields = hiveHandle->requiredSubfields();

      auto it = expectedSubfields.find(name);
      ASSERT_TRUE(it != expectedSubfields.end())
          << "Unexpected column: " << name;
      const auto& expected = it->second;

      ASSERT_EQ(subfields.size(), expected.size()) << hiveHandle->toString();

      for (auto i = 0; i < subfields.size(); ++i) {
        EXPECT_EQ(
            subfields[i].toString(), fmt::format("{}{}", name, expected[i]));
      }
    }
  }

  core::PlanNodePtr extractPlanNode(const PlanAndStats& plan) {
    return plan.plan->fragments().at(0).fragment.planNode;
  }
};

TEST_P(LogicalSubfieldTest, structs) {
  auto structType =
      ROW({"s1", "s2", "s3"},
          {BIGINT(), ROW({"s2s1"}, {BIGINT()}), ARRAY(BIGINT())});
  auto rowType = ROW({"s", "i"}, {structType, BIGINT()});
  auto vectors = makeVectors(rowType, 10, 10);
  createTable("structs", vectors);

  auto logicalPlan =
      lp::PlanBuilder()
          .tableScan(kHiveConnectorId, "structs", rowType->names())
          .project({"s.s1", "s.s3[1]"})
          .build();

  auto fragmentedPlan = planVelox(logicalPlan);

  // t2.s = HiveColumnHandle [... requiredSubfields: [ s.s1 s.s3[0] ]]
  verifyRequiredSubfields(
      extractPlanNode(fragmentedPlan), {{"s", {".s1", ".s3[1]"}}});

  auto referencePlan = PlanBuilder()
                           .tableScan("structs", rowType)
                           .project({"s.s1", "s.s3[1]"})
                           .planNode();

  assertSame(referencePlan, fragmentedPlan);
}

TEST_P(LogicalSubfieldTest, maps) {
  FeatureOptions opts;
  opts.rng.seed(1);
  auto vectors = makeFeatures(1, 100, opts, pool_.get());

  const auto rowType = vectors[0]->rowType();
  const auto fields = rowType->names();

  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::FLATTEN_MAP, true);
  config->set<const std::vector<uint32_t>>(
      dwrf::Config::MAP_FLAT_COLS, {2, 3, 4});

  createTable("features", vectors, config);

  testMakeRowFromMap();

  {
    lp::PlanBuilder::Context ctx;
    auto builder =
        lp::PlanBuilder(ctx)
            .tableScan(kHiveConnectorId, "features", fields)
            .project({"uid", "float_features as ff"})
            .join(
                lp::PlanBuilder(ctx)
                    .tableScan(kHiveConnectorId, "features", fields)
                    .filter(
                        "uid % 2 = 1 and cast(float_features[10300::int] as integer) % 2::int = 0::int")
                    .project({"uid as opt_uid", "float_features as opt_ff"}),
                "uid = opt_uid",
                lp::JoinType::kLeft)
            .project(
                {"uid",
                 "opt_uid",
                 "ff[10100::int] as f10",
                 "ff[10200::int] as f20",
                 "opt_ff[10100::int] as o10",
                 "opt_ff[10200::int] as o20"});

    auto plan = veloxString(planVelox(builder.build()).plan);
    // TODO Add verification.
  }
  {
    auto logicalPlan =
        lp::PlanBuilder()
            .tableScan(kHiveConnectorId, "features", fields)
            .project(
                {"float_features[10100::int] as f1",
                 "float_features[10200::int] as f2",
                 "id_score_list_features[200800::int][100000::BIGINT]"})
            .build();

    auto plan = extractPlanNode(planVelox(logicalPlan));
    verifyRequiredSubfields(
        plan,
        {
            {"float_features", {subfield("10100"), subfield("10200")}},
            {"id_score_list_features", {subfield("200800", "[100000]")}},
        });
  }
  {
    auto logicalPlan = lp::PlanBuilder()
                           .tableScan(kHiveConnectorId, "features", fields)
                           .project(
                               {"float_features[10000::int] as ff",
                                "id_score_list_features[200800::int] as sc1",
                                "id_list_features as idlf"})
                           .project({"sc1[1::BIGINT] + 1::REAL as score"})
                           .build();

    auto plan = extractPlanNode(planVelox(logicalPlan));
    verifyRequiredSubfields(
        plan,
        {
            {"id_score_list_features", {subfield("200800", "[1]")}},
        });
  }
  {
    auto logicalPlan = lp::PlanBuilder()
                           .tableScan(kHiveConnectorId, "features", fields)
                           .project(
                               {"float_features[10100::int] as ff",
                                "id_score_list_features[200800::int] as sc1",
                                "id_list_features as idlf",
                                "uid"})
                           .project(
                               {"sc1[1::BIGINT] + 1::REAL as score",
                                "idlf[cast(uid % 100 as INTEGER)] as any"})
                           .build();

    auto plan = extractPlanNode(planVelox(logicalPlan));
    verifyRequiredSubfields(
        plan,
        {
            {"uid", {}},
            {"id_score_list_features", {subfield("200800", "[1]")}},
            {"id_list_features", {"[*]"}},
        });
  }

  declareGenies();

  // Selected fields of genie are accessed. The uid and idslf args are not
  // accessed and should not be in the table scan.
  {
    auto logicalPlan =
        lp::PlanBuilder()
            .tableScan(kHiveConnectorId, "features", fields)
            .project(
                {"genie(uid, float_features, id_list_features, id_score_list_features) as g"})
            // Access some fields of the genie by name, others by index.
            .project(
                {"g.ff[10200::int] as f2",
                 "g[2][10100::int] as f11",
                 "g[2][10200::int] + 22::REAL  as f2b",
                 "g.idlf[201600::int] as idl100"})
            .build();

    auto plan = extractPlanNode(planVelox(logicalPlan));
    verifyRequiredSubfields(
        plan,
        {
            {"uid", {}},
            {"float_features", {subfield("10200"), subfield("10100")}},
            {"id_list_features", {subfield("201600")}},
        });
  }
  // All of genie is returned.
  {
    auto logicalPlan =
        lp::PlanBuilder()
            .tableScan(kHiveConnectorId, "features", fields)
            .project(
                {"genie(uid, float_features, id_list_features, id_score_list_features) as g"})
            .project(
                {"g",
                 "g[2][10100::int] as f10",
                 "g[2][10200::int] as f2",
                 "g[3][200600::int] as idl100"})
            .build();

    auto plan = extractPlanNode(planVelox(logicalPlan));
    verifyRequiredSubfields(
        plan,
        {
            {"uid", {}},
            {"float_features", {}},
            {"id_list_features", {}},
            {"id_score_list_features", {}},
        });
  }

  // We expect the genie to explode and the filters to be first.
  {
    auto builder =
        lp::PlanBuilder()
            .tableScan(kHiveConnectorId, "features", fields)
            .project(
                {"exploding_genie(uid, float_features, id_list_features, id_score_list_features) as g"})
            .project({"g[2] as ff", "g as gg"})
            .project(
                {"ff[10100::int] as f10",
                 "ff[10100::int] as f11",
                 "ff[10200::int] as f2",
                 "gg[2][10200::int] + 22::REAL as f2b",
                 "gg[3][200600::int] as idl100"})
            .filter("f10 < 10::REAL and f11 < 10::REAL");

    auto plan = veloxString(planVelox(builder.build()).plan);
    // TODO Add verification.
  }
  {
    auto builder =
        lp::PlanBuilder()
            .tableScan(kHiveConnectorId, "features", fields)
            .project(
                {"transform(id_list_features[201800::int], x -> x + 1) as ids"});

    auto result = runVelox(builder.build());
    auto expected = extractAndIncrementIdList(vectors, 201800);
    assertEqualResults(expected, result.results);
  }

  testParallelExpr(opts, rowType);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    LogicalSubfieldTests,
    LogicalSubfieldTest,
    testing::ValuesIn(std::vector<int32_t>{1, 2, 3}));

} // namespace facebook::velox::optimizer
