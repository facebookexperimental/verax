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
#include "axiom/optimizer/tests/PlanMatcher.h"
#include "axiom/optimizer/tests/PlanTest.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace lp = facebook::velox::logical_plan;

namespace facebook::velox::optimizer {
namespace {

TEST_F(PlanTest, unnest) {
  const std::vector<std::string> names{"x", "a_a_y", "a_a_z"};

  auto rowVector = makeRowVector(
      names,
      {
          makeFlatVector<int64_t>({
              7,
              10,
              8,
              9,
              10,
          }),
          makeNestedArrayVectorFromJson<int64_t>({
              "[[10, 20, 30], [100, 200, 300]]",
              "[[1, 3, 2], [1, 3, 2]]",
              "[[100, 200, 300], [10, 20, 30]]",
              "[[0, 0, 0], [0, 0, 0]]",
              "[[1, 3, 2], [1, 3, 2]]",
          }),
          makeNestedArrayVectorFromJson<int64_t>({
              "[[10, 30], [100, 300]]",
              "[[2, 1], [1, 2]]",
              "[[100, 300], [10, 30]]",
              "[[0, 0], [0, 0]]",
              "[[2, 1], [1, 2]]",
          }),
      });
  // We need to test the following cases:
  //  If something is after unnest
  //  it can depend and not depend on unnested columns.
  //  And we should also check that any expressions are allowed inside unnest,
  //  not only input column references.
  // - unnest
  // - unnest after unnest
  // - project before and after unnest
  // -- after this we will start to use project to simplify plans --
  // - filter before and after unnest
  // - group by before and after unnest
  // - order by before and after unnest
  // - limit before and after unnest
  // - join before and after unnest
  // - there's no extra columns in projections before unnest

  {
    SCOPED_TRACE("unnest");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .build();

    auto referencePlanUnnest =
        exec::test::PlanBuilder{}
            .values({rowVector})
            .project({
                "x",
                "a_a_y",
                "a_a_z",
                "array_distinct(a_a_y) AS a_a_y_d",
                "array_distinct(a_a_z) AS a_a_z_d",
            })
            .unnest({"x", "a_a_y", "a_a_z"}, {"a_a_y_d", "a_a_z_d"})
            .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("unnest after unnest");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .unnest({
                lp::Sql("array_distinct(a_y)").unnestAs("y"),
                lp::Sql("array_distinct(a_z)").unnestAs("z"),
            })
            .build();

    auto referencePlanUnnest =
        exec::test::PlanBuilder{}
            .values({rowVector})
            .project({
                "x",
                "a_a_y",
                "a_a_z",
                "array_distinct(a_a_y) AS a_a_y_d",
                "array_distinct(a_a_z) AS a_a_z_d",
            })
            .unnest({"x", "a_a_y", "a_a_z"}, {"a_a_y_d", "a_a_z_d"})
            .project({
                "x",
                "a_a_y",
                "a_a_z",
                "a_a_y_d_e",
                "a_a_z_d_e",
                "array_distinct(a_a_y_d_e) AS a_y_d",
                "array_distinct(a_a_z_d_e) AS a_z_d",
            })
            .unnest(
                {"x", "a_a_y", "a_a_z", "a_a_y_d_e", "a_a_z_d_e"},
                {"a_y_d", "a_z_d"})
            .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("project before unnest");

    auto logicalPlanUnnest = lp::PlanBuilder{}
                                 .values({rowVector})
                                 .project({
                                     "x + 1 AS x1",
                                     "array_distinct(a_a_y) AS a_a_y_d",
                                     "array_distinct(a_a_z) AS a_a_z_d",
                                 })
                                 .unnest({
                                     lp::Col("a_a_y_d").unnestAs("a_y"),
                                     lp::Col("a_a_z_d").unnestAs("a_z"),
                                 })
                                 .build();

    auto referencePlanUnnest =
        exec::test::PlanBuilder{}
            .values({rowVector})
            .project({
                "x + 1 AS x1",
                "array_distinct(a_a_y) AS a_a_y_d",
                "array_distinct(a_a_z) AS a_a_z_d",
            })
            .unnest({"x1", "a_a_y_d", "a_a_z_d"}, {"a_a_y_d", "a_a_z_d"})
            .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("project after unnest (independent on unnested columns)");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x + 1 AS x1", "a_y"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .project({"x + 1 AS x1", "a_a_y_d_e"})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("project after unnest (dependent on unnested columns)");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x", "array_distinct(a_y) AS a_y_d"})
            .build();

    auto referencePlanUnnest =
        exec::test::PlanBuilder{}
            .values({rowVector})
            .project({
                "x",
                "array_distinct(a_a_y) AS a_a_y_d",
                "array_distinct(a_a_z) AS a_a_z_d",
            })
            .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
            .project({"x", "array_distinct(a_a_y_d_e) AS a_y_d"})
            .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("filter before unnest");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .filter("x % 2 = 0")
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x", "a_y"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .filter("x % 2 = 0")
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .project({"x", "a_a_y_d_e"})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("filter after unnest (independent on unnested columns)");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .filter("x % 2 = 0")
            .project({"x", "a_y"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .filter("x % 2 = 0")
                                   .project({"x", "a_a_y_d_e"})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("filter after unnest (dependent on unnested columns)");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .unnest({
                lp::Sql("array_distinct(a_y)").unnestAs("y"),
                lp::Sql("array_distinct(a_z)").unnestAs("z"),
            })
            .filter("y % 2 = 0")
            .project({"x", "y"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y_d_e) AS a_y",
                                       "array_distinct(a_a_z_d_e) AS a_z",
                                   })
                                   .unnest({"x"}, {"a_y", "a_z"})
                                   .filter("a_y_e % 2 = 0")
                                   .project({"x", "a_y_e"})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("group by before unnest");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .aggregate(names, {})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x", "a_y", "a_z"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .singleAggregation(names, {})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("group by after unnest");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .aggregate({"x", "a_y", "a_z"}, {})
            .build();

    auto referencePlanUnnest =
        exec::test::PlanBuilder{}
            .values({rowVector})
            .project({
                "x",
                "array_distinct(a_a_y) AS a_a_y_d",
                "array_distinct(a_a_z) AS a_a_z_d",
            })
            .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
            .singleAggregation({"x", "a_a_y_d_e", "a_a_z_d_e"}, {})
            .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("order by before unnest");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .orderBy(names)
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x", "a_y", "a_z"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .orderBy(names, {})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("order by after unnest (independent on unnested columns)");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x", "a_y", "a_z"})
            .orderBy({"x"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .orderBy({"x"}, {})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("order by after unnest (dependent on unnested columns)");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x", "a_y", "a_z"})
            .orderBy({"x", "a_y", "a_z"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .orderBy({"x", "a_a_y_d_e", "a_a_z_d_e"}, {})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("limit before unnest");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .limit(1, 1)
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x", "a_y", "a_z"})
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .limit(1, 1, {})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("limit after unnest");

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .project({"x", "a_y", "a_z"})
            .limit(1, 1)
            .build();

    auto referencePlanUnnest = exec::test::PlanBuilder{}
                                   .values({rowVector})
                                   .project({
                                       "x",
                                       "array_distinct(a_a_y) AS a_a_y_d",
                                       "array_distinct(a_a_z) AS a_a_z_d",
                                   })
                                   .unnest({"x"}, {"a_a_y_d", "a_a_z_d"})
                                   .limit(1, 1, {})
                                   .planNode();

    checkSame(logicalPlanUnnest, referencePlanUnnest);
  }
  {
    SCOPED_TRACE("join before unnest (independent on unnested columns)");

    const std::vector<std::string> expectedNames{"x1", "a_y1", "a_z2"};

    lp::PlanBuilder::Context ctx;
    auto logicalPlanUnnest =
        lp::PlanBuilder{ctx}
            .values({rowVector})
            .project({"x AS x1", "a_a_y AS a_a_y1", "a_a_z AS a_a_z1"})
            .join(
                lp::PlanBuilder{ctx}
                    .values({rowVector})
                    .project({"x AS x2", "a_a_y AS a_a_y2", "a_a_z AS a_a_z2"}),
                "x1 = x2",
                lp::JoinType::kInner)
            .unnest({
                lp::Sql("array_distinct(a_a_y1)").unnestAs("a_y1"),
                lp::Sql("array_distinct(a_a_z1)").unnestAs("a_z1"),
            })
            .unnest({
                lp::Sql("array_distinct(a_a_y2)").unnestAs("a_y2"),
                lp::Sql("array_distinct(a_a_z2)").unnestAs("a_z2"),
            })
            .project(expectedNames)
            .build();
    auto plan = toSingleNodePlan(logicalPlanUnnest);

    auto matcher = core::PlanMatcherBuilder{}
                       .values()
                       .hashJoin(core::PlanMatcherBuilder{}.values().build())
                       .project()
                       .project()
                       .unnest()
                       .project()
                       .unnest()
                       .project()
                       .build();
    ASSERT_TRUE(matcher->match(plan));
    ASSERT_EQ(plan->outputType()->names(), expectedNames);
  }
  {
    SCOPED_TRACE("join before unnest (dependent on unnested columns)");

    const std::vector<std::string> expectedNames{"x1", "a_y1", "a_z2"};

    lp::PlanBuilder::Context ctx;
    auto logicalPlanUnnest =
        lp::PlanBuilder{ctx}
            .values({rowVector})
            .project({"x AS x1", "a_a_y AS a_a_y1", "a_a_z AS a_a_z1"})
            .join(
                lp::PlanBuilder{ctx}
                    .values({rowVector})
                    .project({"x AS x2", "a_a_y AS a_a_y2", "a_a_z AS a_a_z2"}),
                "a_a_y1 = a_a_y2",
                lp::JoinType::kInner)
            .unnest({
                lp::Sql("array_distinct(a_a_y1)").unnestAs("a_y1"),
                lp::Sql("array_distinct(a_a_z1)").unnestAs("a_z1"),
            })
            .unnest({
                lp::Sql("array_distinct(a_a_y2)").unnestAs("a_y2"),
                lp::Sql("array_distinct(a_a_z2)").unnestAs("a_z2"),
            })
            .project(expectedNames)
            .build();
    auto plan = toSingleNodePlan(logicalPlanUnnest);

    auto matcher = core::PlanMatcherBuilder{}
                       .values()
                       .hashJoin(core::PlanMatcherBuilder{}.values().build())
                       .project()
                       .project()
                       .unnest()
                       .project()
                       .unnest()
                       .project()
                       .build();
    ASSERT_TRUE(matcher->match(plan));
    ASSERT_EQ(plan->outputType()->names(), expectedNames);
  }
  {
    SCOPED_TRACE("join after unnest (independent on unnested columns)");

    const std::vector<std::string> expectedNames{"x1", "a_y1", "a_z2"};

    lp::PlanBuilder::Context ctx;
    auto logicalPlanUnnest =
        lp::PlanBuilder{ctx}
            .values({rowVector})
            .project({"x AS x1", "a_a_y AS a_a_y1", "a_a_z AS a_a_z1"})
            .unnest({
                lp::Sql("array_distinct(a_a_y1)").unnestAs("a_y1"),
                lp::Sql("array_distinct(a_a_z1)").unnestAs("a_z1"),
            })
            .join(
                lp::PlanBuilder{ctx}
                    .values({rowVector})
                    .project({"x AS x2", "a_a_y AS a_a_y2", "a_a_z AS a_a_z2"})
                    .unnest({
                        lp::Sql("array_distinct(a_a_y2)").unnestAs("a_y2"),
                        lp::Sql("array_distinct(a_a_z2)").unnestAs("a_z2"),
                    }),
                "x1 = x2",
                lp::JoinType::kInner)
            .project(expectedNames)
            .build();
    auto plan = toSingleNodePlan(logicalPlanUnnest);

    auto matcher = core::PlanMatcherBuilder{}
                       .values()
                       .project()
                       .unnest()
                       .project()
                       .hashJoin(core::PlanMatcherBuilder{}
                                     .values()
                                     .project()
                                     .unnest()
                                     .project()
                                     .build())
                       .build();
    ASSERT_TRUE(matcher->match(plan));
    ASSERT_EQ(plan->outputType()->names(), expectedNames);
  }
  {
    SCOPED_TRACE("join after unnest (dependent on unnested columns)");

    const std::vector<std::string> expectedNames{"x1", "a_y1", "a_z2"};

    lp::PlanBuilder::Context ctx;
    auto logicalPlanUnnest =
        lp::PlanBuilder{ctx}
            .values({rowVector})
            .project({"x AS x1", "a_a_y AS a_a_y1", "a_a_z AS a_a_z1"})
            .unnest({
                lp::Sql("array_distinct(a_a_y1)").unnestAs("a_y1"),
                lp::Sql("array_distinct(a_a_z1)").unnestAs("a_z1"),
            })
            .join(
                lp::PlanBuilder{ctx}
                    .values({rowVector})
                    .project({"x AS x2", "a_a_y AS a_a_y2", "a_a_z AS a_a_z2"})
                    .unnest({
                        lp::Sql("array_distinct(a_a_y2)").unnestAs("a_y2"),
                        lp::Sql("array_distinct(a_a_z2)").unnestAs("a_z2"),
                    }),
                "a_a_y1 = a_a_y2",
                lp::JoinType::kInner)
            .project(expectedNames)
            .build();
    auto plan = toSingleNodePlan(logicalPlanUnnest);

    auto matcher = core::PlanMatcherBuilder{}
                       .values()
                       .project()
                       .unnest()
                       .project()
                       .hashJoin(core::PlanMatcherBuilder{}
                                     .values()
                                     .project()
                                     .unnest()
                                     .project()
                                     .build())
                       .build();
    ASSERT_TRUE(matcher->match(plan));
    ASSERT_EQ(plan->outputType()->names(), expectedNames);
  }
  {
    SCOPED_TRACE("there's no extra columns in projections before unnest");

    const std::vector<std::string> expectedNames{"x", "y"};

    auto logicalPlanUnnest =
        lp::PlanBuilder{}
            .values({rowVector})
            .unnest({
                lp::Sql("array_distinct(a_a_y)").unnestAs("a_y"),
                lp::Sql("array_distinct(a_a_z)").unnestAs("a_z"),
            })
            .unnest({
                lp::Sql("array_distinct(a_y)").unnestAs("y"),
                lp::Sql("array_distinct(a_z)").unnestAs("z"),
            })
            .project(expectedNames)
            .build();
    auto plan = toSingleNodePlan(logicalPlanUnnest);

    // names like this because they're autogenerated by the optimizer
    auto matcher =
        core::PlanMatcherBuilder()
            .values()
            .project({"x", "array_distinct(a_a_y)", "array_distinct(a_a_z)"})
            .unnest({"x"}, {"__r3", "__r4"})
            .project({"x", "array_distinct(a_y)", "array_distinct(a_z)"})
            .unnest({"x"}, {"__r3", "__r4"})
            .project(expectedNames)
            .build();
    ASSERT_TRUE(matcher->match(plan));
    ASSERT_EQ(plan->outputType()->names(), expectedNames);
  }
}

} // namespace
} // namespace facebook::velox::optimizer
