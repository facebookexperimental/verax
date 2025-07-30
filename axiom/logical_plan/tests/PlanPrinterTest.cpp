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

#include "axiom/logical_plan/PlanPrinter.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "axiom/logical_plan/PlanBuilder.h"
#include "axiom/optimizer/connectors/tests/TestConnector.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

namespace facebook::velox::logical_plan {
namespace {

class PlanPrinterTest : public testing::Test {
 protected:
  static constexpr auto kTestConnectorId = "test";

  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();

    auto connector =
        std::make_shared<connector::TestConnector>(kTestConnectorId);
    connector->addTable(
        "test", ROW({"a", "b", "c"}, {BIGINT(), DOUBLE(), VARCHAR()}));
    connector::registerConnector(connector);
  }

  void TearDown() override {
    connector::unregisterConnector(kTestConnectorId);
  }

  static std::vector<std::string> toLines(const LogicalPlanNodePtr& plan) {
    auto text = PlanPrinter::toText(*plan);
    return toLines(text);
  }

  static std::vector<std::string> toSummaryLines(
      const LogicalPlanNodePtr& plan,
      const PlanSummaryOptions& options = {}) {
    auto text = PlanPrinter::summarizeToText(*plan, options);
    return toLines(text);
  }

  static std::vector<std::string> toLines(const std::string& text) {
    LOG(INFO) << std::endl << text;

    std::vector<std::string> lines;
    folly::split("\n", text, lines);

    return lines;
  }
};

TEST_F(PlanPrinterTest, values) {
  auto plan = PlanBuilder()
                  .values(
                      ROW({"a"}, {BIGINT()}),
                      std::vector<Variant>{Variant::row({123LL})})
                  .filter("a > 10")
                  .with({"a + 2 as b"})
                  .project({"a + b as c"})
                  .sort({"c DESC"})
                  .limit(5)
                  .build();

  auto lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Limit"),
          testing::StartsWith("  - Sort"),
          testing::StartsWith("    - Project"),
          testing::StartsWith("        c := plus(a, b)"),
          testing::StartsWith("      - Project"),
          testing::StartsWith("          a := a"),
          testing::StartsWith("          b := plus(a, 2)"),
          testing::StartsWith("        - Filter: gt(a, 10)"),
          testing::StartsWith("          - Values"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  using namespace testing;

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          Eq("- LIMIT [5]: 1 fields: c BIGINT"),
          Eq("      limit: 5"),
          Eq("  - SORT [4]: 1 fields: c BIGINT"),
          Eq("    - PROJECT [3]: 1 fields: c BIGINT"),
          Eq("          expressions: call: 1, field: 2"),
          Eq("          functions: plus: 1"),
          Eq("          projections: 1 out of 1"),
          Eq("      - PROJECT [2]: 2 fields: a BIGINT, b BIGINT"),
          Eq("            expressions: call: 1, constant: 1, field: 2"),
          Eq("            functions: plus: 1"),
          Eq("            constants: BIGINT: 1"),
          Eq("            projections: 1 out of 2"),
          Eq("        - FILTER [1]: 1 fields: a BIGINT"),
          Eq("              predicate: gt(a, 10)"),
          Eq("              expressions: call: 1, constant: 1, field: 1"),
          Eq("              functions: gt: 1"),
          Eq("              constants: BIGINT: 1"),
          Eq("          - VALUES [0]: 1 fields: a BIGINT"),
          Eq("                rows: 1"),
          Eq("")));

  // Empty variant vector - zero rows:
  plan =
      PlanBuilder()
          .values(ROW({"a", "b"}, {BIGINT(), REAL()}), std::vector<Variant>{})
          .build();
  lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Values: 0 rows -> ROW<a:BIGINT,b:REAL>"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::Eq("- VALUES [0]: 2 fields: a BIGINT, b REAL"),
          testing::Eq("      rows: 0"),
          testing::Eq("")));
}

TEST_F(PlanPrinterTest, tableScan) {
  auto plan = PlanBuilder()
                  .tableScan(kTestConnectorId, "test", {"a", "b"})
                  .with({"cast(a as double) * b as c"})
                  .build();

  auto lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Project"),
          testing::StartsWith("    a := a"),
          testing::StartsWith("    b := b"),
          testing::StartsWith("    c := multiply(CAST(a AS DOUBLE), b)"),
          testing::StartsWith("  - TableScan"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::Eq("- PROJECT [1]: 3 fields: a BIGINT, b DOUBLE, c DOUBLE"),
          testing::Eq("      expressions: CAST: 1, call: 1, field: 4"),
          testing::Eq("      functions: multiply: 1"),
          testing::Eq("      projections: 1 out of 3"),
          testing::Eq("  - TABLE_SCAN [0]: 2 fields: a BIGINT, b DOUBLE"),
          testing::Eq("        table: test"),
          testing::Eq("        connector: test"),
          testing::Eq("")));
}

TEST_F(PlanPrinterTest, aggregate) {
  auto rowType = ROW({"a", "b"}, {INTEGER(), INTEGER()});
  std::vector<Variant> data{
      Variant::row({1, 10}),
      Variant::row({2, 20}),
      Variant::row({2, 21}),
      Variant::row({3, 30}),
  };

  auto plan =
      PlanBuilder()
          .values(rowType, data)
          .aggregate(
              {"a"}, {"sum(b) as total", "avg(b) as mean", "min(b + 1::int)"})
          .with({"total + 1", "mean * 0.3"})
          .build();

  auto lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Project"),
          testing::StartsWith("    a := a"),
          testing::StartsWith("    total := total"),
          testing::StartsWith("    mean := mean"),
          testing::StartsWith("    min := min"),
          testing::StartsWith("    expr := plus(total, 1)"),
          testing::StartsWith("    expr_0 := multiply(mean, 0.3)"),
          testing::StartsWith("  - Aggregate"),
          testing::StartsWith("      total := sum(b)"),
          testing::StartsWith("      mean := avg(b)"),
          testing::StartsWith("      min := min(plus(b, "),
          testing::StartsWith("    - Values"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  using namespace testing;

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          Eq("- PROJECT [2]: 6 fields: a INTEGER, total BIGINT, mean DOUBLE, min INTEGER, expr BIGINT, ..."),
          Eq("      expressions: call: 2, constant: 2, field: 6"),
          Eq("      functions: multiply: 1, plus: 1"),
          Eq("      constants: BIGINT: 1, DOUBLE: 1"),
          Eq("      projections: 2 out of 6"),
          Eq("  - AGGREGATE [1]: 4 fields: a INTEGER, total BIGINT, mean DOUBLE, min INTEGER"),
          Eq("    - VALUES [0]: 2 fields: a INTEGER, b INTEGER"),
          Eq("          rows: 4"),
          Eq("")));
}

TEST_F(PlanPrinterTest, union) {
  auto type = ROW({"a", "b"}, {INTEGER(), DOUBLE()});

  std::vector<Variant> data1{
      Variant::row({1, 1.1}),
      Variant::row({2, 2.2}),
  };

  std::vector<Variant> data2{
      Variant::row({3, 3.3}),
      Variant::row({4, 4.4}),
  };

  auto context = PlanBuilder::Context();
  auto plan = PlanBuilder(context)
                  .values(type, data1)
                  .unionAll(PlanBuilder(context).values(type, data2))
                  .with({"a::double + b"})
                  .build();

  auto lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Project"),
          testing::StartsWith("    a := a"),
          testing::StartsWith("    b := b"),
          testing::StartsWith("    expr := plus(CAST(a AS DOUBLE), b)"),
          testing::StartsWith("  - UNION ALL"),
          testing::StartsWith("    - Values"),
          testing::StartsWith("    - Values"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  using namespace testing;

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          Eq("- PROJECT [3]: 3 fields: a INTEGER, b DOUBLE, expr DOUBLE"),
          Eq("      expressions: CAST: 1, call: 1, field: 4"),
          Eq("      functions: plus: 1"),
          Eq("      projections: 1 out of 3"),
          Eq("  - UNION ALL [2]: 2 fields: a INTEGER, b DOUBLE"),
          Eq("    - VALUES [0]: 2 fields: a INTEGER, b DOUBLE"),
          Eq("          rows: 2"),
          Eq("    - VALUES [1]: 2 fields: a_0 INTEGER, b_1 DOUBLE"),
          Eq("          rows: 2"),
          Eq("")));
}

TEST_F(PlanPrinterTest, join) {
  auto leftType = ROW({"key", "v"}, {INTEGER(), INTEGER()});
  std::vector<Variant> leftData{
      Variant::row({1, 10}),
      Variant::row({2, 20}),
      Variant::row({2, 21}),
      Variant::row({3, 30}),
  };

  auto rightType = ROW({"key", "w"}, {INTEGER(), INTEGER()});
  std::vector<Variant> rightData{
      Variant::row({1, 11}),
      Variant::row({2, 22}),
  };

  auto context = PlanBuilder::Context();
  auto plan = PlanBuilder(context)
                  .values(leftType, leftData)
                  .as("l")
                  .join(
                      PlanBuilder(context).values(rightType, rightData).as("r"),
                      "l.key = r.key",
                      JoinType::kLeft)
                  .with({"l.v + r.w as z"})
                  .build();

  auto lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Project"),
          testing::StartsWith("    key := key"),
          testing::StartsWith("    v := v"),
          testing::StartsWith("    key_0 := key_0"),
          testing::StartsWith("    w := w"),
          testing::StartsWith("    z := plus(v, w)"),
          testing::StartsWith("  - Join LEFT: eq(key, key_0)"),
          testing::StartsWith("    - Values: 4 rows"),
          testing::StartsWith("    - Values: 2 rows"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  using namespace testing;

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          Eq("- PROJECT [3]: 5 fields: key INTEGER, v INTEGER, key_0 INTEGER, w INTEGER, z INTEGER"),
          Eq("      expressions: call: 1, field: 6"),
          Eq("      functions: plus: 1"),
          Eq("      projections: 1 out of 5"),
          Eq("  - JOIN LEFT [2]: 4 fields: key INTEGER, v INTEGER, key_0 INTEGER, w INTEGER"),
          Eq("        condition: eq(key, key_0)"),
          Eq("    - VALUES [0]: 2 fields: key INTEGER, v INTEGER"),
          Eq("          rows: 4"),
          Eq("    - VALUES [1]: 2 fields: key_0 INTEGER, w INTEGER"),
          Eq("          rows: 2"),
          Eq("")));
}

TEST_F(PlanPrinterTest, crossJoin) {
  auto leftType = ROW({"key", "v"}, {INTEGER(), INTEGER()});
  std::vector<Variant> leftData{
      Variant::row({1, 10}),
      Variant::row({2, 20}),
      Variant::row({3, 30}),
  };

  auto rightType = ROW({"key", "w"}, {INTEGER(), INTEGER()});
  std::vector<Variant> rightData{
      Variant::row({1, 11}),
      Variant::row({2, 22}),
  };

  PlanBuilder::Context context;
  auto plan = PlanBuilder(context)
                  .values(leftType, leftData)
                  .as("l")
                  .join(
                      PlanBuilder(context).values(rightType, rightData).as("r"),
                      "",
                      JoinType::kInner)
                  .with({"l.v + r.w as z"})
                  .build();

  auto lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Project"),
          testing::StartsWith("    key := key"),
          testing::StartsWith("    v := v"),
          testing::StartsWith("    key_0 := key_0"),
          testing::StartsWith("    w := w"),
          testing::StartsWith("    z := plus(v, w)"),
          testing::StartsWith("  - Join INNER:"), // Condition is empty
          testing::StartsWith("    - Values: 3 rows"),
          testing::StartsWith("    - Values: 2 rows"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  using namespace testing;

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          Eq("- PROJECT [3]: 5 fields: key INTEGER, v INTEGER, key_0 INTEGER, w INTEGER, z INTEGER"),
          Eq("      expressions: call: 1, field: 6"),
          Eq("      functions: plus: 1"),
          Eq("      projections: 1 out of 5"),
          Eq("  - JOIN INNER [2]: 4 fields: key INTEGER, v INTEGER, key_0 INTEGER, w INTEGER"),
          Eq("    - VALUES [0]: 2 fields: key INTEGER, v INTEGER"),
          Eq("          rows: 3"),
          Eq("    - VALUES [1]: 2 fields: key_0 INTEGER, w INTEGER"),
          Eq("          rows: 2"),
          Eq("")));
}

TEST_F(PlanPrinterTest, specialForms) {
  auto rowType =
      ROW({"a", "b", "c"},
          {INTEGER(), INTEGER(), ROW({"x", "y"}, {BOOLEAN(), REAL()})});
  std::vector<Variant> data{
      Variant::row({1, 10, Variant::row({true, 0.1f})}),
      Variant::row({2, 20, Variant::row({true, 0.2f})}),
      Variant::row({2, 21, Variant::row({true, 0.21f})}),
      Variant::row({3, 30, Variant::row({true, 0.3f})}),
  };

  auto plan =
      PlanBuilder()
          .values(rowType, data)
          .map({
              "a > b AND b > 10::int as a",
              "a < b OR b > 0::int as b",
              "cast(a as double) * 1.2",
              "try_cast(a / b as varchar)",
              "try(a / b)",
              "c.x",
              "coalesce(a, b, 33::int) as c",
              "if(a > b, a, b)",
              "case a when 1::int then 'a' when 2::int then 'b' else 'c' end",
              "c[2]",
          })
          .build();

  auto lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Project"),
          testing::StartsWith("    a := a_0"),
          testing::StartsWith("    b := b_1"),
          testing::StartsWith("    expr "),
          testing::StartsWith("    expr_2 "),
          testing::StartsWith("    expr_3 "),
          testing::StartsWith("    expr_4 "),
          testing::StartsWith("    c := c_5"),
          testing::StartsWith("    expr_6 "),
          testing::StartsWith("    expr_7 "),
          testing::StartsWith("    expr_8 "),
          testing::StartsWith("  - Project"),
          testing::StartsWith("      a_0 := AND"),
          testing::StartsWith("      b_1 := OR"),
          testing::StartsWith("      expr := multiply(CAST("),
          testing::StartsWith("      expr_2 := TRY_CAST"),
          testing::StartsWith("      expr_3 := TRY"),
          testing::StartsWith("      expr_4 := DEREFERENCE(c, x)"),
          testing::StartsWith("      c_5 := COALESCE"),
          testing::StartsWith("      expr_6 := IF"),
          testing::StartsWith("      expr_7 := SWITCH"),
          testing::StartsWith("      expr_8 := DEREFERENCE(c, 1)"),
          testing::StartsWith("    - Values: 4 rows"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  using namespace testing;

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          Eq("- PROJECT [2]: 10 fields: a BOOLEAN, b BOOLEAN, expr DOUBLE, expr_2 VARCHAR, expr_3 INTEGER, ..."),
          Eq("      expressions: field: 10"),
          Eq("  - PROJECT [1]: 10 fields: a_0 BOOLEAN, b_1 BOOLEAN, expr DOUBLE, expr_2 VARCHAR, expr_3 INTEGER, ..."),
          Eq("        expressions: AND: 1, CAST: 6, COALESCE: 1, DEREFERENCE: 2, IF: 1, OR: 1, SWITCH: 1, TRY: 1, TRY_CAST: 1, call: 10, constant: 11, field: 21"),
          Eq("        functions: divide: 2, eq: 2, gt: 4, lt: 1, multiply: 1"),
          Eq("        constants: BIGINT: 5, DOUBLE: 1, INTEGER: 1, VARCHAR: 4"),
          Eq("        projections: 8 out of 10"),
          Eq("        dereferences: 2 out of 10"),
          Eq("    - VALUES [0]: 3 fields: a INTEGER, b INTEGER, c ROW(2)"),
          Eq("          rows: 4"),
          Eq("")));

  lines = toSummaryLines(
      plan, {.project = {.maxProjections = 3, .maxDereferences = 1}});

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          Eq("- PROJECT [2]: 10 fields: a BOOLEAN, b BOOLEAN, expr DOUBLE, expr_2 VARCHAR, expr_3 INTEGER, ..."),
          Eq("      expressions: field: 10"),
          Eq("  - PROJECT [1]: 10 fields: a_0 BOOLEAN, b_1 BOOLEAN, expr DOUBLE, expr_2 VARCHAR, expr_3 INTEGER, ..."),
          Eq("        expressions: AND: 1, CAST: 6, COALESCE: 1, DEREFERENCE: 2, IF: 1, OR: 1, SWITCH: 1, TRY: 1, TRY_CAST: 1, call: 10, constant: 11, field: 21"),
          Eq("        functions: divide: 2, eq: 2, gt: 4, lt: 1, multiply: 1"),
          Eq("        constants: BIGINT: 5, DOUBLE: 1, INTEGER: 1, VARCHAR: 4"),
          Eq("        projections: 8 out of 10"),
          Eq("          a_0: AND(gt(a, b), gt(b, CAST(10 AS INTEGER)))"),
          Eq("          b_1: OR(lt(a, b), gt(b, CAST(0 AS INTEGER)))"),
          Eq("          expr: multiply(CAST(a AS DOUBLE), 1.2)"),
          Eq("          ... 5 more"),
          Eq("        dereferences: 2 out of 10"),
          Eq("          expr_4: DEREFERENCE(c, x)"),
          Eq("          ... 1 more"),
          Eq("    - VALUES [0]: 3 fields: a INTEGER, b INTEGER, c ROW(2)"),
          Eq("          rows: 4"),
          Eq("")));
}

TEST_F(PlanPrinterTest, lambda) {
  auto rowType = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  std::vector<Variant> data{
      Variant::row({1L, 10L}),
      Variant::row({2L, 20L}),
  };

  auto plan = PlanBuilder()
                  .values(rowType, data)
                  .map({"filter(sequence(1::bigint, a), x -> (x > b))"})
                  .build();

  auto lines = toLines(plan);

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          testing::StartsWith("- Project"),
          testing::StartsWith(
              "    expr := filter(sequence(CAST(1 AS BIGINT), a), x -> gt(x, b))"),
          testing::StartsWith("  - Values"),
          testing::Eq("")));

  lines = toSummaryLines(plan);

  using namespace testing;

  EXPECT_THAT(
      lines,
      testing::ElementsAre(
          Eq("- PROJECT [1]: 1 fields: expr ARRAY"),
          Eq("      expressions: CAST: 1, call: 3, constant: 1, field: 3, lambda: 1"),
          Eq("      functions: filter: 1, gt: 1, sequence: 1"),
          Eq("      constants: BIGINT: 1"),
          Eq("      projections: 1 out of 1"),
          Eq("  - VALUES [0]: 2 fields: a BIGINT, b BIGINT"),
          Eq("        rows: 2"),
          Eq("")));
}

} // namespace
} // namespace facebook::velox::logical_plan
