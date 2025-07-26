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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "axiom/logical_plan/NameAllocator.h"
#include "axiom/logical_plan/NameMappings.h"

namespace facebook::velox::logical_plan {

TEST(NameMappingsTest, basic) {
  NameAllocator allocator;

  auto newName = [&](const std::string& name) {
    return allocator.newName(name);
  };

  NameMappings mappings;

  auto reverseLookup = [&](const std::string& id) {
    auto names = mappings.reverseLookup(id);

    std::vector<std::string> strings;
    strings.reserve(names.size());
    for (auto& name : names) {
      strings.push_back(name.toString());
    }

    return strings;
  };

  auto makeNamesEq = [&](std::initializer_list<std::string> names) {
    return testing::UnorderedElementsAreArray(names);
  };

  {
    mappings.add("a", newName("a"), BOOLEAN());
    mappings.add("b", newName("b"), BOOLEAN());
    mappings.add("c", newName("c"), BOOLEAN());

    EXPECT_EQ(
        mappings.lookup("a"),
        (NameMappings::IdAndType{.id = "a", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("b"),
        (NameMappings::IdAndType{.id = "b", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("c"),
        (NameMappings::IdAndType{.id = "c", .type = BOOLEAN()}));

    mappings.setAlias("t");

    EXPECT_EQ(
        mappings.lookup("t", "a"),
        (NameMappings::IdAndType{.id = "a", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("t", "b"),
        (NameMappings::IdAndType{.id = "b", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("t", "c"),
        (NameMappings::IdAndType{.id = "c", .type = BOOLEAN()}));

    EXPECT_THAT(reverseLookup("a"), makeNamesEq({"a", "t.a"}));
    EXPECT_THAT(reverseLookup("b"), makeNamesEq({"b", "t.b"}));
    EXPECT_THAT(reverseLookup("c"), makeNamesEq({"c", "t.c"}));

    NameMappings other;
    other.add("a", newName("a"), BOOLEAN());
    other.add("c", newName("c"), BOOLEAN());
    other.add("e", newName("e"), BOOLEAN());

    mappings.merge(other);

    // "a" and "c" are no longer accessible w/o the alias. "a" from other is not
    // accessible at all.

    EXPECT_EQ(mappings.lookup("a"), std::nullopt);
    EXPECT_EQ(
        mappings.lookup("t", "a"),
        (NameMappings::IdAndType{.id = "a", .type = BOOLEAN()}));

    EXPECT_EQ(
        mappings.lookup("b"),
        (NameMappings::IdAndType{.id = "b", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("t", "b"),
        (NameMappings::IdAndType{.id = "b", .type = BOOLEAN()}));

    EXPECT_EQ(mappings.lookup("c"), std::nullopt);
    EXPECT_EQ(
        mappings.lookup("t", "c"),
        (NameMappings::IdAndType{.id = "c", .type = BOOLEAN()}));

    EXPECT_EQ(
        mappings.lookup("e"),
        (NameMappings::IdAndType{.id = "e", .type = BOOLEAN()}));

    EXPECT_THAT(reverseLookup("a"), makeNamesEq({"t.a"}));
    EXPECT_THAT(reverseLookup("b"), makeNamesEq({"b", "t.b"}));
    EXPECT_THAT(reverseLookup("c"), makeNamesEq({"t.c"}));
    EXPECT_THAT(reverseLookup("e"), makeNamesEq({"e"}));
  }

  {
    allocator.reset();
    mappings.reset();

    mappings.add("a", newName("a"), BOOLEAN());
    mappings.add("b", newName("b"), BOOLEAN());
    mappings.add("c", newName("c"), BOOLEAN());
    mappings.setAlias("t");

    NameMappings other;
    other.add("a", newName("a"), BOOLEAN());
    other.add("c", newName("c"), BOOLEAN());
    other.add("e", newName("e"), BOOLEAN());
    other.setAlias("u");
    mappings.merge(other);

    // "a" and "c" are no longer accessible w/o the alias.

    EXPECT_EQ(mappings.lookup("a"), std::nullopt);
    EXPECT_EQ(
        mappings.lookup("t", "a"),
        (NameMappings::IdAndType{.id = "a", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("u", "a"),
        (NameMappings::IdAndType{.id = "a_0", .type = BOOLEAN()}));

    EXPECT_EQ(
        mappings.lookup("b"),
        (NameMappings::IdAndType{.id = "b", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("t", "b"),
        (NameMappings::IdAndType{.id = "b", .type = BOOLEAN()}));
    EXPECT_EQ(mappings.lookup("u", "b"), std::nullopt);

    EXPECT_EQ(mappings.lookup("c"), std::nullopt);
    EXPECT_EQ(
        mappings.lookup("t", "c"),
        (NameMappings::IdAndType{.id = "c", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("u", "c"),
        (NameMappings::IdAndType{.id = "c_1", .type = BOOLEAN()}));

    EXPECT_EQ(
        mappings.lookup("e"),
        (NameMappings::IdAndType{.id = "e", .type = BOOLEAN()}));
    EXPECT_EQ(mappings.lookup("t", "e"), std::nullopt);
    EXPECT_EQ(
        mappings.lookup("u", "e"),
        (NameMappings::IdAndType{.id = "e", .type = BOOLEAN()}));

    EXPECT_THAT(reverseLookup("a"), makeNamesEq({"t.a"}));
    EXPECT_THAT(reverseLookup("b"), makeNamesEq({"b", "t.b"}));
    EXPECT_THAT(reverseLookup("c"), makeNamesEq({"t.c"}));

    EXPECT_THAT(reverseLookup("a_0"), makeNamesEq({"u.a"}));
    EXPECT_THAT(reverseLookup("c_1"), makeNamesEq({"u.c"}));
    EXPECT_THAT(reverseLookup("e"), makeNamesEq({"e", "u.e"}));

    mappings.setAlias("v");

    // Only b and e are still accessible.

    EXPECT_EQ(mappings.lookup("a"), std::nullopt);
    EXPECT_EQ(
        mappings.lookup("b"),
        (NameMappings::IdAndType{.id = "b", .type = BOOLEAN()}));
    EXPECT_EQ(
        mappings.lookup("v", "b"),
        (NameMappings::IdAndType{.id = "b", .type = BOOLEAN()}));
    EXPECT_EQ(mappings.lookup("c"), std::nullopt);
    EXPECT_EQ(
        mappings.lookup("v", "e"),
        (NameMappings::IdAndType{.id = "e", .type = BOOLEAN()}));

    EXPECT_THAT(reverseLookup("b"), makeNamesEq({"b", "v.b"}));
    EXPECT_THAT(reverseLookup("e"), makeNamesEq({"e", "v.e"}));
  }
}

} // namespace facebook::velox::logical_plan
