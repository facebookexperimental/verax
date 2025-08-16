/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#pragma once

#include "AstNodesAll.h"
#include "AstBuilder.h"
#include <string>
#include <memory>

namespace facebook::velox::sql {

struct ParseResult {
  NodePtr ast;
  std::vector<std::string> warnings;
  bool hasErrors;
  std::string errorMessage;
};

class PrestoSqlAstParser {
public:
  explicit PrestoSqlAstParser(const ParsingOptions& options = ParsingOptions{})
      : options_(options) {}

  /// Parse a SQL statement and return the AST
  ParseResult parseStatement(const std::string& sql);

  /// Parse a SQL expression and return the AST
  ParseResult parseExpression(const std::string& sql);

  /// Parse a SQL query specifically
  std::shared_ptr<Query> parseQuery(const std::string& sql);

  /// Set parsing options
  void setOptions(const ParsingOptions& options) {
    options_ = options;
  }

  const ParsingOptions& getOptions() const {
    return options_;
  }

private:
  ParsingOptions options_;
  
  // Internal helper method to handle common parsing logic
  ParseResult doParse(
      const std::string& sql,
      std::function<antlrcpp::Any(AstBuilder&, antlr4::tree::ParseTree*)> parseFunction);
};

} // namespace facebook::velox::sql