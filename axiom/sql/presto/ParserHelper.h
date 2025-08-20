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

#include "axiom/sql/presto/ast/UpperCaseInputStream.h"
#include "axiom/sql/presto/grammar/PrestoSqlLexer.h"
#include "axiom/sql/presto/grammar/PrestoSqlParser.h"

namespace axiom::sql::presto {

/// Creates Presto SQL parser.
class ParserHelper {
 public:
  explicit ParserHelper(const std::string& sql)
      : inputStream_(std::make_unique<UpperCaseInputStream>(sql)),
        lexer_(std::make_unique<PrestoSqlLexer>(inputStream_.get())),
        tokenStream_(std::make_unique<antlr4::CommonTokenStream>(lexer_.get())),
        parser_(std::make_unique<PrestoSqlParser>(tokenStream_.get())) {}

  PrestoSqlParser& parser() const {
    return *parser_;
  }

 private:
  std::unique_ptr<antlr4::ANTLRInputStream> inputStream_;
  std::unique_ptr<PrestoSqlLexer> lexer_;
  std::unique_ptr<antlr4::CommonTokenStream> tokenStream_;
  std::unique_ptr<PrestoSqlParser> parser_;
};
} // namespace axiom::sql::presto
