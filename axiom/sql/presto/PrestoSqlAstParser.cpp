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

#include "PrestoSqlAstParser.h"
#include "PrestoSqlLexer.h"
#include "PrestoSqlParser.h"
#include "tests/UpperCasedInput.h"
#include "antlr4-runtime.h"
#include <sstream>

namespace facebook::velox::sql {

class ErrorListener : public antlr4::BaseErrorListener {
public:
  void syntaxError(
      antlr4::Recognizer* recognizer,
      antlr4::Token* offendingSymbol,
      size_t line,
      size_t charPositionInLine,
      const std::string& msg,
      std::exception_ptr e) override {
    std::ostringstream oss;
    oss << "line " << line << ":" << charPositionInLine << " " << msg;
    errors_.push_back(oss.str());
  }

  const std::vector<std::string>& getErrors() const {
    return errors_;
  }

  void clear() {
    errors_.clear();
  }

private:
  std::vector<std::string> errors_;
};

ParseResult PrestoSqlAstParser::parseStatement(const std::string& sql) {
  return doParse(sql, [](AstBuilder& builder, antlr4::tree::ParseTree* tree) {
    auto ctx = dynamic_cast<PrestoSqlParser::SingleStatementContext*>(tree);
    return builder.visitSingleStatement(ctx);
  });
}

ParseResult PrestoSqlAstParser::parseExpression(const std::string& sql) {
  return doParse(sql, [](AstBuilder& builder, antlr4::tree::ParseTree* tree) {
    auto ctx = dynamic_cast<PrestoSqlParser::StandaloneExpressionContext*>(tree);
    return builder.visitStandaloneExpression(ctx);
  });
}

std::shared_ptr<Query> PrestoSqlAstParser::parseQuery(const std::string& sql) {
  auto result = parseStatement(sql);
  if (result.hasErrors || !result.ast) {
    return nullptr;
  }
  
  return result.ast->as<Query>();
}

ParseResult PrestoSqlAstParser::doParse(
    const std::string& sql,
    std::function<antlrcpp::Any(AstBuilder&, antlr4::tree::ParseTree*)> parseFunction) {
  
  ParseResult result;
  result.hasErrors = false;
  
  try {
    // Create ANTLR input stream with case-insensitive wrapper
    UpperCasedInput input(sql);
    
    // Create lexer
    PrestoSqlLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    
    // Create parser
    PrestoSqlParser parser(&tokens);
    
    // Set up error handling
    ErrorListener errorListener;
    parser.removeErrorListeners();
    parser.addErrorListener(&errorListener);
    
    lexer.removeErrorListeners();
    lexer.addErrorListener(&errorListener);
    
    // Parse based on the provided function
    antlr4::tree::ParseTree* tree;
    if (sql.find("SELECT") != std::string::npos || sql.find("select") != std::string::npos) {
      tree = parser.singleStatement();
    } else {
      // Try to parse as expression first
      try {
        tree = parser.standaloneExpression();
      } catch (...) {
        // Fall back to statement parsing
        tree = parser.singleStatement();
      }
    }
    
    // Check for parsing errors
    const auto& errors = errorListener.getErrors();
    if (!errors.empty()) {
      result.hasErrors = true;
      result.errorMessage = errors[0]; // Report first error
      return result;
    }
    
    // Build AST using visitor
    std::vector<std::string> warnings;
    ParsingOptions optionsWithWarnings = options_;
    optionsWithWarnings.warningConsumer = [&warnings](const std::string& warning) {
      warnings.push_back(warning);
    };
    
    AstBuilder builder(optionsWithWarnings);
    
    try {
      auto anyResult = parseFunction(builder, tree);
      result.ast = std::any_cast<NodePtr>(anyResult);
      result.warnings = warnings;
    } catch (const std::bad_any_cast& e) {
      result.hasErrors = true;
      result.errorMessage = "AST building failed: bad any_cast - " + std::string(e.what());
    } catch (const std::exception& e) {
      result.hasErrors = true;
      result.errorMessage = "AST building failed: " + std::string(e.what());
    }
    
  } catch (const std::exception& e) {
    result.hasErrors = true;
    result.errorMessage = "Parsing failed: " + std::string(e.what());
  }
  
  return result;
}

} // namespace facebook::velox::sql