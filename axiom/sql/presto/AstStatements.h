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

#include "AstNode.h"
#include "AstVisitor.h"
#include <optional>
#include <vector>

namespace facebook::velox::sql {

// Forward declarations
class QualifiedName;
class TableElement;
class Property;
class ColumnDefinition;
class Expression;
class Relation;
class PrincipalSpecification;
class GrantorSpecification;
class RoutineCharacteristics;
class ExplainOption;
class CallArgument;
class UpdateAssignment;
class RoutineBody;
class SqlParameterDeclaration;
class TransactionMode;

// DDL Statements
class CreateTable : public Statement {
public:
  CreateTable(
      std::shared_ptr<QualifiedName> name,
      const std::vector<std::shared_ptr<TableElement>>& elements,
      bool notExists,
      const std::vector<std::shared_ptr<Property>>& properties,
      std::optional<std::string> comment = std::nullopt)
      : Statement(NodeType::kCreateTable),
        name_(name), elements_(elements), notExists_(notExists), properties_(properties), comment_(comment) {}
  
  const std::shared_ptr<QualifiedName>& getName() const { return name_; }
  const std::vector<std::shared_ptr<TableElement>>& getElements() const { return elements_; }
  bool isNotExists() const { return notExists_; }
  const std::vector<std::shared_ptr<Property>>& getProperties() const { return properties_; }
  const std::optional<std::string>& getComment() const { return comment_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> name_;
  std::vector<std::shared_ptr<TableElement>> elements_;
  bool notExists_;
  std::vector<std::shared_ptr<Property>> properties_;
  std::optional<std::string> comment_;
};

class CreateTableAsSelect : public Statement {
public:
  CreateTableAsSelect(
      std::shared_ptr<QualifiedName> name,
      StatementPtr query,
      bool notExists,
      const std::vector<std::shared_ptr<Property>>& properties,
      bool withData,
      std::optional<std::string> comment = std::nullopt)
      : Statement(NodeType::kCreateTableAsSelect),
        name_(name), query_(query), notExists_(notExists), properties_(properties), withData_(withData), comment_(comment) {}
  
  const std::shared_ptr<QualifiedName>& getName() const { return name_; }
  const StatementPtr& getQuery() const { return query_; }
  bool isNotExists() const { return notExists_; }
  const std::vector<std::shared_ptr<Property>>& getProperties() const { return properties_; }
  bool isWithData() const { return withData_; }
  const std::optional<std::string>& getComment() const { return comment_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> name_;
  StatementPtr query_;
  bool notExists_;
  std::vector<std::shared_ptr<Property>> properties_;
  bool withData_;
  std::optional<std::string> comment_;
};

class CreateView : public Statement {
public:
  enum class Security { INVOKER, DEFINER };
  
  CreateView(
      std::shared_ptr<QualifiedName> name,
      StatementPtr query,
      bool replace,
      std::optional<Security> security = std::nullopt,
      std::optional<std::string> comment = std::nullopt)
      : Statement(NodeType::kCreateView),
        name_(name), query_(query), replace_(replace), security_(security), comment_(comment) {}
  
  const std::shared_ptr<QualifiedName>& getName() const { return name_; }
  const StatementPtr& getQuery() const { return query_; }
  bool isReplace() const { return replace_; }
  const std::optional<Security>& getSecurity() const { return security_; }
  const std::optional<std::string>& getComment() const { return comment_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> name_;
  StatementPtr query_;
  bool replace_;
  std::optional<Security> security_;
  std::optional<std::string> comment_;
};

class CreateMaterializedView : public Statement {
public:
  CreateMaterializedView(
      std::shared_ptr<QualifiedName> name,
      StatementPtr query,
      bool notExists,
      const std::vector<std::shared_ptr<Property>>& properties,
      std::optional<std::string> comment = std::nullopt)
      : Statement(NodeType::kCreateMaterializedView),
        name_(name), query_(query), notExists_(notExists), properties_(properties), comment_(comment) {}
  
  const std::shared_ptr<QualifiedName>& getName() const { return name_; }
  const StatementPtr& getQuery() const { return query_; }
  bool isNotExists() const { return notExists_; }
  const std::vector<std::shared_ptr<Property>>& getProperties() const { return properties_; }
  const std::optional<std::string>& getComment() const { return comment_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> name_;
  StatementPtr query_;
  bool notExists_;
  std::vector<std::shared_ptr<Property>> properties_;
  std::optional<std::string> comment_;
};

class CreateSchema : public Statement {
public:
  CreateSchema(
      std::shared_ptr<QualifiedName> schemaName,
      bool notExists,
      const std::vector<std::shared_ptr<Property>>& properties)
      : Statement(NodeType::kCreateSchema),
        schemaName_(schemaName), notExists_(notExists), properties_(properties) {}
  
  const std::shared_ptr<QualifiedName>& getSchemaName() const { return schemaName_; }
  bool isNotExists() const { return notExists_; }
  const std::vector<std::shared_ptr<Property>>& getProperties() const { return properties_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> schemaName_;
  bool notExists_;
  std::vector<std::shared_ptr<Property>> properties_;
};

class CreateFunction : public Statement {
public:
  CreateFunction(
      std::shared_ptr<QualifiedName> functionName,
      bool replace,
      const std::vector<std::shared_ptr<SqlParameterDeclaration>>& parameters,
      const std::string& returnType,
      std::shared_ptr<RoutineCharacteristics> characteristics,
      std::shared_ptr<RoutineBody> body)
      : Statement(NodeType::kCreateFunction),
        functionName_(functionName), replace_(replace), parameters_(parameters), 
        returnType_(returnType), characteristics_(characteristics), body_(body) {}
  
  const std::shared_ptr<QualifiedName>& getFunctionName() const { return functionName_; }
  bool isReplace() const { return replace_; }
  const std::vector<std::shared_ptr<SqlParameterDeclaration>>& getParameters() const { return parameters_; }
  const std::string& getReturnType() const { return returnType_; }
  const std::shared_ptr<RoutineCharacteristics>& getCharacteristics() const { return characteristics_; }
  const std::shared_ptr<RoutineBody>& getBody() const { return body_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> functionName_;
  bool replace_;
  std::vector<std::shared_ptr<SqlParameterDeclaration>> parameters_;
  std::string returnType_;
  std::shared_ptr<RoutineCharacteristics> characteristics_;
  std::shared_ptr<RoutineBody> body_;
};

class CreateRole : public Statement {
public:
  CreateRole(const std::string& name, std::optional<std::shared_ptr<PrincipalSpecification>> grantor = std::nullopt)
      : Statement(NodeType::kCreateRole), name_(name), grantor_(grantor) {}
  
  const std::string& getName() const { return name_; }
  const std::optional<std::shared_ptr<PrincipalSpecification>>& getGrantor() const { return grantor_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::string name_;
  std::optional<std::shared_ptr<PrincipalSpecification>> grantor_;
};

class CreateType : public Statement {
public:
  CreateType(
      std::shared_ptr<QualifiedName> name,
      const std::vector<std::string>& enumValues)
      : Statement(NodeType::kCreateType), name_(name), enumValues_(enumValues) {}
  
  const std::shared_ptr<QualifiedName>& getName() const { return name_; }
  const std::vector<std::string>& getEnumValues() const { return enumValues_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> name_;
  std::vector<std::string> enumValues_;
};

// Drop Statements
class DropTable : public Statement {
public:
  DropTable(std::shared_ptr<QualifiedName> tableName, bool exists)
      : Statement(NodeType::kDropTable), tableName_(tableName), exists_(exists) {}
  
  const std::shared_ptr<QualifiedName>& getTableName() const { return tableName_; }
  bool isExists() const { return exists_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> tableName_;
  bool exists_;
};

class DropView : public Statement {
public:
  DropView(std::shared_ptr<QualifiedName> viewName, bool exists)
      : Statement(NodeType::kDropView), viewName_(viewName), exists_(exists) {}
  
  const std::shared_ptr<QualifiedName>& getViewName() const { return viewName_; }
  bool isExists() const { return exists_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> viewName_;
  bool exists_;
};

class DropMaterializedView : public Statement {
public:
  DropMaterializedView(std::shared_ptr<QualifiedName> viewName, bool exists)
      : Statement(NodeType::kDropMaterializedView), viewName_(viewName), exists_(exists) {}
  
  const std::shared_ptr<QualifiedName>& getViewName() const { return viewName_; }
  bool isExists() const { return exists_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> viewName_;
  bool exists_;
};

class DropSchema : public Statement {
public:
  enum class DropBehavior { RESTRICT, CASCADE };
  
  DropSchema(std::shared_ptr<QualifiedName> schemaName, bool exists, DropBehavior behavior)
      : Statement(NodeType::kDropSchema), schemaName_(schemaName), exists_(exists), behavior_(behavior) {}
  
  const std::shared_ptr<QualifiedName>& getSchemaName() const { return schemaName_; }
  bool isExists() const { return exists_; }
  DropBehavior getBehavior() const { return behavior_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> schemaName_;
  bool exists_;
  DropBehavior behavior_;
};

// DML Statements
class Insert : public Statement {
public:
  Insert(
      std::shared_ptr<QualifiedName> target,
      std::optional<std::vector<std::shared_ptr<Identifier>>> columns,
      StatementPtr query)
      : Statement(NodeType::kInsert), target_(target), columns_(columns), query_(query) {}
  
  const std::shared_ptr<QualifiedName>& getTarget() const { return target_; }
  const std::optional<std::vector<std::shared_ptr<Identifier>>>& getColumns() const { return columns_; }
  const StatementPtr& getQuery() const { return query_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> target_;
  std::optional<std::vector<std::shared_ptr<Identifier>>> columns_;
  StatementPtr query_;
};

class UpdateAssignment : public Node {
public:
  UpdateAssignment(std::shared_ptr<Identifier> name, ExpressionPtr value)
      : Node(NodeType::kUpdateAssignment), name_(name), value_(value) {}
  
  const std::shared_ptr<Identifier>& getName() const { return name_; }
  const ExpressionPtr& getValue() const { return value_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<Identifier> name_;
  ExpressionPtr value_;
};

class Update : public Statement {
public:
  Update(
      std::shared_ptr<QualifiedName> table,
      const std::vector<std::shared_ptr<UpdateAssignment>>& assignments,
      std::optional<ExpressionPtr> where = std::nullopt)
      : Statement(NodeType::kUpdate), table_(table), assignments_(assignments), where_(where) {}
  
  const std::shared_ptr<QualifiedName>& getTable() const { return table_; }
  const std::vector<std::shared_ptr<UpdateAssignment>>& getAssignments() const { return assignments_; }
  const std::optional<ExpressionPtr>& getWhere() const { return where_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> table_;
  std::vector<std::shared_ptr<UpdateAssignment>> assignments_;
  std::optional<ExpressionPtr> where_;
};

class Delete : public Statement {
public:
  Delete(std::shared_ptr<QualifiedName> table, std::optional<ExpressionPtr> where = std::nullopt)
      : Statement(NodeType::kDelete), table_(table), where_(where) {}
  
  const std::shared_ptr<QualifiedName>& getTable() const { return table_; }
  const std::optional<ExpressionPtr>& getWhere() const { return where_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> table_;
  std::optional<ExpressionPtr> where_;
};

// Utility Statements
class Explain : public Statement {
public:
  Explain(StatementPtr statement, const std::vector<std::shared_ptr<ExplainOption>>& options)
      : Statement(NodeType::kExplain), statement_(statement), options_(options) {}
  
  const StatementPtr& getStatement() const { return statement_; }
  const std::vector<std::shared_ptr<ExplainOption>>& getOptions() const { return options_; }
  
  void accept(AstVisitor* visitor) override;

private:
  StatementPtr statement_;
  std::vector<std::shared_ptr<ExplainOption>> options_;
};

class Analyze : public Statement {
public:
  Analyze(std::shared_ptr<QualifiedName> tableName, const std::vector<std::shared_ptr<Property>>& properties)
      : Statement(NodeType::kAnalyze), tableName_(tableName), properties_(properties) {}
  
  const std::shared_ptr<QualifiedName>& getTableName() const { return tableName_; }
  const std::vector<std::shared_ptr<Property>>& getProperties() const { return properties_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> tableName_;
  std::vector<std::shared_ptr<Property>> properties_;
};

class Call : public Statement {
public:
  Call(std::shared_ptr<QualifiedName> name, const std::vector<std::shared_ptr<CallArgument>>& arguments)
      : Statement(NodeType::kCall), name_(name), arguments_(arguments) {}
  
  const std::shared_ptr<QualifiedName>& getName() const { return name_; }
  const std::vector<std::shared_ptr<CallArgument>>& getArguments() const { return arguments_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::shared_ptr<QualifiedName> name_;
  std::vector<std::shared_ptr<CallArgument>> arguments_;
};

// Transaction Statements
class StartTransaction : public Statement {
public:
  explicit StartTransaction(const std::vector<std::shared_ptr<TransactionMode>>& transactionModes)
      : Statement(NodeType::kStartTransaction), transactionModes_(transactionModes) {}
  
  const std::vector<std::shared_ptr<TransactionMode>>& getTransactionModes() const { return transactionModes_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::vector<std::shared_ptr<TransactionMode>> transactionModes_;
};

class Commit : public Statement {
public:
  Commit() : Statement(NodeType::kCommit) {}
  void accept(AstVisitor* visitor) override;
};

class Rollback : public Statement {
public:
  Rollback() : Statement(NodeType::kRollback) {}
  void accept(AstVisitor* visitor) override;
};

} // namespace facebook::velox::sql