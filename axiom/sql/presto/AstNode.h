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

#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace facebook::velox::sql {

class AstVisitor;

enum class NodeType {
  // Base types
  kNode,
  kStatement,
  kExpression,
  kRelation,
  kQueryBody,
  
  // Literals
  kBooleanLiteral,
  kStringLiteral,
  kLongLiteral,
  kDoubleLiteral,
  kDecimalLiteral,
  kGenericLiteral,
  kNullLiteral,
  kBinaryLiteral,
  kCharLiteral,
  kTimeLiteral,
  kTimestampLiteral,
  kIntervalLiteral,
  kEnumLiteral,
  
  // Identifiers and References
  kIdentifier,
  kQualifiedName,
  kDereferenceExpression,
  kFieldReference,
  kSymbolReference,
  kParameter,
  
  // Arithmetic Expressions
  kArithmeticBinaryExpression,
  kArithmeticUnaryExpression,
  
  // Comparison Expressions
  kComparisonExpression,
  kBetweenPredicate,
  kInPredicate,
  kInListExpression,
  kIsNullPredicate,
  kIsNotNullPredicate,
  kLikePredicate,
  kExistsPredicate,
  kQuantifiedComparisonExpression,
  
  // Logical Expressions
  kLogicalBinaryExpression,
  kNotExpression,
  
  // Conditional Expressions
  kIfExpression,
  kCoalesceExpression,
  kNullIfExpression,
  kSearchedCaseExpression,
  kSimpleCaseExpression,
  kWhenClause,
  kTryExpression,
  
  // Function and Call Expressions
  kFunctionCall,
  kCast,
  kExtract,
  kCurrentTime,
  kCurrentUser,
  kAtTimeZone,
  
  // Complex Expressions
  kSubqueryExpression,
  kArrayConstructor,
  kRow,
  kSubscriptExpression,
  kLambdaExpression,
  kLambdaArgumentDeclaration,
  kBindExpression,
  kGroupingOperation,
  kTableVersionExpression,
  
  // Query Structures
  kQuery,
  kQuerySpecification,
  kSelect,
  kSelectItem,
  kSingleColumn,
  kAllColumns,
  kWith,
  kWithQuery,
  kOrderBy,
  kSortItem,
  kGroupBy,
  kGroupingSets,
  kCube,
  kRollup,
  kSimpleGroupBy,
  kOffset,
  kHaving,
  kLimit,
  kWhere,
  
  // Relations
  kTable,
  kAliasedRelation,
  kSampledRelation,
  kTableSubquery,
  kLateral,
  kUnnest,
  kValues,
  
  // Joins
  kJoin,
  kJoinOn,
  kJoinUsing,
  kNaturalJoin,
  
  // Set Operations
  kUnion,
  kIntersect,
  kExcept,
  
  // DDL Statements
  kCreateTable,
  kCreateTableAsSelect,
  kCreateView,
  kCreateMaterializedView,
  kCreateSchema,
  kCreateFunction,
  kCreateRole,
  kCreateType,
  kDropTable,
  kDropView,
  kDropMaterializedView,
  kDropSchema,
  kDropFunction,
  kDropRole,
  kAlterFunction,
  kAddColumn,
  kDropColumn,
  kRenameColumn,
  kRenameTable,
  kRenameSchema,
  kAddConstraint,
  kDropConstraint,
  kTruncateTable,
  
  // DML Statements
  kInsert,
  kUpdate,
  kDelete,
  kUpdateAssignment,
  
  // Show Statements
  kShowTables,
  kShowColumns,
  kShowSchemas,
  kShowCatalogs,
  kShowFunctions,
  kShowCreate,
  kShowCreateFunction,
  kShowSession,
  kShowStats,
  kShowGrants,
  kShowRoles,
  kShowRoleGrants,
  
  // Transaction Statements
  kStartTransaction,
  kCommit,
  kRollback,
  kSetSession,
  kResetSession,
  kSetRole,
  
  // Security Statements
  kGrant,
  kRevoke,
  kGrantRoles,
  kRevokeRoles,
  
  // Utility Statements
  kExplain,
  kAnalyze,
  kCall,
  kPrepare,
  kExecute,
  kDeallocate,
  kUse,
  kDescribeInput,
  kDescribeOutput,
  kRefreshMaterializedView,
  kAlterRoutineCharacteristics,
  
  // Table Elements
  kColumnDefinition,
  kLikeClause,
  kConstraintSpecification,
  
  // Support Classes
  kProperty,
  kCallArgument,
  kWindow,
  kWindowFrame,
  kFrameBound,
  kIsolation,
  kTransactionAccessMode,
  kPrincipalSpecification,
  kGrantorSpecification,
  kRoutineCharacteristics,
  kExternalBodyReference,
  kReturn,
  kSqlParameterDeclaration,
  
  // Explain Options
  kExplainFormat,
  kExplainType,
  kExplainOption
};

class Node {
public:
  explicit Node(NodeType type) : type_(type) {}
  virtual ~Node() = default;

  NodeType getType() const { return type_; }
  
  virtual void accept(AstVisitor* visitor) = 0;
  
  template<typename T>
  T* as() {
    return dynamic_cast<T*>(this);
  }
  
  template<typename T>
  const T* as() const {
    return dynamic_cast<const T*>(this);
  }

private:
  NodeType type_;
};

using NodePtr = std::shared_ptr<Node>;

class Expression : public Node {
public:
  explicit Expression(NodeType type) : Node(type) {}
};

using ExpressionPtr = std::shared_ptr<Expression>;

class Statement : public Node {
public:
  explicit Statement(NodeType type) : Node(type) {}
};

using StatementPtr = std::shared_ptr<Statement>;

class Relation : public Node {
public:
  explicit Relation(NodeType type) : Node(type) {}
};

using RelationPtr = std::shared_ptr<Relation>;

// Forward declarations
class Identifier;
class QualifiedName;

} // namespace facebook::velox::sql