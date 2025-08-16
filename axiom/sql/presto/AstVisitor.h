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

namespace facebook::velox::sql {

// Forward declarations - organized by category
class Node;

// Literals
class BooleanLiteral;
class StringLiteral;
class BinaryLiteral;
class CharLiteral;
class LongLiteral;
class DoubleLiteral;
class DecimalLiteral;
class GenericLiteral;
class NullLiteral;
class TimeLiteral;
class TimestampLiteral;
class IntervalLiteral;
class EnumLiteral;

// Identifiers and References
class Identifier;
class QualifiedName;
class DereferenceExpression;
class FieldReference;
class SymbolReference;
class Parameter;

// Arithmetic and Comparison Expressions
class ArithmeticBinaryExpression;
class ArithmeticUnaryExpression;
class ComparisonExpression;
class BetweenPredicate;
class InPredicate;
class InListExpression;
class IsNullPredicate;
class IsNotNullPredicate;
class LikePredicate;
class ExistsPredicate;
class QuantifiedComparisonExpression;

// Logical Expressions
class LogicalBinaryExpression;
class NotExpression;

// Conditional Expressions
class IfExpression;
class CoalesceExpression;
class NullIfExpression;
class WhenClause;
class SearchedCaseExpression;
class SimpleCaseExpression;
class TryExpression;

// Function and Call Expressions
class FunctionCall;
class Cast;
class Extract;
class CurrentTime;
class CurrentUser;
class AtTimeZone;

// Complex Expressions
class SubqueryExpression;
class ArrayConstructor;
class Row;
class SubscriptExpression;
class LambdaExpression;
class LambdaArgumentDeclaration;
class BindExpression;
class GroupingOperation;
class TableVersionExpression;

// Query structures
class Query;
class QuerySpecification;
class Select;
class SingleColumn;
class AllColumns;
class With;
class WithQuery;
class OrderBy;
class SortItem;
class GroupBy;
class SimpleGroupBy;
class GroupingSets;
class Cube;
class Rollup;
class Offset;

// Relations
class Table;
class AliasedRelation;
class SampledRelation;
class TableSubquery;
class Lateral;
class Unnest;
class Values;

// Joins
class Join;
class JoinOn;
class JoinUsing;
class NaturalJoin;

// Set Operations
class Union;
class Intersect;
class Except;

// DDL Statements
class CreateTable;
class CreateTableAsSelect;
class CreateView;
class CreateMaterializedView;
class CreateSchema;
class CreateFunction;
class CreateRole;
class CreateType;
class DropTable;
class DropView;
class DropMaterializedView;
class DropSchema;

// DML Statements
class Insert;
class UpdateAssignment;
class Update;
class Delete;

// Utility Statements
class Explain;
class Analyze;
class Call;

// Transaction Statements
class StartTransaction;
class Commit;
class Rollback;

// Table Elements
class ColumnDefinition;
class LikeClause;
class ConstraintSpecification;

// Support Classes
class Property;
class CallArgument;
class Window;
class WindowFrame;
class FrameBound;
class PrincipalSpecification;
class GrantorSpecification;
class Isolation;
class TransactionAccessMode;
class SqlParameterDeclaration;
class RoutineCharacteristics;
class ExternalBodyReference;
class Return;
class ExplainFormat;
class ExplainType;

class AstVisitor {
public:
  virtual ~AstVisitor() = default;

  // Literals
  virtual void visitBooleanLiteral(BooleanLiteral* node) {}
  virtual void visitStringLiteral(StringLiteral* node) {}
  virtual void visitBinaryLiteral(BinaryLiteral* node) {}
  virtual void visitCharLiteral(CharLiteral* node) {}
  virtual void visitLongLiteral(LongLiteral* node) {}
  virtual void visitDoubleLiteral(DoubleLiteral* node) {}
  virtual void visitDecimalLiteral(DecimalLiteral* node) {}
  virtual void visitGenericLiteral(GenericLiteral* node) {}
  virtual void visitNullLiteral(NullLiteral* node) {}
  virtual void visitTimeLiteral(TimeLiteral* node) {}
  virtual void visitTimestampLiteral(TimestampLiteral* node) {}
  virtual void visitIntervalLiteral(IntervalLiteral* node) {}
  virtual void visitEnumLiteral(EnumLiteral* node) {}

  // Identifiers and References
  virtual void visitIdentifier(Identifier* node) {}
  virtual void visitQualifiedName(QualifiedName* node) {}
  virtual void visitDereferenceExpression(DereferenceExpression* node) {}
  virtual void visitFieldReference(FieldReference* node) {}
  virtual void visitSymbolReference(SymbolReference* node) {}
  virtual void visitParameter(Parameter* node) {}

  // Arithmetic and Comparison Expressions
  virtual void visitArithmeticBinaryExpression(ArithmeticBinaryExpression* node) {}
  virtual void visitArithmeticUnaryExpression(ArithmeticUnaryExpression* node) {}
  virtual void visitComparisonExpression(ComparisonExpression* node) {}
  virtual void visitBetweenPredicate(BetweenPredicate* node) {}
  virtual void visitInPredicate(InPredicate* node) {}
  virtual void visitInListExpression(InListExpression* node) {}
  virtual void visitIsNullPredicate(IsNullPredicate* node) {}
  virtual void visitIsNotNullPredicate(IsNotNullPredicate* node) {}
  virtual void visitLikePredicate(LikePredicate* node) {}
  virtual void visitExistsPredicate(ExistsPredicate* node) {}
  virtual void visitQuantifiedComparisonExpression(QuantifiedComparisonExpression* node) {}

  // Logical Expressions
  virtual void visitLogicalBinaryExpression(LogicalBinaryExpression* node) {}
  virtual void visitNotExpression(NotExpression* node) {}

  // Conditional Expressions
  virtual void visitIfExpression(IfExpression* node) {}
  virtual void visitCoalesceExpression(CoalesceExpression* node) {}
  virtual void visitNullIfExpression(NullIfExpression* node) {}
  virtual void visitWhenClause(WhenClause* node) {}
  virtual void visitSearchedCaseExpression(SearchedCaseExpression* node) {}
  virtual void visitSimpleCaseExpression(SimpleCaseExpression* node) {}
  virtual void visitTryExpression(TryExpression* node) {}

  // Function and Call Expressions
  virtual void visitFunctionCall(FunctionCall* node) {}
  virtual void visitCast(Cast* node) {}
  virtual void visitExtract(Extract* node) {}
  virtual void visitCurrentTime(CurrentTime* node) {}
  virtual void visitCurrentUser(CurrentUser* node) {}
  virtual void visitAtTimeZone(AtTimeZone* node) {}

  // Complex Expressions
  virtual void visitSubqueryExpression(SubqueryExpression* node) {}
  virtual void visitArrayConstructor(ArrayConstructor* node) {}
  virtual void visitRow(Row* node) {}
  virtual void visitSubscriptExpression(SubscriptExpression* node) {}
  virtual void visitLambdaExpression(LambdaExpression* node) {}
  virtual void visitLambdaArgumentDeclaration(LambdaArgumentDeclaration* node) {}
  virtual void visitBindExpression(BindExpression* node) {}
  virtual void visitGroupingOperation(GroupingOperation* node) {}
  virtual void visitTableVersionExpression(TableVersionExpression* node) {}

  // Query structures
  virtual void visitQuery(Query* node) {}
  virtual void visitQuerySpecification(QuerySpecification* node) {}
  virtual void visitSelect(Select* node) {}
  virtual void visitSingleColumn(SingleColumn* node) {}
  virtual void visitAllColumns(AllColumns* node) {}
  virtual void visitWith(With* node) {}
  virtual void visitWithQuery(WithQuery* node) {}
  virtual void visitOrderBy(OrderBy* node) {}
  virtual void visitSortItem(SortItem* node) {}
  virtual void visitGroupBy(GroupBy* node) {}
  virtual void visitSimpleGroupBy(SimpleGroupBy* node) {}
  virtual void visitGroupingSets(GroupingSets* node) {}
  virtual void visitCube(Cube* node) {}
  virtual void visitRollup(Rollup* node) {}
  virtual void visitOffset(Offset* node) {}

  // Relations
  virtual void visitTable(Table* node) {}
  virtual void visitAliasedRelation(AliasedRelation* node) {}
  virtual void visitSampledRelation(SampledRelation* node) {}
  virtual void visitTableSubquery(TableSubquery* node) {}
  virtual void visitLateral(Lateral* node) {}
  virtual void visitUnnest(Unnest* node) {}
  virtual void visitValues(Values* node) {}

  // Joins
  virtual void visitJoin(Join* node) {}
  virtual void visitJoinOn(JoinOn* node) {}
  virtual void visitJoinUsing(JoinUsing* node) {}
  virtual void visitNaturalJoin(NaturalJoin* node) {}

  // Set Operations
  virtual void visitUnion(Union* node) {}
  virtual void visitIntersect(Intersect* node) {}
  virtual void visitExcept(Except* node) {}

  // DDL Statements
  virtual void visitCreateTable(CreateTable* node) {}
  virtual void visitCreateTableAsSelect(CreateTableAsSelect* node) {}
  virtual void visitCreateView(CreateView* node) {}
  virtual void visitCreateMaterializedView(CreateMaterializedView* node) {}
  virtual void visitCreateSchema(CreateSchema* node) {}
  virtual void visitCreateFunction(CreateFunction* node) {}
  virtual void visitCreateRole(CreateRole* node) {}
  virtual void visitCreateType(CreateType* node) {}
  virtual void visitDropTable(DropTable* node) {}
  virtual void visitDropView(DropView* node) {}
  virtual void visitDropMaterializedView(DropMaterializedView* node) {}
  virtual void visitDropSchema(DropSchema* node) {}

  // DML Statements
  virtual void visitInsert(Insert* node) {}
  virtual void visitUpdateAssignment(UpdateAssignment* node) {}
  virtual void visitUpdate(Update* node) {}
  virtual void visitDelete(Delete* node) {}

  // Utility Statements
  virtual void visitExplain(Explain* node) {}
  virtual void visitAnalyze(Analyze* node) {}
  virtual void visitCall(Call* node) {}

  // Transaction Statements
  virtual void visitStartTransaction(StartTransaction* node) {}
  virtual void visitCommit(Commit* node) {}
  virtual void visitRollback(Rollback* node) {}

  // Table Elements
  virtual void visitColumnDefinition(ColumnDefinition* node) {}
  virtual void visitLikeClause(LikeClause* node) {}
  virtual void visitConstraintSpecification(ConstraintSpecification* node) {}

  // Support Classes
  virtual void visitProperty(Property* node) {}
  virtual void visitCallArgument(CallArgument* node) {}
  virtual void visitWindow(Window* node) {}
  virtual void visitWindowFrame(WindowFrame* node) {}
  virtual void visitFrameBound(FrameBound* node) {}
  virtual void visitPrincipalSpecification(PrincipalSpecification* node) {}
  virtual void visitGrantorSpecification(GrantorSpecification* node) {}
  virtual void visitIsolation(Isolation* node) {}
  virtual void visitTransactionAccessMode(TransactionAccessMode* node) {}
  virtual void visitSqlParameterDeclaration(SqlParameterDeclaration* node) {}
  virtual void visitRoutineCharacteristics(RoutineCharacteristics* node) {}
  virtual void visitExternalBodyReference(ExternalBodyReference* node) {}
  virtual void visitReturn(Return* node) {}
  virtual void visitExplainFormat(ExplainFormat* node) {}
  virtual void visitExplainType(ExplainType* node) {}

  // Generic visit method
  virtual void visit(Node* node) {}
};

} // namespace facebook::velox::sql