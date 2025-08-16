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

#include "AstNodesAll.h"

namespace facebook::velox::sql {

// Literal implementations
void BooleanLiteral::accept(AstVisitor* visitor) { visitor->visitBooleanLiteral(this); }
void StringLiteral::accept(AstVisitor* visitor) { visitor->visitStringLiteral(this); }
void BinaryLiteral::accept(AstVisitor* visitor) { visitor->visitBinaryLiteral(this); }
void CharLiteral::accept(AstVisitor* visitor) { visitor->visitCharLiteral(this); }
void LongLiteral::accept(AstVisitor* visitor) { visitor->visitLongLiteral(this); }
void DoubleLiteral::accept(AstVisitor* visitor) { visitor->visitDoubleLiteral(this); }
void DecimalLiteral::accept(AstVisitor* visitor) { visitor->visitDecimalLiteral(this); }
void GenericLiteral::accept(AstVisitor* visitor) { visitor->visitGenericLiteral(this); }
void NullLiteral::accept(AstVisitor* visitor) { visitor->visitNullLiteral(this); }
void TimeLiteral::accept(AstVisitor* visitor) { visitor->visitTimeLiteral(this); }
void TimestampLiteral::accept(AstVisitor* visitor) { visitor->visitTimestampLiteral(this); }
void IntervalLiteral::accept(AstVisitor* visitor) { visitor->visitIntervalLiteral(this); }
void EnumLiteral::accept(AstVisitor* visitor) { visitor->visitEnumLiteral(this); }

// Identifier and Reference implementations
void Identifier::accept(AstVisitor* visitor) { visitor->visitIdentifier(this); }
void QualifiedName::accept(AstVisitor* visitor) { visitor->visitQualifiedName(this); }
void DereferenceExpression::accept(AstVisitor* visitor) { visitor->visitDereferenceExpression(this); }
void FieldReference::accept(AstVisitor* visitor) { visitor->visitFieldReference(this); }
void SymbolReference::accept(AstVisitor* visitor) { visitor->visitSymbolReference(this); }
void Parameter::accept(AstVisitor* visitor) { visitor->visitParameter(this); }

// Arithmetic and Comparison Expression implementations
void ArithmeticBinaryExpression::accept(AstVisitor* visitor) { visitor->visitArithmeticBinaryExpression(this); }
void ArithmeticUnaryExpression::accept(AstVisitor* visitor) { visitor->visitArithmeticUnaryExpression(this); }
void ComparisonExpression::accept(AstVisitor* visitor) { visitor->visitComparisonExpression(this); }
void BetweenPredicate::accept(AstVisitor* visitor) { visitor->visitBetweenPredicate(this); }
void InPredicate::accept(AstVisitor* visitor) { visitor->visitInPredicate(this); }
void InListExpression::accept(AstVisitor* visitor) { visitor->visitInListExpression(this); }
void IsNullPredicate::accept(AstVisitor* visitor) { visitor->visitIsNullPredicate(this); }
void IsNotNullPredicate::accept(AstVisitor* visitor) { visitor->visitIsNotNullPredicate(this); }
void LikePredicate::accept(AstVisitor* visitor) { visitor->visitLikePredicate(this); }
void ExistsPredicate::accept(AstVisitor* visitor) { visitor->visitExistsPredicate(this); }
void QuantifiedComparisonExpression::accept(AstVisitor* visitor) { visitor->visitQuantifiedComparisonExpression(this); }

// Logical Expression implementations
void LogicalBinaryExpression::accept(AstVisitor* visitor) { visitor->visitLogicalBinaryExpression(this); }
void NotExpression::accept(AstVisitor* visitor) { visitor->visitNotExpression(this); }

// Conditional Expression implementations
void IfExpression::accept(AstVisitor* visitor) { visitor->visitIfExpression(this); }
void CoalesceExpression::accept(AstVisitor* visitor) { visitor->visitCoalesceExpression(this); }
void NullIfExpression::accept(AstVisitor* visitor) { visitor->visitNullIfExpression(this); }
void WhenClause::accept(AstVisitor* visitor) { visitor->visitWhenClause(this); }
void SearchedCaseExpression::accept(AstVisitor* visitor) { visitor->visitSearchedCaseExpression(this); }
void SimpleCaseExpression::accept(AstVisitor* visitor) { visitor->visitSimpleCaseExpression(this); }
void TryExpression::accept(AstVisitor* visitor) { visitor->visitTryExpression(this); }

// Function and Call Expression implementations
void FunctionCall::accept(AstVisitor* visitor) { visitor->visitFunctionCall(this); }
void Cast::accept(AstVisitor* visitor) { visitor->visitCast(this); }
void Extract::accept(AstVisitor* visitor) { visitor->visitExtract(this); }
void CurrentTime::accept(AstVisitor* visitor) { visitor->visitCurrentTime(this); }
void CurrentUser::accept(AstVisitor* visitor) { visitor->visitCurrentUser(this); }
void AtTimeZone::accept(AstVisitor* visitor) { visitor->visitAtTimeZone(this); }

// Complex Expression implementations
void SubqueryExpression::accept(AstVisitor* visitor) { visitor->visitSubqueryExpression(this); }
void ArrayConstructor::accept(AstVisitor* visitor) { visitor->visitArrayConstructor(this); }
void Row::accept(AstVisitor* visitor) { visitor->visitRow(this); }
void SubscriptExpression::accept(AstVisitor* visitor) { visitor->visitSubscriptExpression(this); }
void LambdaExpression::accept(AstVisitor* visitor) { visitor->visitLambdaExpression(this); }
void LambdaArgumentDeclaration::accept(AstVisitor* visitor) { visitor->visitLambdaArgumentDeclaration(this); }
void BindExpression::accept(AstVisitor* visitor) { visitor->visitBindExpression(this); }
void GroupingOperation::accept(AstVisitor* visitor) { visitor->visitGroupingOperation(this); }
void TableVersionExpression::accept(AstVisitor* visitor) { visitor->visitTableVersionExpression(this); }

// Query Structure implementations
void Query::accept(AstVisitor* visitor) { visitor->visitQuery(this); }
void QuerySpecification::accept(AstVisitor* visitor) { visitor->visitQuerySpecification(this); }
void Select::accept(AstVisitor* visitor) { visitor->visitSelect(this); }
void SingleColumn::accept(AstVisitor* visitor) { visitor->visitSingleColumn(this); }
void AllColumns::accept(AstVisitor* visitor) { visitor->visitAllColumns(this); }
void With::accept(AstVisitor* visitor) { visitor->visitWith(this); }
void WithQuery::accept(AstVisitor* visitor) { visitor->visitWithQuery(this); }
void OrderBy::accept(AstVisitor* visitor) { visitor->visitOrderBy(this); }
void SortItem::accept(AstVisitor* visitor) { visitor->visitSortItem(this); }
void GroupBy::accept(AstVisitor* visitor) { visitor->visitGroupBy(this); }
void SimpleGroupBy::accept(AstVisitor* visitor) { visitor->visitSimpleGroupBy(this); }
void GroupingSets::accept(AstVisitor* visitor) { visitor->visitGroupingSets(this); }
void Cube::accept(AstVisitor* visitor) { visitor->visitCube(this); }
void Rollup::accept(AstVisitor* visitor) { visitor->visitRollup(this); }
void Offset::accept(AstVisitor* visitor) { visitor->visitOffset(this); }

// Relation implementations
void Table::accept(AstVisitor* visitor) { visitor->visitTable(this); }
void AliasedRelation::accept(AstVisitor* visitor) { visitor->visitAliasedRelation(this); }
void SampledRelation::accept(AstVisitor* visitor) { visitor->visitSampledRelation(this); }
void TableSubquery::accept(AstVisitor* visitor) { visitor->visitTableSubquery(this); }
void Lateral::accept(AstVisitor* visitor) { visitor->visitLateral(this); }
void Unnest::accept(AstVisitor* visitor) { visitor->visitUnnest(this); }
void Values::accept(AstVisitor* visitor) { visitor->visitValues(this); }

// Join implementations
void Join::accept(AstVisitor* visitor) { visitor->visitJoin(this); }
void JoinOn::accept(AstVisitor* visitor) { visitor->visitJoinOn(this); }
void JoinUsing::accept(AstVisitor* visitor) { visitor->visitJoinUsing(this); }
void NaturalJoin::accept(AstVisitor* visitor) { visitor->visitNaturalJoin(this); }

// Set Operation implementations
void Union::accept(AstVisitor* visitor) { visitor->visitUnion(this); }
void Intersect::accept(AstVisitor* visitor) { visitor->visitIntersect(this); }
void Except::accept(AstVisitor* visitor) { visitor->visitExcept(this); }

// DDL Statement implementations
void CreateTable::accept(AstVisitor* visitor) { visitor->visitCreateTable(this); }
void CreateTableAsSelect::accept(AstVisitor* visitor) { visitor->visitCreateTableAsSelect(this); }
void CreateView::accept(AstVisitor* visitor) { visitor->visitCreateView(this); }
void CreateMaterializedView::accept(AstVisitor* visitor) { visitor->visitCreateMaterializedView(this); }
void CreateSchema::accept(AstVisitor* visitor) { visitor->visitCreateSchema(this); }
void CreateFunction::accept(AstVisitor* visitor) { visitor->visitCreateFunction(this); }
void CreateRole::accept(AstVisitor* visitor) { visitor->visitCreateRole(this); }
void CreateType::accept(AstVisitor* visitor) { visitor->visitCreateType(this); }
void DropTable::accept(AstVisitor* visitor) { visitor->visitDropTable(this); }
void DropView::accept(AstVisitor* visitor) { visitor->visitDropView(this); }
void DropMaterializedView::accept(AstVisitor* visitor) { visitor->visitDropMaterializedView(this); }
void DropSchema::accept(AstVisitor* visitor) { visitor->visitDropSchema(this); }

// DML Statement implementations
void Insert::accept(AstVisitor* visitor) { visitor->visitInsert(this); }
void UpdateAssignment::accept(AstVisitor* visitor) { visitor->visitUpdateAssignment(this); }
void Update::accept(AstVisitor* visitor) { visitor->visitUpdate(this); }
void Delete::accept(AstVisitor* visitor) { visitor->visitDelete(this); }

// Utility Statement implementations
void Explain::accept(AstVisitor* visitor) { visitor->visitExplain(this); }
void Analyze::accept(AstVisitor* visitor) { visitor->visitAnalyze(this); }
void Call::accept(AstVisitor* visitor) { visitor->visitCall(this); }

// Transaction Statement implementations
void StartTransaction::accept(AstVisitor* visitor) { visitor->visitStartTransaction(this); }
void Commit::accept(AstVisitor* visitor) { visitor->visitCommit(this); }
void Rollback::accept(AstVisitor* visitor) { visitor->visitRollback(this); }

// Table Element implementations
void ColumnDefinition::accept(AstVisitor* visitor) { visitor->visitColumnDefinition(this); }
void LikeClause::accept(AstVisitor* visitor) { visitor->visitLikeClause(this); }
void ConstraintSpecification::accept(AstVisitor* visitor) { visitor->visitConstraintSpecification(this); }

// Support Class implementations
void Property::accept(AstVisitor* visitor) { visitor->visitProperty(this); }
void CallArgument::accept(AstVisitor* visitor) { visitor->visitCallArgument(this); }
void Window::accept(AstVisitor* visitor) { visitor->visitWindow(this); }
void WindowFrame::accept(AstVisitor* visitor) { visitor->visitWindowFrame(this); }
void FrameBound::accept(AstVisitor* visitor) { visitor->visitFrameBound(this); }
void PrincipalSpecification::accept(AstVisitor* visitor) { visitor->visitPrincipalSpecification(this); }
void GrantorSpecification::accept(AstVisitor* visitor) { visitor->visitGrantorSpecification(this); }
void Isolation::accept(AstVisitor* visitor) { visitor->visitIsolation(this); }
void TransactionAccessMode::accept(AstVisitor* visitor) { visitor->visitTransactionAccessMode(this); }
void SqlParameterDeclaration::accept(AstVisitor* visitor) { visitor->visitSqlParameterDeclaration(this); }
void RoutineCharacteristics::accept(AstVisitor* visitor) { visitor->visitRoutineCharacteristics(this); }
void ExternalBodyReference::accept(AstVisitor* visitor) { visitor->visitExternalBodyReference(this); }
void Return::accept(AstVisitor* visitor) { visitor->visitReturn(this); }
void ExplainFormat::accept(AstVisitor* visitor) { visitor->visitExplainFormat(this); }
void ExplainType::accept(AstVisitor* visitor) { visitor->visitExplainType(this); }

} // namespace facebook::velox::sql