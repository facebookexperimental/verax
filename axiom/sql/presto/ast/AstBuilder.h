// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <antlr4-runtime/antlr4-runtime.h>
#include <iostream>
#include "axiom/sql/presto/ast/AstNodesAll.h"
#include "axiom/sql/presto/grammar/PrestoSqlParser.h"
#include "axiom/sql/presto/grammar/PrestoSqlVisitor.h"

namespace axiom::sql::presto {

class AstBuilder : public PrestoSqlVisitor {
 public:
  AstBuilder(bool enableTracing = false) : enableTracing_{enableTracing} {}

  antlrcpp::Any visitSingleStatement(
      PrestoSqlParser::SingleStatementContext* ctx) override;

  antlrcpp::Any visitStandaloneExpression(
      PrestoSqlParser::StandaloneExpressionContext* ctx) override;

  antlrcpp::Any visitStandaloneRoutineBody(
      PrestoSqlParser::StandaloneRoutineBodyContext* ctx) override;

  antlrcpp::Any visitStatementDefault(
      PrestoSqlParser::StatementDefaultContext* ctx) override;

  antlrcpp::Any visitUse(PrestoSqlParser::UseContext* ctx) override;

  antlrcpp::Any visitCreateSchema(
      PrestoSqlParser::CreateSchemaContext* ctx) override;

  antlrcpp::Any visitDropSchema(
      PrestoSqlParser::DropSchemaContext* ctx) override;

  antlrcpp::Any visitRenameSchema(
      PrestoSqlParser::RenameSchemaContext* ctx) override;

  antlrcpp::Any visitCreateTableAsSelect(
      PrestoSqlParser::CreateTableAsSelectContext* ctx) override;

  antlrcpp::Any visitCreateTable(
      PrestoSqlParser::CreateTableContext* ctx) override;

  antlrcpp::Any visitDropTable(PrestoSqlParser::DropTableContext* ctx) override;

  antlrcpp::Any visitInsertInto(
      PrestoSqlParser::InsertIntoContext* ctx) override;

  antlrcpp::Any visitDelete(PrestoSqlParser::DeleteContext* ctx) override;

  antlrcpp::Any visitTruncateTable(
      PrestoSqlParser::TruncateTableContext* ctx) override;

  antlrcpp::Any visitRenameTable(
      PrestoSqlParser::RenameTableContext* ctx) override;

  antlrcpp::Any visitRenameColumn(
      PrestoSqlParser::RenameColumnContext* ctx) override;

  antlrcpp::Any visitDropColumn(
      PrestoSqlParser::DropColumnContext* ctx) override;

  antlrcpp::Any visitAddColumn(PrestoSqlParser::AddColumnContext* ctx) override;

  antlrcpp::Any visitAddConstraint(
      PrestoSqlParser::AddConstraintContext* ctx) override;

  antlrcpp::Any visitDropConstraint(
      PrestoSqlParser::DropConstraintContext* ctx) override;

  antlrcpp::Any visitAlterColumnSetNotNull(
      PrestoSqlParser::AlterColumnSetNotNullContext* ctx) override;

  antlrcpp::Any visitAlterColumnDropNotNull(
      PrestoSqlParser::AlterColumnDropNotNullContext* ctx) override;

  antlrcpp::Any visitSetTableProperties(
      PrestoSqlParser::SetTablePropertiesContext* ctx) override;

  antlrcpp::Any visitAnalyze(PrestoSqlParser::AnalyzeContext* ctx) override;

  antlrcpp::Any visitCreateType(
      PrestoSqlParser::CreateTypeContext* ctx) override;

  antlrcpp::Any visitCreateView(
      PrestoSqlParser::CreateViewContext* ctx) override;

  antlrcpp::Any visitRenameView(
      PrestoSqlParser::RenameViewContext* ctx) override;

  antlrcpp::Any visitDropView(PrestoSqlParser::DropViewContext* ctx) override;

  antlrcpp::Any visitCreateMaterializedView(
      PrestoSqlParser::CreateMaterializedViewContext* ctx) override;

  antlrcpp::Any visitDropMaterializedView(
      PrestoSqlParser::DropMaterializedViewContext* ctx) override;

  antlrcpp::Any visitRefreshMaterializedView(
      PrestoSqlParser::RefreshMaterializedViewContext* ctx) override;

  antlrcpp::Any visitCreateFunction(
      PrestoSqlParser::CreateFunctionContext* ctx) override;

  antlrcpp::Any visitAlterFunction(
      PrestoSqlParser::AlterFunctionContext* ctx) override;

  antlrcpp::Any visitDropFunction(
      PrestoSqlParser::DropFunctionContext* ctx) override;

  antlrcpp::Any visitCall(PrestoSqlParser::CallContext* ctx) override;

  antlrcpp::Any visitCreateRole(
      PrestoSqlParser::CreateRoleContext* ctx) override;

  antlrcpp::Any visitDropRole(PrestoSqlParser::DropRoleContext* ctx) override;

  antlrcpp::Any visitGrantRoles(
      PrestoSqlParser::GrantRolesContext* ctx) override;

  antlrcpp::Any visitRevokeRoles(
      PrestoSqlParser::RevokeRolesContext* ctx) override;

  antlrcpp::Any visitSetRole(PrestoSqlParser::SetRoleContext* ctx) override;

  antlrcpp::Any visitGrant(PrestoSqlParser::GrantContext* ctx) override;

  antlrcpp::Any visitRevoke(PrestoSqlParser::RevokeContext* ctx) override;

  antlrcpp::Any visitShowGrants(
      PrestoSqlParser::ShowGrantsContext* ctx) override;

  antlrcpp::Any visitExplain(PrestoSqlParser::ExplainContext* ctx) override;

  antlrcpp::Any visitShowCreateTable(
      PrestoSqlParser::ShowCreateTableContext* ctx) override;

  antlrcpp::Any visitShowCreateView(
      PrestoSqlParser::ShowCreateViewContext* ctx) override;

  antlrcpp::Any visitShowCreateMaterializedView(
      PrestoSqlParser::ShowCreateMaterializedViewContext* ctx) override;

  antlrcpp::Any visitShowCreateFunction(
      PrestoSqlParser::ShowCreateFunctionContext* ctx) override;

  antlrcpp::Any visitShowTables(
      PrestoSqlParser::ShowTablesContext* ctx) override;

  antlrcpp::Any visitShowSchemas(
      PrestoSqlParser::ShowSchemasContext* ctx) override;

  antlrcpp::Any visitShowCatalogs(
      PrestoSqlParser::ShowCatalogsContext* ctx) override;

  antlrcpp::Any visitShowColumns(
      PrestoSqlParser::ShowColumnsContext* ctx) override;

  antlrcpp::Any visitShowStats(PrestoSqlParser::ShowStatsContext* ctx) override;

  antlrcpp::Any visitShowStatsForQuery(
      PrestoSqlParser::ShowStatsForQueryContext* ctx) override;

  antlrcpp::Any visitShowRoles(PrestoSqlParser::ShowRolesContext* ctx) override;

  antlrcpp::Any visitShowRoleGrants(
      PrestoSqlParser::ShowRoleGrantsContext* ctx) override;

  antlrcpp::Any visitShowFunctions(
      PrestoSqlParser::ShowFunctionsContext* ctx) override;

  antlrcpp::Any visitShowSession(
      PrestoSqlParser::ShowSessionContext* ctx) override;

  antlrcpp::Any visitSetSession(
      PrestoSqlParser::SetSessionContext* ctx) override;

  antlrcpp::Any visitResetSession(
      PrestoSqlParser::ResetSessionContext* ctx) override;

  antlrcpp::Any visitStartTransaction(
      PrestoSqlParser::StartTransactionContext* ctx) override;

  antlrcpp::Any visitCommit(PrestoSqlParser::CommitContext* ctx) override;

  antlrcpp::Any visitRollback(PrestoSqlParser::RollbackContext* ctx) override;

  antlrcpp::Any visitPrepare(PrestoSqlParser::PrepareContext* ctx) override;

  antlrcpp::Any visitDeallocate(
      PrestoSqlParser::DeallocateContext* ctx) override;

  antlrcpp::Any visitExecute(PrestoSqlParser::ExecuteContext* ctx) override;

  antlrcpp::Any visitDescribeInput(
      PrestoSqlParser::DescribeInputContext* ctx) override;

  antlrcpp::Any visitDescribeOutput(
      PrestoSqlParser::DescribeOutputContext* ctx) override;

  antlrcpp::Any visitUpdate(PrestoSqlParser::UpdateContext* ctx) override;

  antlrcpp::Any visitQuery(PrestoSqlParser::QueryContext* ctx) override;

  antlrcpp::Any visitWith(PrestoSqlParser::WithContext* ctx) override;

  antlrcpp::Any visitTableElement(
      PrestoSqlParser::TableElementContext* ctx) override;

  antlrcpp::Any visitColumnDefinition(
      PrestoSqlParser::ColumnDefinitionContext* ctx) override;

  antlrcpp::Any visitLikeClause(
      PrestoSqlParser::LikeClauseContext* ctx) override;

  antlrcpp::Any visitProperties(
      PrestoSqlParser::PropertiesContext* ctx) override;

  antlrcpp::Any visitProperty(PrestoSqlParser::PropertyContext* ctx) override;

  antlrcpp::Any visitSqlParameterDeclaration(
      PrestoSqlParser::SqlParameterDeclarationContext* ctx) override;

  antlrcpp::Any visitRoutineCharacteristics(
      PrestoSqlParser::RoutineCharacteristicsContext* ctx) override;

  antlrcpp::Any visitRoutineCharacteristic(
      PrestoSqlParser::RoutineCharacteristicContext* ctx) override;

  antlrcpp::Any visitAlterRoutineCharacteristics(
      PrestoSqlParser::AlterRoutineCharacteristicsContext* ctx) override;

  antlrcpp::Any visitAlterRoutineCharacteristic(
      PrestoSqlParser::AlterRoutineCharacteristicContext* ctx) override;

  antlrcpp::Any visitRoutineBody(
      PrestoSqlParser::RoutineBodyContext* ctx) override;

  antlrcpp::Any visitReturnStatement(
      PrestoSqlParser::ReturnStatementContext* ctx) override;

  antlrcpp::Any visitExternalBodyReference(
      PrestoSqlParser::ExternalBodyReferenceContext* ctx) override;

  antlrcpp::Any visitLanguage(PrestoSqlParser::LanguageContext* ctx) override;

  antlrcpp::Any visitDeterminism(
      PrestoSqlParser::DeterminismContext* ctx) override;

  antlrcpp::Any visitNullCallClause(
      PrestoSqlParser::NullCallClauseContext* ctx) override;

  antlrcpp::Any visitExternalRoutineName(
      PrestoSqlParser::ExternalRoutineNameContext* ctx) override;

  antlrcpp::Any visitQueryNoWith(
      PrestoSqlParser::QueryNoWithContext* ctx) override;

  antlrcpp::Any visitQueryTermDefault(
      PrestoSqlParser::QueryTermDefaultContext* ctx) override;

  antlrcpp::Any visitSetOperation(
      PrestoSqlParser::SetOperationContext* ctx) override;

  antlrcpp::Any visitQueryPrimaryDefault(
      PrestoSqlParser::QueryPrimaryDefaultContext* ctx) override;

  antlrcpp::Any visitTable(PrestoSqlParser::TableContext* ctx) override;

  antlrcpp::Any visitInlineTable(
      PrestoSqlParser::InlineTableContext* ctx) override;

  antlrcpp::Any visitSubquery(PrestoSqlParser::SubqueryContext* ctx) override;

  antlrcpp::Any visitSortItem(PrestoSqlParser::SortItemContext* ctx) override;

  antlrcpp::Any visitQuerySpecification(
      PrestoSqlParser::QuerySpecificationContext* ctx) override;

  antlrcpp::Any visitGroupBy(PrestoSqlParser::GroupByContext* ctx) override;

  antlrcpp::Any visitSingleGroupingSet(
      PrestoSqlParser::SingleGroupingSetContext* ctx) override;

  antlrcpp::Any visitRollup(PrestoSqlParser::RollupContext* ctx) override;

  antlrcpp::Any visitCube(PrestoSqlParser::CubeContext* ctx) override;

  antlrcpp::Any visitMultipleGroupingSets(
      PrestoSqlParser::MultipleGroupingSetsContext* ctx) override;

  antlrcpp::Any visitGroupingSet(
      PrestoSqlParser::GroupingSetContext* ctx) override;

  antlrcpp::Any visitNamedQuery(
      PrestoSqlParser::NamedQueryContext* ctx) override;

  antlrcpp::Any visitSetQuantifier(
      PrestoSqlParser::SetQuantifierContext* ctx) override;

  antlrcpp::Any visitSelectSingle(
      PrestoSqlParser::SelectSingleContext* ctx) override;

  antlrcpp::Any visitSelectAll(PrestoSqlParser::SelectAllContext* ctx) override;

  antlrcpp::Any visitRelationDefault(
      PrestoSqlParser::RelationDefaultContext* ctx) override;

  antlrcpp::Any visitJoinRelation(
      PrestoSqlParser::JoinRelationContext* ctx) override;

  antlrcpp::Any visitJoinType(PrestoSqlParser::JoinTypeContext* ctx) override;

  antlrcpp::Any visitJoinCriteria(
      PrestoSqlParser::JoinCriteriaContext* ctx) override;

  antlrcpp::Any visitSampledRelation(
      PrestoSqlParser::SampledRelationContext* ctx) override;

  antlrcpp::Any visitSampleType(
      PrestoSqlParser::SampleTypeContext* ctx) override;

  antlrcpp::Any visitAliasedRelation(
      PrestoSqlParser::AliasedRelationContext* ctx) override;

  antlrcpp::Any visitColumnAliases(
      PrestoSqlParser::ColumnAliasesContext* ctx) override;

  antlrcpp::Any visitTableName(PrestoSqlParser::TableNameContext* ctx) override;

  antlrcpp::Any visitSubqueryRelation(
      PrestoSqlParser::SubqueryRelationContext* ctx) override;

  antlrcpp::Any visitUnnest(PrestoSqlParser::UnnestContext* ctx) override;

  antlrcpp::Any visitLateral(PrestoSqlParser::LateralContext* ctx) override;

  antlrcpp::Any visitParenthesizedRelation(
      PrestoSqlParser::ParenthesizedRelationContext* ctx) override;

  antlrcpp::Any visitExpression(
      PrestoSqlParser::ExpressionContext* ctx) override;

  antlrcpp::Any visitLogicalNot(
      PrestoSqlParser::LogicalNotContext* ctx) override;

  antlrcpp::Any visitPredicated(
      PrestoSqlParser::PredicatedContext* ctx) override;

  antlrcpp::Any visitLogicalBinary(
      PrestoSqlParser::LogicalBinaryContext* ctx) override;

  antlrcpp::Any visitComparison(
      PrestoSqlParser::ComparisonContext* ctx) override;

  antlrcpp::Any visitQuantifiedComparison(
      PrestoSqlParser::QuantifiedComparisonContext* ctx) override;

  antlrcpp::Any visitBetween(PrestoSqlParser::BetweenContext* ctx) override;

  antlrcpp::Any visitInList(PrestoSqlParser::InListContext* ctx) override;

  antlrcpp::Any visitInSubquery(
      PrestoSqlParser::InSubqueryContext* ctx) override;

  antlrcpp::Any visitLike(PrestoSqlParser::LikeContext* ctx) override;

  antlrcpp::Any visitNullPredicate(
      PrestoSqlParser::NullPredicateContext* ctx) override;

  antlrcpp::Any visitDistinctFrom(
      PrestoSqlParser::DistinctFromContext* ctx) override;

  antlrcpp::Any visitValueExpressionDefault(
      PrestoSqlParser::ValueExpressionDefaultContext* ctx) override;

  antlrcpp::Any visitConcatenation(
      PrestoSqlParser::ConcatenationContext* ctx) override;

  antlrcpp::Any visitArithmeticBinary(
      PrestoSqlParser::ArithmeticBinaryContext* ctx) override;

  antlrcpp::Any visitArithmeticUnary(
      PrestoSqlParser::ArithmeticUnaryContext* ctx) override;

  antlrcpp::Any visitAtTimeZone(
      PrestoSqlParser::AtTimeZoneContext* ctx) override;

  antlrcpp::Any visitDereference(
      PrestoSqlParser::DereferenceContext* ctx) override;

  antlrcpp::Any visitTypeConstructor(
      PrestoSqlParser::TypeConstructorContext* ctx) override;

  antlrcpp::Any visitSpecialDateTimeFunction(
      PrestoSqlParser::SpecialDateTimeFunctionContext* ctx) override;

  antlrcpp::Any visitSubstring(PrestoSqlParser::SubstringContext* ctx) override;

  antlrcpp::Any visitCast(PrestoSqlParser::CastContext* ctx) override;

  antlrcpp::Any visitLambda(PrestoSqlParser::LambdaContext* ctx) override;

  antlrcpp::Any visitParenthesizedExpression(
      PrestoSqlParser::ParenthesizedExpressionContext* ctx) override;

  antlrcpp::Any visitParameter(PrestoSqlParser::ParameterContext* ctx) override;

  antlrcpp::Any visitNormalize(PrestoSqlParser::NormalizeContext* ctx) override;

  antlrcpp::Any visitIntervalLiteral(
      PrestoSqlParser::IntervalLiteralContext* ctx) override;

  antlrcpp::Any visitNumericLiteral(
      PrestoSqlParser::NumericLiteralContext* ctx) override;

  antlrcpp::Any visitBooleanLiteral(
      PrestoSqlParser::BooleanLiteralContext* ctx) override;

  antlrcpp::Any visitSimpleCase(
      PrestoSqlParser::SimpleCaseContext* ctx) override;

  antlrcpp::Any visitColumnReference(
      PrestoSqlParser::ColumnReferenceContext* ctx) override;

  antlrcpp::Any visitNullLiteral(
      PrestoSqlParser::NullLiteralContext* ctx) override;

  antlrcpp::Any visitRowConstructor(
      PrestoSqlParser::RowConstructorContext* ctx) override;

  antlrcpp::Any visitSubscript(PrestoSqlParser::SubscriptContext* ctx) override;

  antlrcpp::Any visitSubqueryExpression(
      PrestoSqlParser::SubqueryExpressionContext* ctx) override;

  antlrcpp::Any visitBinaryLiteral(
      PrestoSqlParser::BinaryLiteralContext* ctx) override;

  antlrcpp::Any visitCurrentUser(
      PrestoSqlParser::CurrentUserContext* ctx) override;

  antlrcpp::Any visitExtract(PrestoSqlParser::ExtractContext* ctx) override;

  antlrcpp::Any visitStringLiteral(
      PrestoSqlParser::StringLiteralContext* ctx) override;

  antlrcpp::Any visitArrayConstructor(
      PrestoSqlParser::ArrayConstructorContext* ctx) override;

  antlrcpp::Any visitFunctionCall(
      PrestoSqlParser::FunctionCallContext* ctx) override;

  antlrcpp::Any visitExists(PrestoSqlParser::ExistsContext* ctx) override;

  antlrcpp::Any visitPosition(PrestoSqlParser::PositionContext* ctx) override;

  antlrcpp::Any visitSearchedCase(
      PrestoSqlParser::SearchedCaseContext* ctx) override;

  antlrcpp::Any visitGroupingOperation(
      PrestoSqlParser::GroupingOperationContext* ctx) override;

  antlrcpp::Any visitBasicStringLiteral(
      PrestoSqlParser::BasicStringLiteralContext* ctx) override;

  antlrcpp::Any visitUnicodeStringLiteral(
      PrestoSqlParser::UnicodeStringLiteralContext* ctx) override;

  antlrcpp::Any visitNullTreatment(
      PrestoSqlParser::NullTreatmentContext* ctx) override;

  antlrcpp::Any visitTimeZoneInterval(
      PrestoSqlParser::TimeZoneIntervalContext* ctx) override;

  antlrcpp::Any visitTimeZoneString(
      PrestoSqlParser::TimeZoneStringContext* ctx) override;

  antlrcpp::Any visitComparisonOperator(
      PrestoSqlParser::ComparisonOperatorContext* ctx) override;

  antlrcpp::Any visitComparisonQuantifier(
      PrestoSqlParser::ComparisonQuantifierContext* ctx) override;

  antlrcpp::Any visitBooleanValue(
      PrestoSqlParser::BooleanValueContext* ctx) override;

  antlrcpp::Any visitInterval(PrestoSqlParser::IntervalContext* ctx) override;

  antlrcpp::Any visitIntervalField(
      PrestoSqlParser::IntervalFieldContext* ctx) override;

  antlrcpp::Any visitNormalForm(
      PrestoSqlParser::NormalFormContext* ctx) override;

  antlrcpp::Any visitTypes(PrestoSqlParser::TypesContext* ctx) override;

  antlrcpp::Any visitType(PrestoSqlParser::TypeContext* ctx) override;

  antlrcpp::Any visitTypeParameter(
      PrestoSqlParser::TypeParameterContext* ctx) override;

  antlrcpp::Any visitBaseType(PrestoSqlParser::BaseTypeContext* ctx) override;

  antlrcpp::Any visitWhenClause(
      PrestoSqlParser::WhenClauseContext* ctx) override;

  antlrcpp::Any visitFilter(PrestoSqlParser::FilterContext* ctx) override;

  antlrcpp::Any visitOver(PrestoSqlParser::OverContext* ctx) override;

  antlrcpp::Any visitWindowFrame(
      PrestoSqlParser::WindowFrameContext* ctx) override;

  antlrcpp::Any visitUnboundedFrame(
      PrestoSqlParser::UnboundedFrameContext* ctx) override;

  antlrcpp::Any visitCurrentRowBound(
      PrestoSqlParser::CurrentRowBoundContext* ctx) override;

  antlrcpp::Any visitBoundedFrame(
      PrestoSqlParser::BoundedFrameContext* ctx) override;

  antlrcpp::Any visitUpdateAssignment(
      PrestoSqlParser::UpdateAssignmentContext* ctx) override;

  antlrcpp::Any visitExplainFormat(
      PrestoSqlParser::ExplainFormatContext* ctx) override;

  antlrcpp::Any visitExplainType(
      PrestoSqlParser::ExplainTypeContext* ctx) override;

  antlrcpp::Any visitIsolationLevel(
      PrestoSqlParser::IsolationLevelContext* ctx) override;

  antlrcpp::Any visitTransactionAccessMode(
      PrestoSqlParser::TransactionAccessModeContext* ctx) override;

  antlrcpp::Any visitReadUncommitted(
      PrestoSqlParser::ReadUncommittedContext* ctx) override;

  antlrcpp::Any visitReadCommitted(
      PrestoSqlParser::ReadCommittedContext* ctx) override;

  antlrcpp::Any visitRepeatableRead(
      PrestoSqlParser::RepeatableReadContext* ctx) override;

  antlrcpp::Any visitSerializable(
      PrestoSqlParser::SerializableContext* ctx) override;

  antlrcpp::Any visitPositionalArgument(
      PrestoSqlParser::PositionalArgumentContext* ctx) override;

  antlrcpp::Any visitNamedArgument(
      PrestoSqlParser::NamedArgumentContext* ctx) override;

  antlrcpp::Any visitPrivilege(PrestoSqlParser::PrivilegeContext* ctx) override;

  antlrcpp::Any visitQualifiedName(
      PrestoSqlParser::QualifiedNameContext* ctx) override;

  antlrcpp::Any visitTableVersion(
      PrestoSqlParser::TableVersionContext* ctx) override;

  antlrcpp::Any visitTableversionasof(
      PrestoSqlParser::TableversionasofContext* ctx) override;

  antlrcpp::Any visitTableversionbefore(
      PrestoSqlParser::TableversionbeforeContext* ctx) override;

  antlrcpp::Any visitCurrentUserGrantor(
      PrestoSqlParser::CurrentUserGrantorContext* ctx) override;

  antlrcpp::Any visitCurrentRoleGrantor(
      PrestoSqlParser::CurrentRoleGrantorContext* ctx) override;

  antlrcpp::Any visitSpecifiedPrincipal(
      PrestoSqlParser::SpecifiedPrincipalContext* ctx) override;

  antlrcpp::Any visitUserPrincipal(
      PrestoSqlParser::UserPrincipalContext* ctx) override;

  antlrcpp::Any visitRolePrincipal(
      PrestoSqlParser::RolePrincipalContext* ctx) override;

  antlrcpp::Any visitUnspecifiedPrincipal(
      PrestoSqlParser::UnspecifiedPrincipalContext* ctx) override;

  antlrcpp::Any visitRoles(PrestoSqlParser::RolesContext* ctx) override;

  antlrcpp::Any visitUnquotedIdentifier(
      PrestoSqlParser::UnquotedIdentifierContext* ctx) override;

  antlrcpp::Any visitQuotedIdentifier(
      PrestoSqlParser::QuotedIdentifierContext* ctx) override;

  antlrcpp::Any visitBackQuotedIdentifier(
      PrestoSqlParser::BackQuotedIdentifierContext* ctx) override;

  antlrcpp::Any visitDigitIdentifier(
      PrestoSqlParser::DigitIdentifierContext* ctx) override;

  antlrcpp::Any visitDecimalLiteral(
      PrestoSqlParser::DecimalLiteralContext* ctx) override;

  antlrcpp::Any visitDoubleLiteral(
      PrestoSqlParser::DoubleLiteralContext* ctx) override;

  antlrcpp::Any visitIntegerLiteral(
      PrestoSqlParser::IntegerLiteralContext* ctx) override;

  antlrcpp::Any visitConstraintSpecification(
      PrestoSqlParser::ConstraintSpecificationContext* ctx) override;

  antlrcpp::Any visitNamedConstraintSpecification(
      PrestoSqlParser::NamedConstraintSpecificationContext* ctx) override;

  antlrcpp::Any visitUnnamedConstraintSpecification(
      PrestoSqlParser::UnnamedConstraintSpecificationContext* ctx) override;

  antlrcpp::Any visitConstraintType(
      PrestoSqlParser::ConstraintTypeContext* ctx) override;

  antlrcpp::Any visitConstraintQualifiers(
      PrestoSqlParser::ConstraintQualifiersContext* ctx) override;

  antlrcpp::Any visitConstraintQualifier(
      PrestoSqlParser::ConstraintQualifierContext* ctx) override;

  antlrcpp::Any visitConstraintRely(
      PrestoSqlParser::ConstraintRelyContext* ctx) override;

  antlrcpp::Any visitConstraintEnabled(
      PrestoSqlParser::ConstraintEnabledContext* ctx) override;

  antlrcpp::Any visitConstraintEnforced(
      PrestoSqlParser::ConstraintEnforcedContext* ctx) override;

  antlrcpp::Any visitNonReserved(
      PrestoSqlParser::NonReservedContext* ctx) override;

 private:
  template <typename T, typename TContext>
  std::vector<std::shared_ptr<T>> visitAllContext(
      const std::vector<TContext*>& ctxs) {
    std::vector<std::shared_ptr<T>> result;
    result.reserve(ctxs.size());
    for (auto ctx : ctxs) {
      result.emplace_back(visit(ctx).template as<std::shared_ptr<T>>());
    }
    return result;
  }

  template <typename T, typename TContext>
  std::shared_ptr<T> visitTyped(TContext* ctx) {
    if (ctx == nullptr) {
      return nullptr;
    }
    return visit(ctx).template as<std::shared_ptr<T>>();
  }

  template <typename TContext>
  axiom::sql::presto::ExpressionPtr visitExpression(TContext* ctx) {
    return visitTyped<axiom::sql::presto::Expression>(ctx);
  }

  axiom::sql::presto::IdentifierPtr visitIdentifier(
      PrestoSqlParser::IdentifierContext* ctx) {
    return visitTyped<axiom::sql::presto::Identifier>(ctx);
  }

  antlrcpp::Any aggregateResult(
      antlrcpp::Any aggregate,
      const antlrcpp::Any& nextResult) {
    if (aggregate.isNotNull()) {
      throw std::runtime_error(
          "aggregateResult called with non-null aggregate");
    }

    return nextResult;
  }

  axiom::sql::presto::QualifiedNamePtr getQualifiedName(
      PrestoSqlParser::QualifiedNameContext* ctx);

  void trace(const std::string& name) const;

  const bool enableTracing_;
};

} // namespace axiom::sql::presto
