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

#pragma once

#include "axiom/connectors/ConnectorMetadata.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive {

/// Describes a single partition of a Hive table. If the table is
/// bucketed, this resolves to a single file. If the table is
/// partitioned and not bucketed, this resolves to a leaf
/// directory. If the table is not bucketed and not partitioned,
/// this resolves to the directory corresponding to the table.
struct HivePartitionHandle : public PartitionHandle {
  HivePartitionHandle(
      const std::unordered_map<std::string, std::optional<std::string>>
          partitionKeys,
      std::optional<int32_t> tableBucketNumber)
      : partitionKeys(partitionKeys), tableBucketNumber(tableBucketNumber) {}

  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys;
  const std::optional<int32_t> tableBucketNumber;
};

class HiveConnectorSession : public connector::ConnectorSession {
 public:
  ~HiveConnectorSession() override = default;
};

class HivePartitionType : public connector::PartitionType {
 public:
  HivePartitionType(
      int32_t numBuckets,
      std::vector<TypePtr> partitionKeyTypes = {})
      : numBuckets_(numBuckets),
        partitionKeyTypes_(std::move(partitionKeyTypes)) {}

  std::optional<int32_t> numPartitions() const override {
    return numBuckets_;
  }

  // Types are compatible if the bucket count one is an interger multiple of the
  // other. The partition to use for copartitioning is the one with the fewer
  // buckets.
  const PartitionType* copartition(const PartitionType& any) const override;

  core::PartitionFunctionSpecPtr makeSpec(
      const std::vector<column_index_t>& channels,
      const std::vector<VectorPtr>& constants,
      bool isLocal) const override;

  const std::vector<TypePtr>& partitionKeyTypes() const override {
    return partitionKeyTypes_;
  }

  std::string toString() const override;

 private:
  const int32_t numBuckets_;
  const std::vector<TypePtr> partitionKeyTypes_;
};

/// Describes a Hive table layout. Adds a file format and a list of
/// Hive partitioning columns and an optional bucket count to the base
/// TableLayout. The partitioning in TableLayout referes to bucketing.
/// 'numBuckets' is the number of Hive buckets if
/// 'partitionColumns' is not empty. 'hivePartitionColumns' refers to Hive
/// partitioning, i.e. columns whose value gives a directory in the ile storage
/// tree.
class HiveTableLayout : public TableLayout {
 public:
  HiveTableLayout(
      std::string name,
      const Table* table,
      connector::Connector* connector,
      std::vector<const Column*> columns,
      std::vector<const Column*> partitioning,
      std::vector<const Column*> orderColumns,
      std::vector<SortOrder> sortOrder,
      std::vector<const Column*> lookupKeys,
      std::vector<const Column*> hivePartitionColumns,
      dwio::common::FileFormat fileFormat,
      std::optional<int32_t> numBuckets = std::nullopt)
      : TableLayout{std::move(name), table, connector, std::move(columns), std::move(partitioning), std::move(orderColumns), std::move(sortOrder), std::move(lookupKeys), true},
        fileFormat_{fileFormat},
        hivePartitionColumns_{std::move(hivePartitionColumns)},
        numBuckets_{numBuckets},
        partitionType_{
            numBuckets.value_or(0),
            extractPartitionKeyTypes(partitionColumns())} {}

  const PartitionType* partitionType() const override {
    return partitionColumns().empty() ? nullptr : &partitionType_;
  }

  dwio::common::FileFormat fileFormat() const {
    return fileFormat_;
  }

  const std::vector<const Column*>& hivePartitionColumns() const {
    return hivePartitionColumns_;
  }

  std::optional<int32_t> numBuckets() const {
    return numBuckets_;
  }

 protected:
  const dwio::common::FileFormat fileFormat_;
  const std::vector<const Column*> hivePartitionColumns_;
  const std::optional<int32_t> numBuckets_;

 private:
  static std::vector<TypePtr> extractPartitionKeyTypes(
      const std::vector<const Column*>& partitionColumns) {
    std::vector<TypePtr> types;
    types.reserve(partitionColumns.size());
    for (const auto* column : partitionColumns) {
      types.push_back(column->type());
    }
    return types;
  }

  const HivePartitionType partitionType_;
};

class HiveConnectorMetadata : public ConnectorMetadata {
 public:
  explicit HiveConnectorMetadata(HiveConnector* hiveConnector)
      : hiveConnector_(hiveConnector),
        hiveConfig_(
            std::make_shared<HiveConfig>(hiveConnector->connectorConfig())) {}

  ColumnHandlePtr createColumnHandle(
      const TableLayout& layout,
      const std::string& columnName,
      std::vector<common::Subfield> subfields = {},
      std::optional<TypePtr> castToType = std::nullopt,
      SubfieldMapping subfieldMapping = {}) override;

  ConnectorTableHandlePtr createTableHandle(
      const TableLayout& layout,
      std::vector<ColumnHandlePtr> columnHandles,
      core::ExpressionEvaluator& evaluator,
      std::vector<core::TypedExprPtr> filters,
      std::vector<core::TypedExprPtr>& rejectedFilters,
      RowTypePtr dataColumns,
      std::optional<LookupKeys> lookupKeys) override;

  ConnectorInsertTableHandlePtr createInsertTableHandle(
      const TableLayout& layout,
      const RowTypePtr& rowType,
      const folly::F14FastMap<std::string, std::string>& options,
      WriteKind kind,
      const ConnectorSessionPtr& session) override;

  void finishWrite(
      const TableLayout& layout,
      const velox::connector::ConnectorInsertTableHandlePtr& handle,
      WriteKind kind,
      const ConnectorSessionPtr& session,
      bool success,
      const std::vector<velox::RowVectorPtr>& results) override {
    VELOX_UNSUPPORTED();
  }

  std::vector<ColumnHandlePtr> rowIdHandles(
      const TableLayout& layout,
      WriteKind kind) override {
    VELOX_UNSUPPORTED();
  }

  virtual dwio::common::FileFormat fileFormat() const = 0;

 protected:
  virtual void ensureInitialized() const {}

  virtual void validateOptions(
      const folly::F14FastMap<std::string, std::string>& options) const;

  virtual std::shared_ptr<connector::hive::LocationHandle> makeLocationHandle(
      std::string targetDirectory,
      std::optional<std::string> writeDirectory,
      connector::hive::LocationHandle::TableType tableType =
          connector::hive::LocationHandle::TableType::kNew) = 0;

  /// Returns the path to the filesystem root for the data managed by
  /// 'this'. Directories inside this correspond to schemas and
  /// tables.
  virtual std::string dataPath() const = 0;

  virtual std::string makeStagingDirectory() {
    VELOX_UNSUPPORTED();
  }

  HiveConnector* const hiveConnector_;
  const std::shared_ptr<HiveConfig> hiveConfig_;
};

} // namespace facebook::velox::connector::hive
