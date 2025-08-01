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

#include "axiom/optimizer/connectors/ConnectorMetadata.h"
#include "velox/runner/Runner.h"

namespace facebook::velox::connector {

/// A runner::SplitSource that encapsulates a connector::SplitSource.
/// runner::SplitSource does not depend on ConnectorMetadata.h, thus we have a
/// proxy between the two.
class ConnectorSplitSource : public runner::SplitSource {
 public:
  ConnectorSplitSource(std::shared_ptr<connector::SplitSource> source)
      : source_(std::move(source)) {}

  std::vector<SplitAndGroup> getSplits(uint64_t targetBytes) override;

 private:
  std::shared_ptr<connector::SplitSource> source_;
};

/// Generic SplitSourceFactory that delegates the work to ConnectorMetadata.
class ConnectorSplitSourceFactory : public runner::SplitSourceFactory {
 public:
  ConnectorSplitSourceFactory(SplitOptions options = {}) : options_(options) {}

  std::shared_ptr<runner::SplitSource> splitSourceForScan(
      const core::TableScanNode& scan) override;

 protected:
  SplitOptions options_;
};

} // namespace facebook::velox::connector
