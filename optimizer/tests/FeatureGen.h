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

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::optimizer::test {

struct FeatureOptions {
  int32_t numFloat{10};
  int32_t numInt{10};
  int32_t numIdList{10};
  int32_t idListMaxCard{1000};
  int32_t idListMinCard{10};
  int32_t idListMaxDistinct{1000};
  int32_t numIdScoreList{5};

  /// Structs for use in reading the features. One field for each
  /// key. Filled in by makeFeatures().
  RowTypePtr floatStruct;
  RowTypePtr idListStruct;
  RowTypePtr idScoreListStruct;
};

std::vector<RowVectorPtr> makeFeatures(
    int32_t numBatches,
    int32_t batchSize,
    FeatureOptions& opts,
    memory::MemoryPool* pool);

} // namespace facebook::velox::optimizer::test
