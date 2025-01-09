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

#include "optimizer/Plan.h" //@manual
#include "optimizer/PlanUtils.h" //@manual
#include "optimizer/QueryGraph.h" //@manual
#include "velox/common/base/SimdUtil.h"
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox::optimizer {

Cost filterCost(CPSpan<Expr> conjuncts) {
  return Cost();
}

} // namespace facebook::velox::optimizer