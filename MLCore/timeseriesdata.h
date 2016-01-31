// Copyright 2015 Byungkuk Choi.

#ifndef MLCORE_TIMESERIESDATA_H_
#define MLCORE_TIMESERIESDATA_H_

#include <map>
#include <utility>
#include <vector>

#include "MLCore/mathtypes.h"

namespace ML {

using T_Sample = std::pair<int, VecN>;
using TimeSeriesMap = std::map<int, VecN>;
using TimeSeriesDense = std::vector<VecN>;

T_Sample MakeTimeSample(int const& f, int n_d, ...);

}  // namespace ML

#endif  // MLCORE_TIMESERIESDATA_H_
