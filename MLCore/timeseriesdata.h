// Copyright 2015 Byungkuk Choi.

#ifndef MLCORE_TIMESERIESDATA_H_
#define MLCORE_TIMESERIESDATA_H_

#include <utility>
#include <map>

#include "MLCore/mathtypes.h"

namespace ML {

typedef std::pair<int, VecN> T_Sample;
typedef std::map<int, VecN> TimeSeriesMap;

T_Sample MakeTimeSample(const int& f, int n_d, ...);

}  // namespace ML

#endif  // MLCORE_TIMESERIESDATA_H_
