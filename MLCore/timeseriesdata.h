// Copyright 2015 Byungkuk Choi.

#ifndef MLCORE_TIMESERIESDATA_H_
#define MLCORE_TIMESERIESDATA_H_

#include <utility>
#include <vector>

#include "MLCore/mathtypes.h"

namespace ML {

typedef std::pair<int, VecN> T_Sample;
typedef std::vector<T_Sample> TimeSeries;

}  // namespace ML

#endif  // MLCORE_TIMESERIESDATA_H_
