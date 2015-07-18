// Copyright 2015 Byungkuk Choi.

#ifndef MLTIMESERIESDATA_HPP_
#define MLTIMESERIESDATA_HPP_

#include <utility>
#include <vector>

#include "MLCore/mathtypes.h"

namespace ML {

typedef std::pair<int, VecN> T_Sample;
typedef std::vector<T_Sample> TimeSeries;

}  // namespace ML

#endif  // MLTIMESERIESDATA_HPP_
