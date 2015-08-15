// Copyright (c) 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_MULTILEVELBSPLINEINTERPOLATION_H_
#define MLINTERPOLATION_MULTILEVELBSPLINEINTERPOLATION_H_

#include <memory>
#include "MLCore/timeseriesdata.h"

namespace ML {

class MultiLevelBSplineInterpolation {
 public:
  MultiLevelBSplineInterpolation(const int& D,
                                 const TimeSeriesMap& time_series_map);

  ~MultiLevelBSplineInterpolation();

  bool solve(const int& initial_n_knots, const int& level, MatNxN* result_mat);

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML
#endif  // MLINTERPOLATION_MULTILEVELBSPLINEINTERPOLATION_H_
