// Copyright (c) 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_MULTILEVELBSPLINEINTERPOLATION_H_
#define MLINTERPOLATION_MULTILEVELBSPLINEINTERPOLATION_H_

#include <ml/interpolation/interpolation.h>

namespace ML {

class MultiLevelBSplineInterpolation : public Interpolation {
 public:
  MultiLevelBSplineInterpolation(int const& D,
                                 TimeSeriesMap const& time_series_map);

  ~MultiLevelBSplineInterpolation();

  bool solve(int const& initial_n_knots, int const& level,
             MatNxN* result_mat) final;

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML
#endif  // MLINTERPOLATION_MULTILEVELBSPLINEINTERPOLATION_H_
