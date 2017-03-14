// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_
#define MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_

#include "interpolation/interpolation.h"

namespace ML {

class GaussianInterpolationNoisy : public Interpolation {
 public:
  GaussianInterpolationNoisy(int const& D,
                             TimeSeriesMap const& time_series_map);

  explicit GaussianInterpolationNoisy(TimeSeriesDense const& time_series_dense,
                                      int const& sampling_rate = 1);

  ~GaussianInterpolationNoisy();

  bool solve(float const& lambda, float const& alpha, MatNxN* Mu,
             MatNxN* Sigma = nullptr) final;

  void setBoundaryConstraint(int const& b_type);

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_
