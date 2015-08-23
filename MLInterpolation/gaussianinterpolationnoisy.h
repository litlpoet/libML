// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_
#define MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_

#include "MLInterpolation/interpolation.h"

namespace ML {

class GaussianInterpolationNoisy : public Interpolation {
 public:
  GaussianInterpolationNoisy(const int& D,
                             const TimeSeriesMap& time_series_data);

  ~GaussianInterpolationNoisy();

  bool solve(const float& lambda, MatNxN* Mu, MatNxN* Sigma = nullptr) final;

  void setBoundaryConstraint(const int& b_type);

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_
