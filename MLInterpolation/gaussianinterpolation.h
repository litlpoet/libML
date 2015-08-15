// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_GAUSSIANINTERPOLATION_H_
#define MLINTERPOLATION_GAUSSIANINTERPOLATION_H_

#include "MLInterpolation/interpolation.h"

namespace ML {

class GaussianInterpolation : public Interpolation {
 public:
  GaussianInterpolation(const int& D, const TimeSeriesMap& time_series_data);

  ~GaussianInterpolation();

  bool solve(const float& lambda, MatNxN* Mu, MatNxN* Sigma = nullptr) final;

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_GAUSSIANINTERPOLATION_H_
