// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_GAUSSIANINTERPOLATION_H_
#define MLINTERPOLATION_GAUSSIANINTERPOLATION_H_

#include "interpolation/interpolation.h"

namespace ML {

class GaussianInterpolation : public Interpolation {
 public:
  GaussianInterpolation(int const& D, TimeSeriesMap const& time_series_data);

  ~GaussianInterpolation() final;

  bool solve(float const& lambda, float const& alpha, MatNxN* Mu,
             MatNxN* Sigma = nullptr) final;

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_GAUSSIANINTERPOLATION_H_
