// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_GAUSSIANINTERPOLATION_H_
#define MLINTERPOLATION_GAUSSIANINTERPOLATION_H_

#include <ml/interpolation/interpolation.h>

namespace ML
{

class GaussianInterpolation : public Interpolation
{
 public:
  GaussianInterpolation(size_t const& D, TimeSeriesMap const& time_series_data);

  ~GaussianInterpolation() final;

  bool
  solve(Scalar const& lambda, Scalar const& alpha, MatNxN* Mu, MatNxN* Sigma = nullptr) final;

 private:
  class Imple;
  unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_GAUSSIANINTERPOLATION_H_
