// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_
#define MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_

#include <ml/interpolation/interpolation.h>

namespace ML
{

class GaussianInterpolationNoisy : public Interpolation
{
 public:
  GaussianInterpolationNoisy(size_t const& D, TimeSeriesMap const& time_series_map);

  explicit GaussianInterpolationNoisy(TimeSeriesDense const& time_series_dense,
                                      size_t const&          sampling_rate = 1);

  ~GaussianInterpolationNoisy();

  bool
  solve(Scalar const& lambda, Scalar const& alpha, MatNxN* Mu, MatNxN* Sigma = nullptr) final;

  void
  setBoundaryConstraint(short const& b_type);

 private:
  class Imple;
  unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_GAUSSIANINTERPOLATIONNOISY_H_
