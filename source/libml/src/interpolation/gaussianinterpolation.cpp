// Copyright 2015 Byungkuk Choi.

#include "gaussianinterpolation_imple.h"

namespace ML
{

/**
 * @brief GaussianInterpolation : Interpolating time-series samples using
 *                                Gaussian
 * @param D : the total number of discrete time samples
 * @param T : the time of interpolation region. interpolation will be done
 *            on [0, T] region
 * @param lambda : precision
 * @param time_series_map : vector of pair<int, VecN> data
 */
GaussianInterpolation::GaussianInterpolation(size_t const& D, TimeSeriesMap const& time_series_map)
    : Interpolation(D, time_series_map)
    , _p(new Imple(D, dataDimension(), time_series_map))
{
}

GaussianInterpolation::~GaussianInterpolation() = default;

bool
GaussianInterpolation::solve(Scalar const& lambda, Scalar const& alpha, MatNxN* Mu, MatNxN* Sigma)
{
  (void)alpha;
  bool res = _p->solveMean(timeDimension(), dataDimension(), Mu);
  if (Sigma)
    res |= _p->solveVariance(timeDimension(), lambda, Sigma);
  return res;
}

}  // namespace ML
