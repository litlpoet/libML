// Copyright 2015 Byungkuk Choi.

#include "gaussianinterpolationnoisy_imple.h"

namespace ML
{

GaussianInterpolationNoisy::GaussianInterpolationNoisy(size_t const&        n_dim_D,
                                                       TimeSeriesMap const& time_series_map)
    : Interpolation(n_dim_D, time_series_map)
    , _p(new Imple(n_dim_D, dataDimension(), time_series_map))
{
}

GaussianInterpolationNoisy::GaussianInterpolationNoisy(TimeSeriesDense const& time_series_dense,
                                                       size_t const&          sampling_rate)
    : Interpolation(time_series_dense)
    , _p(new Imple(timeDimension(), dataDimension(), sampling_rate, time_series_dense))
{
}

GaussianInterpolationNoisy::~GaussianInterpolationNoisy() = default;

bool
GaussianInterpolationNoisy::solve(Scalar const& lambda,
                                  Scalar const& alpha,
                                  MatNxN*       Mu,
                                  MatNxN*       Sig_x_y)
{
  MatNxN S;
  bool   res = _p->solveSigmaAndMu(timeDimension(), dataDimension(), lambda, alpha, Mu, &S);
  if (Sig_x_y)
    *Sig_x_y = S;
  return res;
}

void
GaussianInterpolationNoisy::setBoundaryConstraint(short const& b_type)
{
  auto b = static_cast<BoundaryType const>(b_type);
  if (_p->_boundary != b)
    _p->_prior_dirty = true;
  _p->_boundary      = b;
}

}  // namespace ML
