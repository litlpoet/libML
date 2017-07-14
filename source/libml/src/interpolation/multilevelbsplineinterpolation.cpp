// Copyright (c) 2015 Byungkuk Choi.

#include "multilevelbsplineinterpolation_imple.h"

namespace ML
{

MultiLevelBSplineInterpolation::MultiLevelBSplineInterpolation(size_t const&        D,
                                                               TimeSeriesMap const& time_series_map)
    : Interpolation(D, time_series_map)
    , _p(new Imple(D, dataDimension(), time_series_map))
{
}

MultiLevelBSplineInterpolation::~MultiLevelBSplineInterpolation() = default;

bool
MultiLevelBSplineInterpolation::solve(size_t const& initial_n_knots,
                                      size_t const& level,
                                      MatNxN*       result_mat)
{
  std::cout << "start to solve multi-level bspline" << std::endl;
  auto const& D   = timeDimension();
  auto const& D_X = dataDimension();
  *result_mat     = MatNxN::Zero(D, D_X);
  *_p->_X_l       = *_p->_X;  // deep copy;
  MatNxN M(D, D_X);
  for (auto n_knots = initial_n_knots, l = 0ul; l < level; n_knots *= 2, ++l)
  {
    _p->solveBSpline(D, D_X, n_knots, &M);
    _p->updateNextLevelTargets();
    *result_mat += M;
  }
  std::cout << "solve finished" << std::endl;
  return true;
}

}  // namespace ML
