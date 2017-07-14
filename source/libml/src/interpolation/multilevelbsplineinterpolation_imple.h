#ifndef MULTILEVELBSPLINEINTERPOLATION_IMPLE_H
#define MULTILEVELBSPLINEINTERPOLATION_IMPLE_H

#include <ml/interpolation/multilevelbsplineinterpolation.h>

#include <iostream>
#include <vector>

#include <ml/core/splinebasis.h>

using std::vector;

namespace ML
{

class MultiLevelBSplineInterpolation::Imple
{
 public:
  unique_ptr<MatNxN> _X{nullptr};      // (sorted) given sample data (not changed)
  unique_ptr<MatNxN> _X_l{nullptr};    // (sorted) given sample data for each level
  unique_ptr<MatNxN> _X_apr{nullptr};  // (sorted) approximated sample
  unique_ptr<VecNi>  _frms{nullptr};   // (sorted) frame indexs
  vector<bool>       _has_data_at_t;
  Mat4x4             _CubicBSplBasis;

  Imple(size_t const& D, size_t const& D_X, TimeSeriesMap const& time_series_map)
      : _has_data_at_t(D, false)
  {
    CubicBSpline(&_CubicBSplBasis);
    _prepareSystem(D_X, time_series_map);
  }

  ~Imple() = default;

  void
  solveBSpline(size_t const& D, size_t const& D_X, size_t const& n_knots, MatNxN* M)
  {
    MatNxN CPS;
    _solveControlPoints(D, D_X, n_knots, &CPS);
    _interpolate(D, D_X, n_knots, CPS, M);
  }

  void
  updateNextLevelTargets()
  {
    MatNxN D = (*_X_l) - (*_X_apr);
    (*_X_l)  = D;
  }

 private:
  void
  _prepareSystem(size_t const& D_X, TimeSeriesMap const& time_series_map)
  {
    size_t N = time_series_map.size();
    _X.reset(new MatNxN(N, D_X));
    _X_l.reset(new MatNxN(N, D_X));
    _X_apr.reset(new MatNxN(N, D_X));
    _frms.reset(new VecNi(N));

    // make data matrix
    int i = 0;
    for (auto const& it : time_series_map)
    {
      _has_data_at_t[it.first] = true;
      (*_frms)(i)              = it.first;
      _X->row(i++)             = it.second.transpose();
    }
  }

  void
  _solveControlPoints(size_t const& D, size_t const& D_X, size_t const& n_knots, MatNxN* CPS)
  {
    // cubic B-spline need at least n_knots + 3 cps
    auto n_cps = n_knots + 3;

    // set control points and its weights zero initially
    *CPS   = MatNxN::Zero(n_cps, D_X);
    VecN W = VecN::Zero(n_cps);

    // variables
    size_t c_knot(0);
    size_t cp_id(0);
    Scalar t_l(0.0), sq_sum(0.0);
    RVec4  basis, basis_sq;

    for (auto i = 0l, n = _frms->size(); i < n; ++i)
    {
      c_knot   = _localSplineDomain(D, n_knots, (*_frms)(i), &t_l);
      basis    = _computeBasis(t_l);
      basis_sq = basis.cwiseAbs2();
      sq_sum   = basis_sq.sum();

      for (int j = 0; j < 4; ++j)
      {
        cp_id = c_knot + j;
        CPS->row(cp_id) += basis_sq(j) * _X_l->row(i) * basis(j) / sq_sum;
        W(cp_id) += basis_sq(j);
      }
    }

    for (int i = 0; i < n_cps; ++i)
    {
      if (W(i) > 1.0e-4f)
        CPS->row(i) /= W(i);
      else
        CPS->row(i).setZero();
    }
  }

  void
  _interpolate(
      size_t const& D, size_t const& D_X, size_t const& n_knots, MatNxN const& CPS, MatNxN* M)
  {
    for (auto f = 0ul, i = 0ul; f < D; ++f)
    {
      RVecN res = _evaluate(D, D_X, n_knots, f, CPS);
      M->row(f) = res;
      if (_has_data_at_t.at(f))
        _X_apr->row(i++) = res;
    }
  }

  Scalar
  _frameToTime(size_t const& D, size_t const& n_knots, size_t const& f)
  {
    return static_cast<Scalar>(n_knots * f) / D;
  }

  size_t
  _localSplineDomain(size_t const& D, size_t const& n_knots, size_t const& f, Scalar* t_l)
  {
    auto t    = _frameToTime(D, n_knots, f);
    auto knot = static_cast<size_t>(t);
    knot      = (knot > n_knots - 1) ? n_knots - 1 : knot;
    *t_l      = t - knot;
    return knot;
  }

  RVec4
  _computeBasis(Scalar const& t_l)
  {
    auto t_sq = t_l * t_l;
    return (1.0 / 6.0) * RVec4(t_sq * t_l, t_sq, t_l, 1.0) * _CubicBSplBasis;
  }

  RVecN
  _evaluate(
      size_t const& D, size_t const& D_X, size_t const& n_knots, size_t const& f, MatNxN const& CPS)
  {
    Scalar t_l(0.0);
    auto   knot = _localSplineDomain(D, n_knots, f, &t_l);
    return _evaluateLocal(t_l, CPS.block(knot, 0, 4, D_X));
  }

  template<typename Derived>
  RVecN
  _evaluateLocal(Scalar const& t_l, Eigen::MatrixBase<Derived> const& local_pts)
  {
    return _computeBasis(t_l) * local_pts;
  }
};

}  // namespace ML

#endif  // MULTILEVELBSPLINEINTERPOLATION_IMPLE_H
