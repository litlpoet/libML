// Copyright (c) 2015 Byungkuk Choi.

#include "interpolation/multilevelbsplineinterpolation.h"

#include <iostream>
#include <vector>

#include "core/splinebasis.h"

namespace ML {

class MultiLevelBSplineInterpolation::Imple {
 public:
  MatNxN *_X{nullptr};      // (sorted) given sample data (not changed)
  MatNxN *_X_l{nullptr};    // (sorted) given sample data for each level
  MatNxN *_X_apr{nullptr};  // (sorted) approximated sample
  VecNi *_frms{nullptr};    // (sorted) frame indexs
  std::vector<bool> _has_data_at_t;
  Mat4x4 _CubicBSplBasis;

  Imple(int const &D, int const &D_X, TimeSeriesMap const &time_series_map)
      : _has_data_at_t(D, false) {
    CubicBSpline(&_CubicBSplBasis);
    prepareSystem(D_X, time_series_map);
  }

  ~Imple() {
    if (_X) delete _X;
    if (_X_l) delete _X_l;
    if (_X_apr) delete _X_apr;
    if (_frms) delete _frms;
    _X_l = nullptr;
    _X_apr = nullptr;
    _frms = nullptr;
  }

  bool solveBSpline(int const &D, int const &D_X, int const &n_knots,
                    MatNxN *M) {
    MatNxN CPS;
    solveControlPoints(D, D_X, n_knots, &CPS);
    interpolate(D, D_X, n_knots, CPS, M);
    return true;
  }

  void updateNextLevelTargets() {
    MatNxN D = (*_X_l) - (*_X_apr);
    (*_X_l) = D;
  }

 private:
  void prepareSystem(int const &D_X, TimeSeriesMap const &time_series_map) {
    size_t N = time_series_map.size();
    _X = new MatNxN(N, D_X);
    _X_l = new MatNxN(N, D_X);
    _X_apr = new MatNxN(N, D_X);
    _frms = new VecNi(N);

    // make data matrix
    int i = 0;
    for (auto const &it : time_series_map) {
      _has_data_at_t[it.first] = true;
      (*_frms)(i) = it.first;
      _X->row(i++) = it.second.transpose();
    }
  }

  void solveControlPoints(int const &D, int const &D_X, int const &n_knots,
                          MatNxN *CPS) {
    // cubic B-spline need at least n_knots + 3 cps
    int n_cps = n_knots + 3;

    // set control points and its weights zero initially
    *CPS = MatNxN::Zero(n_cps, D_X);
    VecN W = VecN::Zero(n_cps);

    // variables
    int c_knot(0);
    int cp_id(0);
    float t_l(0.0f), sq_sum(0.0f);
    RVec4 basis, basis_sq;

    for (int i = 0, n = _frms->size(); i < n; ++i) {
      c_knot = localSplineDomain(D, n_knots, (*_frms)(i), &t_l);
      basis = computeBasis(t_l);
      basis_sq = basis.cwiseAbs2();
      sq_sum = basis_sq.sum();

      for (int j = 0; j < 4; ++j) {
        cp_id = c_knot + j;
        CPS->row(cp_id) += basis_sq(j) * _X_l->row(i) * basis(j) / sq_sum;
        W(cp_id) += basis_sq(j);
      }
    }

    for (int i = 0; i < n_cps; ++i) {
      if (W(i) > 1.0e-4f)
        CPS->row(i) /= W(i);
      else
        CPS->row(i).setZero();
    }
  }

  void interpolate(int const &D, int const &D_X, int const &n_knots,
                   MatNxN const &CPS, MatNxN *M) {
    for (int f = 0, i = 0; f < D; ++f) {
      RVecN res = evaluate(D, D_X, n_knots, f, CPS);
      M->row(f) = res;
      if (_has_data_at_t[f]) _X_apr->row(i++) = res;
    }
  }

  float frameToTime(int const &D, int const &n_knots, int const &f) {
    return static_cast<float>(n_knots * f) / D;
  }

  int localSplineDomain(int const &D, int const &n_knots, int const &f,
                        float *t_l) {
    float t = frameToTime(D, n_knots, f);
    int knot = static_cast<int>(t);
    knot = (knot < 0) ? 0 : ((knot > n_knots - 1) ? n_knots - 1 : knot);
    *t_l = t - knot;
    return knot;
  }

  RVec4 computeBasis(float const &t_l) {
    float t_sq = t_l * t_l;
    return (1.0f / 6.0f) * RVec4(t_sq * t_l, t_sq, t_l, 1.0f) * _CubicBSplBasis;
  }

  RVecN evaluate(int const &D, int const &D_X, int const &n_knots, int const &f,
                 MatNxN const &CPS) {
    float t_l;
    int knot = localSplineDomain(D, n_knots, f, &t_l);
    return evaluateLocal(t_l, CPS.block(knot, 0, 4, D_X));
  }

  template <typename Derived>
  RVecN evaluateLocal(float const &t_l,
                      Eigen::MatrixBase<Derived> const &local_pts) {
    return computeBasis(t_l) * local_pts;
  }
};

MultiLevelBSplineInterpolation::MultiLevelBSplineInterpolation(
    int const &D, TimeSeriesMap const &time_series_map)
    : Interpolation(D, time_series_map),
      _p(new Imple(D, dataDimension(), time_series_map)) {}

MultiLevelBSplineInterpolation::~MultiLevelBSplineInterpolation() {}

bool MultiLevelBSplineInterpolation::solve(int const &initial_n_knots,
                                           int const &level,
                                           MatNxN *result_mat) {
  std::cout << "start to solve multi-level bspline" << std::endl;
  bool result = false;
  int const &D = timeDimension();
  int const &D_X = dataDimension();
  *result_mat = MatNxN::Zero(D, D_X);
  *_p->_X_l = *_p->_X;  // deep copy;
  MatNxN M(D, D_X);
  for (int n_knots = initial_n_knots, l = 0; l < level; n_knots *= 2, ++l) {
    result |= _p->solveBSpline(D, D_X, n_knots, &M);
    _p->updateNextLevelTargets();
    *result_mat += M;
  }
  std::cout << "solve finished" << std::endl;
  return result;
}

}  // namespace ML
