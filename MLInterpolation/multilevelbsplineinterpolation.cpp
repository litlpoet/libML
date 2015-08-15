// Copyright 2015 Byungkuk Choi.

#include "MLInterpolation/multilevelbsplineinterpolation.h"

#include <iostream>

#include "MLCore/splinebasis.h"

namespace ML {

class MultiLevelBSplineInterpolation::Imple {
 public:
  int _D;         // discrete time dimension
  int _D_X;       // sample dimension
  MatNxN *_X2;    // (sorted) given sample data
  MatNxN *_X2_p;  // (sorted) approximated sample
  VecNi *_frms;   // (sorted) frame indexs
  TimeSeriesMap _time_series_map;
  std::vector<bool> _has_data_at_t;
  Mat4x4 _CubicBSplBasis;

  Imple(const int &D, const TimeSeriesMap &time_series_map)
      : _D(D),
        _D_X(0),
        _X2(nullptr),
        _X2_p(nullptr),
        _frms(nullptr),
        _time_series_map(time_series_map),
        _has_data_at_t(D, false) {
    CubicBSpline(&_CubicBSplBasis);
    prepareSystem();
  }

  ~Imple() {
    if (_X2) delete _X2;
    _X2 = nullptr;
  }

  bool solveBSpline(const int &n_knots, MatNxN *M) {
    MatNxN CPS;
    solveControlPoints(n_knots, &CPS);
    interpolate(n_knots, CPS, M);
    return true;
  }

  void updateDisplacements() {
    MatNxN D = (*_X2) - (*_X2_p);
    (*_X2) = D;
  }

 private:
  void prepareSystem() {
    const size_t N = _time_series_map.size();
    _D_X = static_cast<int>(_time_series_map.begin()->second.size());

    _X2 = new MatNxN(N, _D_X);
    _X2_p = new MatNxN(N, _D_X);
    _frms = new VecNi(N);

    _X2_p->setZero();

    // make data matrix
    int i = 0;
    for (const auto &it : _time_series_map) {
      _has_data_at_t[it.first] = true;
      (*_frms)(i) = it.first;
      _X2->row(i++) = it.second.transpose();
    }
  }

  void solveControlPoints(const int &n_knots, MatNxN *CPS) {
    // cubic B-spline need at least n_knots + 3 cps
    const int n_cps = n_knots + 3;

    // set control points and its weights zero initially
    *CPS = MatNxN::Zero(n_cps, _D_X);
    VecN W = VecN::Zero(n_cps);

    // variables
    int c_knot;
    int cp_id;
    float t_l, sq_sum;
    RVec4 basis, basis_sq;

    for (int i = 0, n = _frms->size(); i < n; ++i) {
      c_knot = localSplineDomain(n_knots, (*_frms)(i), &t_l);
      basis = computeBasis(t_l);
      basis_sq = basis.cwiseAbs2();
      sq_sum = basis_sq.sum();

      for (int j = 0; j < 4; ++j) {
        cp_id = c_knot + j;
        CPS->row(cp_id) += basis_sq(j) * _X2->row(i) * basis(j) / sq_sum;
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

  void interpolate(const int &n_knots, const MatNxN &CPS, MatNxN *M) {
    for (int f = 0, i = 0; f < _D; ++f) {
      RVecN res = evaluate(n_knots, f, CPS);
      M->row(f) = res;
      if (_has_data_at_t[f]) _X2_p->row(i++) = res;
    }
  }

  float frameToTime(const int &n_knots, const int &f) {
    return static_cast<float>(n_knots * f) / _D;
  }

  int localSplineDomain(const int &n_knots, const int &f, float *t_l) {
    float t = frameToTime(n_knots, f);
    int knot = static_cast<int>(t);
    knot = (knot < 0) ? 0 : ((knot > n_knots - 1) ? n_knots - 1 : knot);
    *t_l = t - knot;
    return knot;
  }

  RVec4 computeBasis(const float &t_l) {
    float t_sq = t_l * t_l;
    return (1.0f / 6.0f) * RVec4(t_sq * t_l, t_sq, t_l, 1.0f) * _CubicBSplBasis;
  }

  RVecN evaluate(const int &n_knots, const int &f, const MatNxN &CPS) {
    float t_l;
    const int knot = localSplineDomain(n_knots, f, &t_l);
    return evaluateLocal(t_l, CPS.block(knot, 0, 4, _D_X));
  }

  template <typename Derived>
  RVecN evaluateLocal(const float &t_l,
                      const Eigen::MatrixBase<Derived> &local_pts) {
    return computeBasis(t_l) * local_pts;
  }
};

MultiLevelBSplineInterpolation::MultiLevelBSplineInterpolation(
    const int &D, const TimeSeriesMap &time_series_map)
    : _p(new MultiLevelBSplineInterpolation::Imple(D, time_series_map)) {}

MultiLevelBSplineInterpolation::~MultiLevelBSplineInterpolation() {}

bool MultiLevelBSplineInterpolation::solve(const int &initial_n_knots,
                                           const int &level,
                                           MatNxN *result_mat) {
  bool result = false;
  std::cout << "start to solve multi-level bspline" << std::endl;
  MatNxN M(_p->_D, _p->_D_X);
  for (int n_knots = initial_n_knots, l = 0; l < level; n_knots *= 2, ++l) {
    result |= _p->solveBSpline(n_knots, &M);
    _p->updateDisplacements();
  }
  *result_mat = M;
  std::cout << "solve finished" << std::endl;
  return result;
}

}  // namespace ML
