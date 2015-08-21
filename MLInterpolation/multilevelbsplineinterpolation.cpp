// Copyright 2015 Byungkuk Choi.

#include "MLInterpolation/multilevelbsplineinterpolation.h"

#include <iostream>
#include <vector>

#include "MLCore/splinebasis.h"

namespace ML {

class MultiLevelBSplineInterpolation::Imple {
 public:
  MatNxN *_X;      // (sorted) given sample data (not changed)
  MatNxN *_X_l;    // (sorted) given sample data for each level
  MatNxN *_X_apr;  // (sorted) approximated sample
  VecNi *_frms;    // (sorted) frame indexs
  std::vector<bool> _has_data_at_t;
  Mat4x4 _CubicBSplBasis;

  Imple(const int &D, const int &D_X, const TimeSeriesMap &time_series_map)
      : _X(nullptr),
        _X_l(nullptr),
        _X_apr(nullptr),
        _frms(nullptr),
        _has_data_at_t(D, false) {
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

  bool solveBSpline(const int &D, const int &D_X, const int &n_knots,
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
  void prepareSystem(const int &D_X, const TimeSeriesMap &time_series_map) {
    const size_t N = time_series_map.size();

    _X = new MatNxN(N, D_X);
    _X_l = new MatNxN(N, D_X);
    _X_apr = new MatNxN(N, D_X);
    _frms = new VecNi(N);

    // make data matrix
    int i = 0;
    for (const auto &it : time_series_map) {
      _has_data_at_t[it.first] = true;
      (*_frms)(i) = it.first;
      _X->row(i++) = it.second.transpose();
    }
  }

  void solveControlPoints(const int &D, const int &D_X, const int &n_knots,
                          MatNxN *CPS) {
    // cubic B-spline need at least n_knots + 3 cps
    const int n_cps = n_knots + 3;

    // set control points and its weights zero initially
    *CPS = MatNxN::Zero(n_cps, D_X);
    VecN W = VecN::Zero(n_cps);

    // variables
    int c_knot;
    int cp_id;
    float t_l, sq_sum;
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

  void interpolate(const int &D, const int &D_X, const int &n_knots,
                   const MatNxN &CPS, MatNxN *M) {
    for (int f = 0, i = 0; f < D; ++f) {
      RVecN res = evaluate(D, D_X, n_knots, f, CPS);
      M->row(f) = res;
      if (_has_data_at_t[f]) _X_apr->row(i++) = res;
    }
  }

  float frameToTime(const int &D, const int &n_knots, const int &f) {
    return static_cast<float>(n_knots * f) / D;
  }

  int localSplineDomain(const int &D, const int &n_knots, const int &f,
                        float *t_l) {
    float t = frameToTime(D, n_knots, f);
    int knot = static_cast<int>(t);
    knot = (knot < 0) ? 0 : ((knot > n_knots - 1) ? n_knots - 1 : knot);
    *t_l = t - knot;
    return knot;
  }

  RVec4 computeBasis(const float &t_l) {
    float t_sq = t_l * t_l;
    return (1.0f / 6.0f) * RVec4(t_sq * t_l, t_sq, t_l, 1.0f) * _CubicBSplBasis;
  }

  RVecN evaluate(const int &D, const int &D_X, const int &n_knots, const int &f,
                 const MatNxN &CPS) {
    float t_l;
    const int knot = localSplineDomain(D, n_knots, f, &t_l);
    return evaluateLocal(t_l, CPS.block(knot, 0, 4, D_X));
  }

  template <typename Derived>
  RVecN evaluateLocal(const float &t_l,
                      const Eigen::MatrixBase<Derived> &local_pts) {
    return computeBasis(t_l) * local_pts;
  }
};

MultiLevelBSplineInterpolation::MultiLevelBSplineInterpolation(
    const int &D, const TimeSeriesMap &time_series_map)
    : Interpolation(D, time_series_map),
      _p(new MultiLevelBSplineInterpolation::Imple(D, dataDimension(),
                                                   time_series_map)) {}

MultiLevelBSplineInterpolation::~MultiLevelBSplineInterpolation() {}

bool MultiLevelBSplineInterpolation::solve(const int &initial_n_knots,
                                           const int &level,
                                           MatNxN *result_mat) {
  std::cout << "start to solve multi-level bspline" << std::endl;
  bool result = false;
  const int &D = timeDimension();
  const int &D_X = dataDimension();
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
