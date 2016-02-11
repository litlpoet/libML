// Copyright (C) 2015 BK

#include "MLRegression/gaussianprocess.h"

#include <Eigen/Dense>
#include <vector>

#include "MLRegression/kernelfunction.h"
#include "MLRegression/kernelsquaredexponential.h"
#include "MLRegression/trainingdataset.h"

namespace ML {

class GPRegression::Imple {
 public:
  bool _is_alpha_dirty{false};
  int _n_dim_D{0};
  int _n_dim_X{0};
  int _n_dim_Y{0};
  MatNxN _L;
  VecN _k_star;
  std::vector<VecN> _alpha;
  TrainingDataSet* _data_set{nullptr};
  KernelFunction* _Kf{nullptr};

  Imple(int const& n_dim_D, int const& n_dim_X, int const& n_dim_Y,
        TimeSeriesMap const& time_series_map)
      : _n_dim_D(n_dim_D), _n_dim_X(n_dim_X), _n_dim_Y(n_dim_Y) {
    prepareSystem(time_series_map);
  }

  ~Imple() {
    if (_Kf) delete _Kf;
  }

  void addTrainingData(VecN const& x, VecN const& y) {
    int const& n = _data_set->size();
    _data_set->append(x, y);

    if (n == 0) {
      _L(0, 0) = sqrt(_Kf->cov(_data_set->x(0), _data_set->x(0)));
      _Kf->setParametersDirty(false);
    } else if (_Kf->isParametersDirty()) {
      compute();
    } else {
      VecN k(n);
      for (int i = 0; i < n; ++i)
        k(i) = _Kf->cov(_data_set->x(i), _data_set->x(n));
      Scalar kappa = _Kf->cov(_data_set->x(n), _data_set->x(n));
      if (_data_set->size() > _L.rows())
        _L.conservativeResize(n + _n_dim_D, n + _n_dim_D);
      _L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
      _L.block(n, 0, 1, n) = k.transpose();
      _L(n, n) = sqrt(kappa - k.dot(k));
    }
    _is_alpha_dirty = true;
  }

  VecN f(VecN const& x) {
    VecN y_star = VecN::Zero(_n_dim_Y);
    if (!_data_set || _data_set->isEmpty()) return y_star;
    compute();
    for (int d = 0; d < _n_dim_Y; ++d) updateAlpha(d);
    updateKStar(x);
    for (int d = 0; d < _n_dim_Y; ++d) y_star(d) = _k_star.dot(_alpha.at(d));
    return y_star;
  }

  void compute() {
    // can previously computed values be used?
    if (!_Kf->isParametersDirty()) return;
    _Kf->setParametersDirty(false);
    int const& n = _data_set->size();
    if (n > _L.rows()) _L.resize(n + _n_dim_D, n + _n_dim_D);
    // compute kernel matrix (lower triangle)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        _L(i, j) = _Kf->cov(_data_set->x(i), _data_set->x(j));
      }
    }
    // perform cholesky factorization
    _L.topLeftCorner(n, n) =
        _L.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
    _is_alpha_dirty = true;
  }

  void updateAlpha(int const& d) {
    // can previously computed values be used?
    if (!_is_alpha_dirty) return;
    if (d == _n_dim_Y - 1) _is_alpha_dirty = false;
    int const& n = _data_set->size();
    _alpha[d].resize(n);
    // Map target values to VectorXd
    std::vector<Scalar> const& targets = _data_set->Y(d);
    Eigen::Map<VecN const> Y(&targets[0], n);
    _alpha[d] = _L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(Y);
    _L.topLeftCorner(n, n)
        .triangularView<Eigen::Lower>()
        .adjoint()
        .solveInPlace(_alpha[d]);
  }

  void updateKStar(VecN const& x_star) {
    int const& n = _data_set->size();
    _k_star.resize(n);
    for (int i = 0; i < n; ++i) _k_star(i) = _Kf->cov(x_star, _data_set->x(i));
  }

 private:
  void prepareSystem(TimeSeriesMap const& time_series_map) {
    _Kf = new KernelSquaredExponential();
    _Kf->init(_n_dim_X);
    _Kf->setParametersDirty(false);
    _alpha.resize(_n_dim_Y);
    _data_set = new TrainingDataSet(_n_dim_Y);
    _L.resize(_n_dim_D, _n_dim_D);
    VecN x(1);
    for (auto const& it : time_series_map) {
      x << it.first;
      addTrainingData(x, it.second);
    }
  }
};

GPRegression::GPRegression(int const& n_dim_D,
                           TimeSeriesMap const& time_series_map)
    : Regression(n_dim_D, time_series_map),
      _p(new Imple(n_dim_D, xDimension(), yDimension(), time_series_map)) {}

GPRegression::~GPRegression() {}

bool GPRegression::solve(MatNxN* Mu, MatNxN* /*Sigma*/) {
  Mu->resize(_p->_n_dim_D, _p->_n_dim_Y);
  VecN x(1);
  for (int i = 0; i < _p->_n_dim_D; ++i) {
    x << i;
    Mu->row(i) = _p->f(x).transpose();
  }
  return true;
}

}  // namespace ML
