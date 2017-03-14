// Copyright (C) 2015 BK

#include "regression/gaussianprocess.h"

#include <Eigen/Dense>
#include <vector>
#include <iostream>

#include "regression/kernelfunction.h"
#include "regression/kernelsquaredexponential.h"
#include "regression/trainingdataset.h"

namespace ML {

const Scalar log2pi = log(2 * M_PI);

class GPRegression::Imple {
 public:
  int _n_dim_D{0};
  int _n_dim_X{0};
  int _n_dim_Y{0};
  std::vector<VecN> _k_stars;
  std::vector<bool> _is_alpha_dirty;
  std::vector<VecN> _alphas;
  std::vector<MatNxN> _Ls;
  std::vector<KernelFunction*> _Kfs;
  TrainingDataSet* _data_set{nullptr};

  explicit Imple(int const& n_dim_D)
      : _n_dim_D(n_dim_D), _data_set(new TrainingDataSet(n_dim_D)) {}

  ~Imple() {
    clearKernels();
    if (_data_set) delete _data_set;
  }

  void clearKernels() {
    for (size_t i = 0, n = _Kfs.size(); i < n; ++i)
      if (_Kfs[i]) delete _Kfs[i];
    _Kfs.clear();
  }

  void initKernels() {
    _Kfs.resize(_n_dim_Y);
    _k_stars.resize(_n_dim_Y);
    for (int d = 0; d < _n_dim_Y; ++d) {
      _Kfs[d] = new KernelSquaredExponential;
      _Kfs[d]->init(_n_dim_X);
    }
  }

  void initKernelMatrices() {
    _Ls.resize(_n_dim_Y);
    for (int d = 0; d < _n_dim_Y; ++d) _Ls[d].resize(_n_dim_D, _n_dim_D);
  }

  void initAlphaVectors() {
    _alphas.resize(_n_dim_Y);
    _is_alpha_dirty.resize(_n_dim_Y);
    for (int d = 0; d < _n_dim_Y; ++d) _is_alpha_dirty[d] = false;
  }

  void prepareSystem(int const& n_dim_X, int const& n_dim_Y,
                     TimeSeriesMap const& time_series_map) {
    _n_dim_X = n_dim_X;
    _n_dim_Y = n_dim_Y;
    clearKernels();
    initKernels();
    initKernelMatrices();
    initAlphaVectors();
    VecN x(_n_dim_X);
    for (auto const& it : time_series_map) {
      x << it.first;
      addTrainingData(x, it.second);
    }
  }

  void addTrainingData(VecN const& x, VecN const& y) {
    int const n = _data_set->size();  // no reference here
    _data_set->append(x, y);
    if (n == 0) {
      for (int d = 0; d < _n_dim_Y; ++d) {
        _Ls[d](0, 0) = sqrt(_Kfs[d]->cov(_data_set->x(0), _data_set->x(0)));
        _Kfs[d]->setParametersDirty(false);
      }
    }  // else if (_Kf->isParametersDirty()) {
    //   compute();
    // }
    else {
      VecN k(n);
      for (int d = 0; d < _n_dim_Y; ++d) {
        MatNxN& L = _Ls[d];
        KernelFunction* const& Kf = _Kfs[d];
        for (int i = 0; i < n; ++i)
          k(i) = Kf->cov(_data_set->x(i), _data_set->x(n));
        Scalar kappa = Kf->cov(_data_set->x(n), _data_set->x(n));
        if (_data_set->size() > L.rows())
          L.conservativeResize(n + _n_dim_D, n + _n_dim_D);
        L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
        L.block(n, 0, 1, n) = k.transpose();
        L(n, n) = sqrt(kappa - k.dot(k));
      }
    }
    for (int d = 0; d < _n_dim_Y; ++d) _is_alpha_dirty[d] = true;
  }

  VecN f(VecN const& x) {
    VecN y_star = VecN::Zero(_n_dim_Y);
    if (!_data_set || _data_set->isEmpty()) return y_star;
    for (int d = 0; d < _n_dim_Y; ++d) {
      compute(d);
      updateAlpha(d);
      updateKStar(d, x);
    }
    for (int d = 0; d < _n_dim_Y; ++d)
      y_star(d) = _k_stars.at(d).dot(_alphas.at(d));
    return y_star;
  }

  void compute(int const& d) {
    // can previously computed values be used?
    if (!_Kfs[d]->isParametersDirty()) return;
    _Kfs[d]->setParametersDirty(false);
    int const& n = _data_set->size();
    if (n > _Ls[d].rows()) _Ls[d].resize(n + _n_dim_D, n + _n_dim_D);
    // compute kernel matrix (lower triangle)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        _Ls[d](i, j) = _Kfs[d]->cov(_data_set->x(i), _data_set->x(j));
      }
    }
    // perform cholesky factorization
    _Ls[d].topLeftCorner(n, n) = _Ls[d]
                                     .topLeftCorner(n, n)
                                     .selfadjointView<Eigen::Lower>()
                                     .llt()
                                     .matrixL();
    _is_alpha_dirty[d] = true;
  }

  void updateAlpha(int const& d) {
    // can previously computed values be used?
    if (!_is_alpha_dirty.at(d)) return;
    _is_alpha_dirty[d] = false;
    int const& n = _data_set->size();
    _alphas[d].resize(n);
    // Map target values to VectorXd
    std::vector<Scalar> const& targets = _data_set->Y(d);
    Eigen::Map<VecN const> Y(&targets[0], n);
    _alphas[d] =
        _Ls[d].topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(Y);
    _Ls[d]
        .topLeftCorner(n, n)
        .triangularView<Eigen::Lower>()
        .adjoint()
        .solveInPlace(_alphas[d]);
  }

  void updateKStar(int const& d, VecN const& x_star) {
    int const& n = _data_set->size();
    _k_stars[d].resize(n);
    for (int i = 0; i < n; ++i)
      _k_stars[d](i) = _Kfs[d]->cov(x_star, _data_set->x(i));
  }
};

GPRegression::GPRegression(int const& n_dim_D)
    : Regression(n_dim_D), _p(new Imple(n_dim_D)) {}

GPRegression::GPRegression(int const& n_dim_D,
                           TimeSeriesMap const& time_series_map)
    : Regression(n_dim_D, time_series_map), _p(new Imple(n_dim_D)) {
  _p->prepareSystem(xDimension(), yDimension(), time_series_map);
}

GPRegression::~GPRegression() {}

void GPRegression::setInitialTrainingData(
    TimeSeriesMap const& time_series_map) {
  setDimensions(time_series_map);
  _p->prepareSystem(xDimension(), yDimension(), time_series_map);
}

void GPRegression::addTrainingData(VecN const& x, VecN const& y) {
  _p->addTrainingData(x, y);
}

void GPRegression::clearTrainingData() { _p->_data_set->clear(); }

bool GPRegression::solve(MatNxN* Mu, MatNxN* /*Sigma*/) {
  Mu->resize(_p->_n_dim_D, _p->_n_dim_Y);
  VecN x(1);
  for (int i = 0; i < _p->_n_dim_D; ++i) {
    x << i;
    Mu->row(i) = _p->f(x).transpose();
  }
  return true;
}

KernelFunction* GPRegression::kernelFunction(int const& d_Y) {
  return _p->_Kfs[d_Y];
}

Scalar GPRegression::logLikelihood(int const& d_Y) {
  _p->compute(d_Y);
  _p->updateAlpha(d_Y);
  int const& n = _p->_data_set->size();
  std::vector<Scalar> const& targets = _p->_data_set->Y(d_Y);
  Eigen::Map<VecN const> y(&targets[0], n);
  double det = 2 * _p->_Ls[d_Y].diagonal().head(n).array().log().sum();
  return -0.5 * y.dot(_p->_alphas.at(d_Y)) - 0.5 * det - 0.5 * n * log2pi;
}

VecN GPRegression::logLikelihoodGrad(int const& d_Y) {
  _p->compute(d_Y);
  _p->updateAlpha(d_Y);
  int const& n = _p->_data_set->size();
  VecN grad = VecN::Zero(_p->_Kfs[d_Y]->parameterDimension());
  VecN g(grad.size());
  MatNxN W = MatNxN::Identity(n, n);

  // compute kernel matrix inverse
  VecN& alpha = _p->_alphas[d_Y];
  MatNxN& L = _p->_Ls[d_Y];
  L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
  L.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(
      W);
  W = alpha * alpha.transpose() - W;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      _p->_Kfs[d_Y]->grad(_p->_data_set->x(i), _p->_data_set->x(j), &g);
      if (i == j)
        grad += W(i, j) * g * 0.5;
      else
        grad += W(i, j) * g;
    }
  }
  return grad;
}

}  // namespace ML
