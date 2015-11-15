// Copyright 2015 Byungkuk Choi.

#include "MLInterpolation/gaussianinterpolation.h"

#include <algorithm>
#include <cassert>
#include <vector>
#include <iostream>

#include "MLCore/mathmatrixpredefined.h"

namespace ML {

class GaussianInterpolation::Imple {
 public:
  MatNxN* _X2;  // (sorted) given sample data
  SpMat* _L1;   // prior for unknowns
  SpMat* _L2;   // prior for knowns
  std::vector<bool> _has_data_at_t;

  Imple(const int& D, const int& D_X, const TimeSeriesMap& time_series_map)
      : _X2(nullptr), _L1(nullptr), _L2(nullptr), _has_data_at_t(D, false) {
    prepareSystem(D, D_X, time_series_map);
  }

  ~Imple() {
    if (_X2) delete _X2;
    if (_L1) delete _L1;
    if (_L2) delete _L2;
    _X2 = nullptr;
    _L1 = nullptr;
    _L2 = nullptr;
  }

  bool solveMean(const int& D, const int& D_X,
                 const TimeSeriesMap& time_series_map, MatNxN* Mu) {
    SparseQR sparse_qr(*_L1);

    MatNxN mean(_L1->cols(), D_X);

    for (int i = 0; i < D_X; ++i)
      mean.col(i) = sparse_qr.solve(-1.0f * (*_L2) * _X2->col(i));
    Mu->resize(D, mean.cols());

    for (int i = 0, j = 0; i < D; ++i)
      if (_has_data_at_t[i])
        Mu->row(i) = time_series_map.at(i).transpose();
      else
        Mu->row(i) = mean.row(j++);

    return true;
  }

  bool solveVariance(const int& D, const float& lambda, MatNxN* Sigma) {
    SpMat L1TL1 = _L1->transpose() * (*_L1) * std::pow(1.0f / lambda, 2);
    SpMat I(L1TL1.rows(), L1TL1.cols());
    I.setIdentity();

    SimplicalCholSpMat chol(L1TL1);
    SpMat L1TL1_inv = chol.solve(I);
    MatNxN var = L1TL1_inv.diagonal().cwiseSqrt();

    Sigma->resize(D, var.cols());

    for (int i = 0, j = 0; i < D; ++i) {
      if (_has_data_at_t[i])
        Sigma->row(i).setZero();
      else
        Sigma->row(i) = var.row(j++);
    }

    return true;
  }

 private:
  void prepareSystem(const int& D, const int& D_X,
                     const TimeSeriesMap& time_series_map) {
    const size_t N = time_series_map.size();

    for (const auto& it : time_series_map) _has_data_at_t[it.first] = true;

    SpMat L;
    MakeFiniteDifferenceMat(D, &L);
    _X2 = new MatNxN(N, D_X);

    // make permutation vector and arrange data accordingly
    VecNi pm_vec(D);
    int j = 0;

    // first, find domain without data samples
    for (int i = 0; i < D; ++i)
      if (!_has_data_at_t[i]) pm_vec[j++] = i;

    // second, find domain with data samples
    int i = 0;
    for (const auto& it : time_series_map) {
      pm_vec[j++] = it.first;
      _X2->row(i++) = it.second.transpose();
    }

    assert(j == D);

    // permute matrix L using permutation matrix (pm_mat)
    PmMat pm_mat(pm_vec);
    SpMat Lp = L * pm_mat;

    _L1 = new SpMat(Lp.topLeftCorner(Lp.rows(), Lp.cols() - N));
    _L2 = new SpMat(Lp.topRightCorner(Lp.rows(), N));
    _L1->makeCompressed();
    _L2->makeCompressed();
  }
};

/**
 * @brief GaussianInterpolation : Interpolating time-series samples using
 * Gaussian
 * @param D : the total number of discrete time samples
 * @param T : the time of interpolation region. interpolation will be done
 * on [0, T] region
 * @param lambda : precision
 * @param time_series_data : vector of pair<int, VecN> data
 */
GaussianInterpolation::GaussianInterpolation(
    const int& D, const TimeSeriesMap& time_series_data)
    : Interpolation(D, time_series_data),
      _p(new GaussianInterpolation::Imple(D, dataDimension(),
                                          time_series_data)) {}

GaussianInterpolation::~GaussianInterpolation() {}

bool GaussianInterpolation::solve(const float& lambda, MatNxN* Mu,
                                  MatNxN* Sigma) {
  bool res =
      _p->solveMean(timeDimension(), dataDimension(), timeSeriesMap(), Mu);
  if (Sigma) res |= _p->solveVariance(timeDimension(), lambda, Sigma);
  return res;
}

}  // namespace ML