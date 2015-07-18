// Copyright 2015 Byungkuk Choi.

#include "MLGaussian/gaussianinterpolation.h"

#include <assert.h>
#include <algorithm>
#include <cassert>
#include <exception>
#include <vector>
#include <iostream>

#include "MLCore/mathmatrixpredefined.h"

namespace ML {

class BadInputException : public std::exception {
  virtual const char* what() const throw() {
    return "GaussianInterpolation | Bad input exception";
  }
} bad_input_ex;

class GaussianInterpolation::Imple {
 public:
  int _D;       // discrete time dimension
  int _D_X;     // sample dimension
  float _T;     // time in second
  MatNxN* _X2;  // (sorted) given sample data
  SpMat* _L1;   // prior for unknowns
  SpMat* _L2;   // prior for knowns
  TimeSeries _time_series_data;
  TimeSeries _sorted_time_series;
  std::vector<bool> _has_data_at_t;

  Imple(const int& D, const float& T, const TimeSeries& time_series_data)
      : _D(D),
        _D_X(0),
        _T(T),
        _X2(nullptr),
        _L1(nullptr),
        _L2(nullptr),
        _time_series_data(time_series_data),
        _sorted_time_series(time_series_data),
        _has_data_at_t(D, false) {
    checkInputValidity();
    prepareSystem();
  }

  ~Imple() {
    if (_X2) delete _X2;
    if (_L1) delete _L1;
    if (_L2) delete _L2;
    _X2 = nullptr;
    _L1 = nullptr;
    _L2 = nullptr;
  }

  bool solveMean(MatNxN* Mu) {
    SparseQR sparse_qr(*_L1);

    MatNxN mean(_L1->cols(), _D_X);

    for (int i = 0; i < _D_X; ++i)
      mean.col(i) = sparse_qr.solve(-1.0f * (*_L2) * _X2->col(i));
    Mu->resize(_D, mean.cols());

    for (int i = 0, j = 0, k = 0; i < _D; ++i)
      if (_has_data_at_t[i])
        Mu->row(i) = _sorted_time_series.at(k++).second.transpose();
      else
        Mu->row(i) = mean.row(j++);

    return true;
  }

  bool solveVariance(const float& lambda, MatNxN* Sigma) {
    SpMat L1TL1 = _L1->transpose() * (*_L1) * std::pow(1.0f / lambda, 2);
    SpMat I(L1TL1.rows(), L1TL1.cols());
    I.setIdentity();

    SimplicalCholSpMat chol(L1TL1);
    SpMat L1TL1_inv = chol.solve(I);
    MatNxN var = L1TL1_inv.diagonal().cwiseSqrt();

    Sigma->resize(_D, var.cols());

    for (int i = 0, j = 0; i < _D; ++i) {
      if (_has_data_at_t[i])
        Sigma->row(i).setZero();
      else
        Sigma->row(i) = var.row(j++);
    }

    return true;
  }

 private:
  void checkInputValidity() {
    if (_D < 2) throw bad_input_ex;

    for (const auto& it : _time_series_data)
      if (it.first < 0 || it.first > _D - 1) throw bad_input_ex;
  }

  void prepareSystem() {
    const size_t N = _time_series_data.size();
    _D_X = static_cast<int>(_time_series_data.front().second.rows());
    //    _has_data_at_t.resize(_D, false);
    for (const auto& it : _time_series_data) _has_data_at_t[it.first] = true;

    _sorted_time_series = _time_series_data;
    std::sort(_sorted_time_series.begin(), _sorted_time_series.end(),
              [](const T_Sample& d1, const T_Sample& d2)
                  -> bool { return d1.first < d2.first; });

    SpMat L;
    // makeLMatrix(_D, &L);
    MakeFiniteDiffernceMat(_D, &L);
    _X2 = new MatNxN(N, _D_X);

    // make permutation vector and arrange data accordingly
    VecNi pm_vec(_D);
    int j = 0;

    // first, find domain without data samples
    for (int i = 0; i < _D; ++i)
      if (!_has_data_at_t[i]) pm_vec[j++] = i;

    // second, find domain with data samples
    for (size_t i = 0, n = _sorted_time_series.size(); i < n; ++i) {
      pm_vec[j++] = _sorted_time_series[i].first;
      _X2->row(i) = _sorted_time_series[i].second.transpose();
    }

    assert(j == _D);

    // permute matrix L using permutation matrix (pm_mat)
    PmMat pm_mat(pm_vec);
    SpMat Lp = L * pm_mat;

    const int n_knowns = static_cast<int>(N);
    _L1 = new SpMat(Lp.topLeftCorner(Lp.rows(), Lp.cols() - n_knowns));
    _L2 = new SpMat(Lp.topRightCorner(Lp.rows(), n_knowns));
    _L1->makeCompressed();
  }

  // void makeLMatrix(const int& dim, SpMat* L) {
  //   std::vector<Trp> triples;
  //   triples.reserve(3 * (dim - 2));

  //   for (auto i = 0; i < dim - 2; ++i) {
  //     triples.push_back(Trp(i, i, -1.f));
  //     triples.push_back(Trp(i, i + 1, 2.f));
  //     triples.push_back(Trp(i, i + 2, -1.f));
  //   }

  //   L->resize(dim - 2, dim);
  //   L->setFromTriplets(triples.begin(), triples.end());
  // }
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
GaussianInterpolation::GaussianInterpolation(const int& D, const float& T,
                                             const TimeSeries& time_series_data)
    : _p(new GaussianInterpolation::Imple(D, T, time_series_data)) {}

GaussianInterpolation::~GaussianInterpolation() {}

bool GaussianInterpolation::solve(const float& lambda, MatNxN* Mu,
                                  MatNxN* Sigma) {
  bool result = false;
  result |= _p->solveMean(Mu);

  if (Sigma) result |= _p->solveVariance(lambda, Sigma);

  return result;
}

}  // namespace ML
