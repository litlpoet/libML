// Copyright 2015 Byungkuk Choi.

#include "MLGaussian/gaussianinterpolationnoisy.h"

#include <Eigen/Dense>
#include <vector>
#include <iostream>

#include "MLCore/mathmatrixpredefined.h"
#include "MLCore/exceptions.h"

namespace ML {

class GaussianInterpolationNoisy::Imple {
 public:
  int _D;      // discrete time dimension
  int _D_X;    // sample dimension
  size_t _N;   // number of observed samples
  MatNxN* _Y;  // given sample values
  SpMat* _A;   // linear gaussian system matrix
  SpMat* _L;   // temporal smoothness prior matrix
  TimeSeriesMap _time_series_map;

  Imple(const int& D, const TimeSeriesMap& time_series_data)
      : _D(D),
        _D_X(0),
        _N(time_series_data.size()),
        _Y(nullptr),
        _A(nullptr),
        _L(nullptr),
        _time_series_map(time_series_data) {
    checkInputValidity();
    prepareSystem();
  }

  ~Imple() {
    if (_Y) delete _Y;
    if (_A) delete _A;
    if (_L) delete _L;
    _Y = nullptr;
    _A = nullptr;
    _L = nullptr;
  }

  bool solveSigmaAndMu(const float& lambda, MatNxN* Mu, MatNxN* Sigma) {
    (*_L) *= lambda;

    MatNxN LTL = MatNxN::Constant(_D, _D, 1e-3);
    LTL += SpMat(_L->transpose() * (*_L));
    MatNxN LTL_inv = LTL.llt().solve(MatNxN::Identity(_D, _D));

    SpMat AT = _A->transpose();
    MatNxN Sigma_y = MatNxN::Identity(_N, _N);
    MatNxN Sigma_y_inv = Sigma_y.inverse();
    MatNxN Sigma_rh = LTL + AT * Sigma_y_inv * (*_A);
    (*Sigma) = Sigma_rh.llt().solve(MatNxN::Identity(_D, _D));

    MatNxN LGS = (*Sigma) * AT * Sigma_y;
    Mu->resize(_D, _D_X);
    for (int i = 0; i < _D_X; ++i) Mu->col(i) = LGS * _Y->col(i);

    return true;
  }

 private:
  void checkInputValidity() {
    BadInputException bad_input_ex(
        "GaussianInterpolationNoisy | Bad input exception");

    if (_D < 2) throw bad_input_ex;

    for (const auto& it : _time_series_map)
      if (it.first < 0 || it.first > _D - 1) throw bad_input_ex;
  }

  void prepareSystem() {
    _D_X = static_cast<int>(_time_series_map.begin()->second.size());

    int i = 0;
    _Y = new MatNxN(_N, _D_X);
    std::vector<Trp> triples;
    for (const auto& it : _time_series_map) {
      _Y->row(i) = it.second.transpose();
      triples.push_back(Trp(i++, it.first, 1));
    }
    _A = new SpMat(_N, _D);
    _A->setFromTriplets(triples.begin(), triples.end());

    _L = new SpMat;
    MakeFiniteDiffernceMat(_D, _L);
  }
};

GaussianInterpolationNoisy::GaussianInterpolationNoisy(
    const int& D, const TimeSeriesMap& time_series_data)
    : _p(new GaussianInterpolationNoisy::Imple(D, time_series_data)) {}

GaussianInterpolationNoisy::~GaussianInterpolationNoisy() {}

bool GaussianInterpolationNoisy::solve(const float& lambda, MatNxN* Mu,
                                       MatNxN* Sigma) {
  bool result = false;
  result |= _p->solveSigmaAndMu(lambda, Mu, Sigma);
  return result;
}

}  // namespace ML
