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
  bool _boundary;     // boundary condition toggle
  bool _prior_dirty;  // dirty bit for prior computation
  int _D;             // discrete time dimension
  int _D_X;           // sample dimension
  size_t _N;          // number of observed samples
  MatNxN* _Y;         // given sample values
  SpMat* _A;          // linear gaussian system matrix
  SpMat* _L;          // temporal smoothness prior matrix
  SpMat _L_p;         // prior mat multiplied by lambda
  TimeSeriesMap _time_series_map;

  Imple(const int& D, const TimeSeriesMap& time_series_data)
      : _boundary(false),
        _prior_dirty(true),
        _D(D),
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
    preparePrior();
    multiplyLambdaToPrior(lambda);

    // MatNxN LTL = MatNxN::Constant(_D, _D, 1e-3);
    MatNxN LTL = SpMat(_L_p.transpose() * _L_p);
    MatNxN LTL_inv = LTL.llt().solve(MatNxN::Identity(_D, _D));

    SpMat AT = _A->transpose();
    MatNxN Sigma_y = MatNxN::Identity(_N, _N);
    MatNxN Sigma_y_inv = Sigma_y.inverse();
    MatNxN Sigma_rh = LTL + AT * Sigma_y_inv * (*_A);
    (*Sigma) = Sigma_rh.llt().solve(MatNxN::Identity(_D, _D));

    MatNxN LGS = (*Sigma) * AT * Sigma_y_inv;
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
  }

  void preparePrior() {
    if (!_prior_dirty) return;

    if (_boundary)
      MakeFiniteDiffereceMatWithBoundary(_D, _L);
    else
      MakeFiniteDiffernceMat(_D, _L);
    _prior_dirty = false;
  }

  void multiplyLambdaToPrior(const float& lambda) {
    _L_p = (*_L);
    if (_boundary)
      _L_p.block(1, 1, _L->rows() - 2, _L->cols() - 2) *= lambda;
    else
      _L_p *= lambda;
  }
};

GaussianInterpolationNoisy::GaussianInterpolationNoisy(
    const int& D, const TimeSeriesMap& time_series_data)
    : _p(new GaussianInterpolationNoisy::Imple(D, time_series_data)) {}

GaussianInterpolationNoisy::~GaussianInterpolationNoisy() {}

int GaussianInterpolationNoisy::dimension() { return _p->_D; }

int GaussianInterpolationNoisy::sampleDimension() { return _p->_D_X; }

bool GaussianInterpolationNoisy::solve(const float& lambda, MatNxN* Mu,
                                       MatNxN* Sigma) {
  bool result = false;
  result |= _p->solveSigmaAndMu(lambda, Mu, Sigma);
  return result;
}

void GaussianInterpolationNoisy::setBoundaryConstraint(const bool& b) {
  if (_p->_boundary != b) _p->_prior_dirty = true;
  _p->_boundary = b;
}
const TimeSeriesMap& GaussianInterpolationNoisy::timeSeriesMap() const {
  return _p->_time_series_map;
}

}  // namespace ML
