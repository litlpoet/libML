// Copyright 2015 Byungkuk Choi.

#include "MLInterpolation/gaussianinterpolationnoisy.h"

#include <Eigen/Dense>
#include <vector>
#include <iostream>

#include "MLCore/mathmatrixpredefined.h"
#include "MLInterpolation/interpolationtypes.h"

namespace ML {

class GaussianInterpolationNoisy::Imple {
 public:
  G_NOISY_BOUNDARY_TYPE _boundary;  // boundary condition type
  bool _prior_dirty;                // dirty bit for prior computation
  size_t _N;                        // number of observed samples
  MatNxN* _Y;                       // given sample values
  SpMat* _A;                        // linear gaussian system matrix
  SpMat* _L;                        // temporal smoothness prior matrix
  SpMat _L_p;                       // prior mat multiplied by lambda

  Imple(const int& D, const int& D_X, const TimeSeriesMap& time_series_data)
      : _boundary(G_BOUND_NONE),
        _prior_dirty(true),
        _N(time_series_data.size()),
        _Y(nullptr),
        _A(nullptr),
        _L(nullptr) {
    prepareSystem(D, D_X, time_series_data);
  }

  ~Imple() {
    if (_Y) delete _Y;
    if (_A) delete _A;
    if (_L) delete _L;
    _Y = nullptr;
    _A = nullptr;
    _L = nullptr;
  }

  bool solveSigmaAndMu(const int& D, const int& D_X, const float& lambda,
                       MatNxN* Mu, MatNxN* Sigma) {
    preparePrior(D);
    multiplyLambdaToPrior(lambda);

    // MatNxN LTL = MatNxN::Constant(_D, _D, 1e-3);
    MatNxN LTL = SpMat(_L_p.transpose() * _L_p);
    MatNxN LTL_inv = LTL.llt().solve(MatNxN::Identity(D, D));

    SpMat AT = _A->transpose();
    MatNxN Sigma_y = MatNxN::Identity(_N, _N);
    MatNxN Sigma_y_inv = Sigma_y.inverse();
    MatNxN Sigma_rh = LTL + AT * Sigma_y_inv * (*_A);
    (*Sigma) = Sigma_rh.llt().solve(MatNxN::Identity(D, D));

    MatNxN LGS = (*Sigma) * AT * Sigma_y_inv;
    Mu->resize(D, D_X);
    for (int i = 0; i < D_X; ++i) Mu->col(i) = LGS * _Y->col(i);

    return true;
  }

 private:
  void prepareSystem(const int& D, const int& D_X,
                     const TimeSeriesMap& time_series_map) {
    int i = 0;
    _Y = new MatNxN(_N, D_X);
    std::vector<Trp> triples;
    for (const auto& it : time_series_map) {
      _Y->row(i) = it.second.transpose();
      triples.push_back(Trp(i++, it.first, 1));
    }
    _A = new SpMat(_N, D);
    _A->setFromTriplets(triples.begin(), triples.end());
    _L = new SpMat;
  }

  void preparePrior(const int& D) {
    if (!_prior_dirty) return;

    if (_boundary == G_BOUND_C1)
      MakeFiniteDifferenceMatWithBoundary(D, _L);
    else if (_boundary == G_BOUND_C2)
      MakeFiniteDifferenceMatWithC2Boundary(D, _L);
    else
      MakeFiniteDifferenceMat(D, _L);
    _prior_dirty = false;
  }

  void multiplyLambdaToPrior(const float& lambda) {
    _L_p = (*_L);
    if (_boundary == G_BOUND_C1)
      _L_p.block(1, 0, _L->rows() - 2, _L->cols()) *= lambda;
    else if (_boundary == G_BOUND_C2)
      _L_p.block(2, 0, _L->rows() - 4, _L->cols()) *= lambda;
    else
      _L_p *= lambda;
  }
};

GaussianInterpolationNoisy::GaussianInterpolationNoisy(
    const int& D, const TimeSeriesMap& time_series_data)
    : Interpolation(D, time_series_data),
      _p(new GaussianInterpolationNoisy::Imple(D, dataDimension(),
                                               time_series_data)) {}

GaussianInterpolationNoisy::~GaussianInterpolationNoisy() {}

bool GaussianInterpolationNoisy::solve(const float& lambda, MatNxN* Mu,
                                       MatNxN* Sigma) {
  MatNxN S;
  bool res =
      _p->solveSigmaAndMu(timeDimension(), dataDimension(), lambda, Mu, &S);
  if (Sigma) *Sigma = S;
  return res;
}

void GaussianInterpolationNoisy::setBoundaryConstraint(const int& b_type) {
  G_NOISY_BOUNDARY_TYPE b = (G_NOISY_BOUNDARY_TYPE)(b_type);
  if (_p->_boundary != b) _p->_prior_dirty = true;
  _p->_boundary = b;
}

}  // namespace ML
