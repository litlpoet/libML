// Copyright 2015 Byungkuk Choi.

#include "MLInterpolation/gaussianinterpolationnoisy.h"

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "MLCore/mathmatrixpredefined.h"
#include "MLInterpolation/interpolationtypes.h"

namespace ML {

class GaussianInterpolationNoisy::Imple {
 public:
  BoundaryType _boundary{BoundaryType::None};  // boundary condition type
  bool _prior_dirty{true};  // dirty bit for prior computation
  size_t _N{0};             // number of observed samples
  MatNxN* _Y{nullptr};      // given sample values
  SpMat* _A{nullptr};       // linear gaussian system matrix
  SpMat* _L{nullptr};       // temporal smoothness prior matrix
  SpMat _L_p;               // prior mat multiplied by lambda

  Imple(int const& D, int const& D_X, TimeSeriesMap const& time_series_data)
      : _N(time_series_data.size()) {
    prepareSystem(D, D_X, time_series_data);
  }

  Imple(int const& D, int const& D_X, TimeSeriesDense const& time_series_dense)
      : _N(D) {
    prepareSystem(D, D_X, time_series_dense);
  }

  ~Imple() {
    if (_Y) delete _Y;
    if (_A) delete _A;
    if (_L) delete _L;
    _Y = nullptr;
    _A = nullptr;
    _L = nullptr;
  }

  bool solveSigmaAndMu(int const& D, int const& D_X, float const& lambda,
                       float const& alpha, MatNxN* Mu, MatNxN* Sigma) {
    std::cout << "time dim: " << D << std::endl;
    std::cout << "data dim: " << D_X << std::endl;
    preparePrior(D);
    multiplyLambdaToPrior(lambda);

    // MatNxN LTL = MatNxN::Constant(_D, _D, 1e-3);
    MatNxN LTL = SpMat(_L_p.transpose() * _L_p);
    MatNxN LTL_inv = LTL.llt().solve(MatNxN::Identity(D, D));
    std::cout << "LTL_inv computed" << std::endl;

    SpMat AT = _A->transpose();
    MatNxN Sigma_y = alpha * MatNxN::Identity(_N, _N);
    MatNxN Sigma_y_inv = Sigma_y.inverse();
    MatNxN Sigma_rh = LTL + AT * Sigma_y_inv * (*_A);
    (*Sigma) = Sigma_rh.llt().solve(MatNxN::Identity(D, D));
    std::cout << "SIGMA computed" << std::endl;

    MatNxN LGS = (*Sigma) * AT * Sigma_y_inv;
    Mu->resize(D, D_X);
    for (int i = 0; i < D_X; ++i) Mu->col(i) = LGS * _Y->col(i);
    std::cout << "Mean computed" << std::endl;
    return true;
  }

 private:
  void prepareSystem(int const& D, int const& D_X,
                     TimeSeriesMap const& time_series_map) {
    int i = 0;
    _Y = new MatNxN(_N, D_X);
    std::vector<Trp> triples;
    for (auto const& it : time_series_map) {
      _Y->row(i) = it.second.transpose();
      triples.push_back(Trp(i++, it.first, 1));
    }
    _A = new SpMat(_N, D);
    _A->setFromTriplets(triples.begin(), triples.end());
    _L = new SpMat;
  }

  void prepareSystem(int const& D, int const& D_X,
                     TimeSeriesDense const& time_series_dense) {
    int i = 0;
    _Y = new MatNxN(_N, D_X);
    std::vector<Trp> triples;
    for (auto const& it : time_series_dense) {
      _Y->row(i) = it.transpose();
      triples.push_back(Trp(i, i++, 1));
    }
    _A = new SpMat(_N, D);
    _A->setFromTriplets(triples.begin(), triples.end());
    _L = new SpMat;
  }

  void preparePrior(int const& D) {
    if (!_prior_dirty) return;

    if (_boundary == BoundaryType::C1)
      MakeFiniteDifferenceMatWithBoundary(D, _L);
    else if (_boundary == BoundaryType::C2)
      MakeFiniteDifferenceMatWithC2Boundary(D, _L);
    else
      MakeFiniteDifferenceMat(D, _L);
    _prior_dirty = false;
  }

  void multiplyLambdaToPrior(float const& lambda) {
    _L_p = (*_L);
    if (_boundary == BoundaryType::C1)
      _L_p.block(1, 0, _L->rows() - 2, _L->cols()) *= lambda;
    else if (_boundary == BoundaryType::C2)
      _L_p.block(2, 0, _L->rows() - 4, _L->cols()) *= lambda;
    else
      _L_p *= lambda;
  }
};

GaussianInterpolationNoisy::GaussianInterpolationNoisy(
    int const& D, TimeSeriesMap const& time_series_data)
    : Interpolation(D, time_series_data),
      _p(new Imple(D, dataDimension(), time_series_data)) {}

GaussianInterpolationNoisy::GaussianInterpolationNoisy(
    TimeSeriesDense const& time_series_dense)
    : Interpolation(time_series_dense),
      _p(new Imple(timeDimension(), dataDimension(), time_series_dense)) {
  std::cout << "dense gpn" << std::endl;
  std::cout << "time dim:" << timeDimension() << std::endl;
  std::cout << "data dim:" << dataDimension() << std::endl;
}

GaussianInterpolationNoisy::~GaussianInterpolationNoisy() {}

bool GaussianInterpolationNoisy::solve(float const& lambda, float const& alpha,
                                       MatNxN* Mu, MatNxN* Sigma) {
  MatNxN S;
  bool res = _p->solveSigmaAndMu(timeDimension(), dataDimension(), lambda,
                                 alpha, Mu, &S);
  if (Sigma) *Sigma = S;
  return res;
}

void GaussianInterpolationNoisy::setBoundaryConstraint(int const& b_type) {
  BoundaryType b = static_cast<BoundaryType>(b_type);
  if (_p->_boundary != b) _p->_prior_dirty = true;
  _p->_boundary = b;
}

}  // namespace ML
