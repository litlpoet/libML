// Copyright 2015 Byungkuk Choi.

#include "MLGaussian/gaussianinterpolationnoisy.h"

#include <Eigen/Dense>
#include <vector>

#include "MLCore/mathmatrixpredefined.h"

namespace ML {

class BadInputException : public std::exception {
  virtual const char* what() const throw() {
    return "GaussianInterpolationNoisy | Bad input exception";
  }
} bad_input_ex;

class GaussianInterpolationNoisy::Imple {
 public:
  int _D;      // discrete time dimension
  int _D_X;    // sample dimension
  MatNxN* _Y;  // given sample values
  SpMat* _A;   // linear gaussian system matrix
  SpMat* _L;   // temporal smoothness prior matrix
  TimeSeriesMap _time_series_map;

  Imple(const int& D, const TimeSeriesMap& time_series_data)
      : _D(D),
        _D_X(0),
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

 private:
  void checkInputValidity() {
    if (_D < 2) throw bad_input_ex;

    for (const auto& it : _time_series_map)
      if (it.first < 0 || it.first > _D - 1) throw bad_input_ex;
  }

  void prepareSystem() {
    const size_t N = _time_series_map.size();
    _D_X = static_cast<int>(_time_series_map.begin()->second.size());

    int i = 0;
    _Y = new MatNxN(N, _D_X);
    std::vector<Trp> triples;
    for (const auto& it : _time_series_map) {
      _Y->row(i++) = it.second.transpose();
      triples.push_back(Trp(i, it.first, 1));
    }

    _A = new SpMat(N, _D);
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

  (*_p->_L) *= lambda;

  MatNxN LTL = _p->_L->transpose() * *(_p->_L);
  LTL = (LTL.array() + 1e-3).matrix();
  MatNxN LTL_inv = LTL.llt().solve(MatNxN::Identity(_p->_D, _p->_D));

  MatNxN Sigma_y = MatNxN::Identity(_p->_A->rows(), _p->_A->rows());
  MatNxN Sigma_y_inv = Sigma_y.inverse();
  MatNxN Sigma_rh = LTL + _p->_A->transpose() * Sigma_y_inv * (*_p->_A);
  (*Sigma) = Sigma_rh.llt().solve(MatNxN::Identity(_p->_D, _p->_D));

  MatNxN PreMat = (*Sigma) * _p->_A->transpose() * Sigma_y;
  Mu->resize(_p->_D, _p->_D_X);
  for ( int i=0; i < _p->_D_X ; ++i)
    Mu->col(i) = PreMat * _p->_Y->col(i);

  return result;
}

}  // namespace ML
