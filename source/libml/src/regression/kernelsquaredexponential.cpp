// Copyright (C) 2015 BK

#include <ml/regression/kernelsquaredexponential.h>

namespace ML {

class KernelSquaredExponential::Imple {
 public:
  Scalar _ell{4.0f};
  Scalar _sf2{1.0f};

  Imple() {}

  ~Imple() {}
};

KernelSquaredExponential::KernelSquaredExponential()
    : KernelFunction(), _p(new Imple) {}

KernelSquaredExponential::~KernelSquaredExponential() {}

bool KernelSquaredExponential::init(int const& n_dim_X) {
  setXDimension(n_dim_X);
  setParameterDimension(2);
  return true;
}

void KernelSquaredExponential::initLogParameters(VecN const& param) {
  KernelFunction::initLogParameters(param);
  VecN const& log_params = logParameters();
  _p->_ell = exp(log_params(0));
  _p->_sf2 = exp(2.0f * log_params(1));
}

Scalar KernelSquaredExponential::cov(VecN const& x1, VecN const& x2) {
  Scalar z = ((x1 - x2) / _p->_ell).squaredNorm();
  return _p->_sf2 * exp(-0.5f * z);
}

void KernelSquaredExponential::grad(VecN const& x1, VecN const& x2,
                                    VecN* grad) {
  Scalar z = ((x1 - x2) / _p->_ell).squaredNorm();
  Scalar k = _p->_sf2 * exp(-0.5 * z);
  *grad << k * z, 2 * k;
}

}  // namespace ML
