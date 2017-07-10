// Copyright (C) 2015 BK

#include <ml/regression/gpoptimizerrprop.h>

#include <algorithm>
#include <iostream>

#include <ml/regression/kernelfunction.h>

namespace ML {

class GPOptimizerRprop::Imple {
 public:
  Scalar _delta0{0.1f};
  Scalar _delta_min{1e-6f};
  Scalar _delta_max{50.0f};
  Scalar _eta_minus{0.5f};
  Scalar _eta_plus{1.2f};
  Scalar _eps_stop{0.0f};

  Imple() {}

  ~Imple() {}

  Scalar sign(Scalar const& x) {
    return (x > 0) ? 1.0 : ((x < 0) ? -1.0f : 0.0);
  }

 private:
};

GPOptimizerRprop::GPOptimizerRprop() : _p(new Imple) {}

GPOptimizerRprop::~GPOptimizerRprop() {}

void GPOptimizerRprop::init(Scalar const& delta0, Scalar const& delta_min,
                            Scalar const& delta_max, Scalar const& eta_minus,
                            Scalar const& eta_plus, Scalar const& eps_stop) {
  _p->_delta0 = delta0;
  _p->_delta_min = delta_min;
  _p->_delta_max = delta_max;
  _p->_eta_minus = eta_minus;
  _p->_eta_plus = eta_plus;
  _p->_eps_stop = eps_stop;
}

void GPOptimizerRprop::maximize(GPRegression* gp, int const& n,
                                bool const& verbose) {
  int const& n_dim_Y = gp->yDimension();
  for (int d = 0; d < n_dim_Y; ++d) {
    KernelFunction* Kf = gp->kernelFunction(d);
    int const& n_dim_param = Kf->parameterDimension();
    VecN delta = VecN::Ones(n_dim_param) * _p->_delta0;
    VecN grad_old = VecN::Zero(n_dim_param);
    VecN params = Kf->logParameters();
    VecN best_params = params;
    Scalar best = log(0);

    for (int i = 0; i < n; ++i) {
      VecN grad = -gp->logLikelihoodGrad(d);
      grad_old = grad_old.cwiseProduct(grad);
      for (int j = 0; j < grad_old.size(); ++j) {
        if (grad_old(j) > 0) {
          delta(j) = std::min(delta(j) * _p->_eta_plus, _p->_delta_max);
        } else if (grad_old(j) < 0) {
          delta(j) = std::max(delta(j) * _p->_eta_minus, _p->_delta_min);
          grad(j) = 0;
        }
        params(j) += -_p->sign(grad(j)) * delta(j);
      }
      grad_old = grad;
      if (grad_old.norm() < _p->_eps_stop) break;
      Kf->initLogParameters(params);
      Scalar lik = gp->logLikelihood(d);
      if (verbose) std::cout << i << " " << -lik << std::endl;
      if (lik > best) {
        best = lik;
        best_params = params;
      }
    }
    Kf->initLogParameters(best_params);
  }
}

}  // namespace ML
