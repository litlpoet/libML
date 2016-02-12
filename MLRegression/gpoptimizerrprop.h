// Copyright (C) 2015 BK

#ifndef MLREGRESSION_GPOPTIMIZERRPROP_H_
#define MLREGRESSION_GPOPTIMIZERRPROP_H_

#include "MLRegression/gaussianprocess.h"

namespace ML {

class GPOptimizerRprop {
 public:
  GPOptimizerRprop();

  virtual ~GPOptimizerRprop();

  void init(Scalar const& delta0 = 0.1f, Scalar const& delta_min = 1e-6f,
            Scalar const& delta_max = 50.0f, Scalar const& eta_minus = 0.5f,
            Scalar const& eta_plus = 1.2f, Scalar const& eps_stop = 0.0f);

  void maximize(GPRegression* gp, int const& n = 100,
                bool const& verbose = false);

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};
}  // namespace ML

#endif  // MLREGRESSION_GPOPTIMIZERRPROP_H_
