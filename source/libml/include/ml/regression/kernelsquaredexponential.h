// Copyright (C) 2015 BK

#ifndef MLREGRESSION_KERNELSQUAREDEXPONENTIAL_H_
#define MLREGRESSION_KERNELSQUAREDEXPONENTIAL_H_

#include <ml/regression/kernelfunction.h>

namespace ML
{

class KernelSquaredExponential : public KernelFunction
{
 public:
  KernelSquaredExponential();

  ~KernelSquaredExponential() override;

  bool
  init(int const& n_dim_X) override;

  void
  initLogParameters(VecN const& log_params) override;

  Scalar
  cov(VecN const& x1, VecN const& x2) override;

  void
  grad(VecN const& x1, VecN const& x2, VecN* grad) override;

 private:
  class Imple;
  unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLREGRESSION_KERNELSQUAREDEXPONENTIAL_H_
