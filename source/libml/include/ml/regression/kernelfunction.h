// Copyright (C) 2015 BK

#ifndef MLREGRESSION_KERNELFUNCTION_H_
#define MLREGRESSION_KERNELFUNCTION_H_

#include <memory>

#include <ml/core/mathtypes.h>

namespace ML {

class KernelFunction {
 public:
  KernelFunction();

  virtual ~KernelFunction();

  virtual bool init(int const& n_dim_X);

  virtual void initLogParameters(VecN const& param);

  virtual float cov(VecN const& x1, VecN const& x2) = 0;

  virtual void grad(VecN const& x1, VecN const& x2, VecN* grad) = 0;

  bool const& isParametersDirty() const;

  int const& xDimension() const;

  int const& parameterDimension() const;

  VecN const& logParameters() const;

  void setParametersDirty(bool const& dirty);

  void setXDimension(int const& n_dim_X);

  void setParameterDimension(int const& n_dim_P);

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLREGRESSION_KERNELFUNCTION_H_
