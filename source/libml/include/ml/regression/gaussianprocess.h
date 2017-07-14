// Copyright (C) 2015 BK

#ifndef MLREGRESSION_GAUSSIANPROCESS_H_
#define MLREGRESSION_GAUSSIANPROCESS_H_

#include <ml/regression/regression.h>

namespace ML
{

class KernelFunction;

class GPRegression : public Regression
{
 public:
  explicit GPRegression(int const& n_dim_D);

  GPRegression(int const& n_dim_D, TimeSeriesMap const& time_series_map);

  ~GPRegression() final;

  void
  setInitialTrainingData(TimeSeriesMap const& time_series_map);

  void
  addTrainingData(VecN const& x, VecN const& y);

  void
  clearTrainingData();

  bool
  solve(MatNxN* Mu, MatNxN* Sigma = nullptr) final;

  KernelFunction*
  kernelFunction(int const& d_Y);

  Scalar
  logLikelihood(int const& d_Y);

  VecN
  logLikelihoodGrad(int const& d_Y);

 private:
  class Imple;
  unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLREGRESSION_GAUSSIANPROCESS_H_
