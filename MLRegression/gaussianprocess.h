// Copyright (C) 2015 BK

#ifndef MLREGRESSION_GAUSSIANPROCESS_H_
#define MLREGRESSION_GAUSSIANPROCESS_H_

#include "MLRegression/regression.h"

namespace ML {

class GPRegression : public Regression {
 public:
  GPRegression(int const& n_dim_D, TimeSeriesMap const& time_series_map);

  ~GPRegression() final;

  bool solve(MatNxN* Mu, MatNxN* Sigma = nullptr) final;

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLREGRESSION_GAUSSIANPROCESS_H_
