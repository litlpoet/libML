// Copyright (C) 2015 BK

#ifndef MLREGRESSION_REGRESSION_H_
#define MLREGRESSION_REGRESSION_H_

#include <memory>

#include <ml/core/timeseriesdata.h>

namespace ML {

class Regression {
 public:
  explicit Regression(int const& n_dim_D);

  Regression(int const& n_dim_D, TimeSeriesMap const& time_series_map);

  virtual ~Regression();

  virtual bool solve(MatNxN* Mu, MatNxN* Sigma = nullptr) = 0;

  void setDimensions(TimeSeriesMap const& time_series_map);

  int const& timeDimension() const;

  int const& xDimension() const;

  int const& yDimension() const;

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLREGRESSION_REGRESSION_H_
