// Copyright 2015 Byungkuk Choi.

#ifndef MLGAUSSIANINTERPOLATION_HPP_
#define MLGAUSSIANINTERPOLATION_HPP_

#include <memory>
#include "MLCore/timeseriesdata.h"

namespace ML {

class GaussianInterpolation {
 public:
  GaussianInterpolation(const int& D, const float& T,
                        const TimeSeries& time_series_data);

  ~GaussianInterpolation();

  bool solve(const float& lambda, MatNxN* Mu, MatNxN* Sigma = nullptr);

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLGAUSSIANINTERPOLATION_HPP_