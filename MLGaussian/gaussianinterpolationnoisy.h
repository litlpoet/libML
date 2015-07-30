// Copyright 2015 Byungkuk Choi.

#ifndef MLGAUSSIAN_GAUSSIANINTERPOLATIONNOISY_H_
#define MLGAUSSIAN_GAUSSIANINTERPOLATIONNOISY_H_

#include <memory>
#include "MLCore/timeseriesdata.h"

namespace ML {

class GaussianInterpolationNoisy {
 public:
  GaussianInterpolationNoisy(const int& D,
                             const TimeSeriesMap& time_series_data);

  ~GaussianInterpolationNoisy();

  int dimension();

  int sampleDimension();

  bool solve(const float& lambda, MatNxN* Mu, MatNxN* Sigma);

  void setBoundaryConstraint(const bool& b);

  const TimeSeriesMap& timeSeriesMap() const;

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLGAUSSIAN_GAUSSIANINTERPOLATIONNOISY_H_
