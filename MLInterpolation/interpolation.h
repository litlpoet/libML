// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_INTERPOLATION_H_
#define MLINTERPOLATION_INTERPOLATION_H_

#include <memory>
#include "MLCore/timeseriesdata.h"

namespace ML {

class Interpolation {
 public:
  Interpolation(const int& D, const TimeSeriesMap& time_series_map);

  virtual ~Interpolation();

  virtual bool solve(const float& lambda, MatNxN* Mu, MatNxN* Sigma = nullptr);

  virtual bool solve(const int& initial_n_knots, const int& level, MatNxN* R);

  const int& timeDimension() const;

  const int& dataDimension() const;

  const TimeSeriesMap& timeSeriesMap() const;

 protected:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_INTERPOLATION_H_
