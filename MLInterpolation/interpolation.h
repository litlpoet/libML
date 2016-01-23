// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_INTERPOLATION_H_
#define MLINTERPOLATION_INTERPOLATION_H_

#include <memory>
#include "MLCore/timeseriesdata.h"

namespace ML {

class Interpolation {
 public:
  Interpolation(int const& D, TimeSeriesMap const& time_series_map);

  virtual ~Interpolation();

  virtual bool solve(float const& lambda, MatNxN* Mu, MatNxN* Sigma = nullptr);

  virtual bool solve(int const& initial_n_knots, int const& level, MatNxN* R);

  int const& timeDimension() const;

  int const& dataDimension() const;

  TimeSeriesMap const& timeSeriesMap() const;

 protected:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_INTERPOLATION_H_
