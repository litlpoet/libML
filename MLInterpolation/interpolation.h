// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_INTERPOLATION_H_
#define MLINTERPOLATION_INTERPOLATION_H_

#include <memory>

#include "MLCore/timeseriesdata.h"

namespace ML {

class Interpolation {
 public:
  Interpolation(int const& D, TimeSeriesMap const& time_series_map);

  Interpolation(TimeSeriesDense const& time_series_dense);

  virtual ~Interpolation();

  virtual bool solve(float const& lambda, float const& alpha, MatNxN* Mu,
                     MatNxN* Sigma = nullptr);

  virtual bool solve(int const& initial_n_knots, int const& level, MatNxN* R);

  int const& timeDimension() const;

  int const& dataDimension() const;

 protected:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_INTERPOLATION_H_
