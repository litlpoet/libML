// Copyright 2015 Byungkuk Choi.

#ifndef MLINTERPOLATION_INTERPOLATION_H_
#define MLINTERPOLATION_INTERPOLATION_H_

#include <memory>

#include <ml/core/timeseriesdata.h>

using std::unique_ptr;

namespace ML
{

class Interpolation
{
 public:
  Interpolation(size_t const& D, TimeSeriesMap const& time_series_map);

  explicit Interpolation(TimeSeriesDense const& time_series_dense);

  virtual ~Interpolation();

  virtual bool
  solve(Scalar const& lambda, Scalar const& alpha, MatNxN* Mu, MatNxN* Sigma = nullptr);

  virtual bool
  solve(size_t const& initial_n_knots, size_t const& level, MatNxN* R);

  size_t const&
  timeDimension() const;

  size_t const&
  dataDimension() const;

 protected:
  class Imple;
  unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLINTERPOLATION_INTERPOLATION_H_
