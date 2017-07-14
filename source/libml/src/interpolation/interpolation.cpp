// Copyright (C) 2015 BK

#include <ml/interpolation/interpolation.h>

#include <iostream>

#include <ml/core/exceptions.h>

namespace ML
{

class Interpolation::Imple
{
 public:
  size_t _D{0};    // discrete time dimension
  size_t _D_X{0};  // sample dimension

  Imple(size_t const& D, TimeSeriesMap const& time_series_map)
      : _D(D)
  {
    checkInputValidity(time_series_map);
  }

  explicit Imple(TimeSeriesDense const& time_series_dense)
      : _D(time_series_dense.size())
  {
    if (_D > 0)
      _D_X = time_series_dense.front().size();
  }

  ~Imple() = default;

 private:
  void
  checkInputValidity(TimeSeriesMap const& time_series_map)
  {
    BadInputException ex_or("Interpolation | Out of range");
    BadInputException ex_bix("Interpolation | Bad input exception");
    if (_D < 2)
      throw ex_bix;
    _D_X = time_series_map.begin()->second.size();
    for (auto const& it : time_series_map)
    {
      // each sample should in between total dimension 'D'
      if (it.first < 0 || it.first > _D - 1)
        throw ex_or;
      // each sample data dimension should be all the same.
      if (_D_X != it.second.size())
        throw ex_bix;
    }
  }
};

Interpolation::Interpolation(size_t const& D, TimeSeriesMap const& time_series_map)
    : _p(new Imple(D, time_series_map))
{
}

Interpolation::Interpolation(TimeSeriesDense const& time_series_dense)
    : _p(new Imple(time_series_dense))
{
}

Interpolation::~Interpolation() = default;

bool
Interpolation::solve(Scalar const&, Scalar const&, MatNxN*, MatNxN*)
{
  return false;
}

bool
Interpolation::solve(size_t const&, size_t const&, MatNxN*)
{
  return false;
}

size_t const&
Interpolation::timeDimension() const
{
  return _p->_D;
}

size_t const&
Interpolation::dataDimension() const
{
  return _p->_D_X;
}

}  // namespace ML
