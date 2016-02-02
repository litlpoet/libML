// Copyright (C) 2015 BK

#include "MLInterpolation/interpolation.h"

#include "iostream"

#include "MLCore/exceptions.h"

namespace ML {

class Interpolation::Imple {
 public:
  int _D{0};    // discrete time dimension
  int _D_X{0};  // sample dimension

  Imple(int const& D, TimeSeriesMap const& time_series_map) : _D(D) {
    checkInputValidity(time_series_map);
  }

  explicit Imple(TimeSeriesDense const& time_series_dense)
      : _D(time_series_dense.size()) {
    if (_D > 0) _D_X = time_series_dense.front().size();
    std::cout << "inside interp:" << _D << " by " << _D_X << std::endl;
  }

  ~Imple() {}

 private:
  void checkInputValidity(TimeSeriesMap const& time_series_map) {
    BadInputException ex_or("GaussianInterpolation | Out of range");
    BadInputException ex_bix("GaussianInterpolation | Bad input exception");
    if (_D < 2) throw ex_bix;
    _D_X = static_cast<int>(time_series_map.begin()->second.size());
    int i = 0;
    for (auto const& it : time_series_map) {
      // each sample should in between total dimension 'D'
      if (it.first < 0 || it.first > _D - 1) throw ex_or;
      // each sample data dimension should be all the same.
      if (_D_X != static_cast<int>(it.second.size())) throw ex_bix;
      std::cout << i++ << std::endl;
    }
  }
};

Interpolation::Interpolation(int const& D, TimeSeriesMap const& time_series_map)
    : _p(new Imple(D, time_series_map)) {}

Interpolation::Interpolation(TimeSeriesDense const& time_series_dense)
    : _p(new Imple(time_series_dense)) {
  std::cout << "dense interp" << std::endl;
}

Interpolation::~Interpolation() {}

bool Interpolation::solve(float const&, float const&, MatNxN*, MatNxN*) {
  return false;
}

bool Interpolation::solve(int const&, int const&, MatNxN*) { return false; }

int const& Interpolation::timeDimension() const { return _p->_D; }

int const& Interpolation::dataDimension() const { return _p->_D_X; }

}  // namespace ML
