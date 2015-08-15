// Copyright (C) 2015 BK

#include "MLInterpolation/interpolation.h"
#include "MLCore/exceptions.h"

namespace ML {

class Interpolation::Imple {
 public:
  int _D;    // discrete time dimension
  int _D_X;  // sample dimension
  TimeSeriesMap _time_series_map;

  Imple(const int& D, const TimeSeriesMap& time_series_map)
      : _D(D), _D_X(0), _time_series_map(time_series_map) {
    checkInputValidity();
  }

  ~Imple() {}

 private:
  void checkInputValidity() {
    BadInputException bad_input_ex(
        "GaussianInterpolation | Bad input exception");

    if (_D < 2) throw bad_input_ex;

    _D_X = static_cast<int>(_time_series_map.begin()->second.size());

    for (const auto& it : _time_series_map) {
      // each sample should in between total dimension 'D'
      if (it.first < 0 || it.first > _D - 1) throw bad_input_ex;
      // each sample data dimension should be all the same.
      if (_D_X != static_cast<int>(it.second.size())) throw bad_input_ex;
    }
  }
};

Interpolation::Interpolation(const int& D, const TimeSeriesMap& time_series_map)
    : _p(new Interpolation::Imple(D, time_series_map)) {}

Interpolation::~Interpolation() {}

bool Interpolation::solve(const float&, MatNxN*, MatNxN*) { return false; }

bool Interpolation::solve(const int&, const int&, MatNxN*) { return false; }

const int& Interpolation::timeDimension() const { return _p->_D; }

const int& Interpolation::dataDimension() const { return _p->_D_X; }

const TimeSeriesMap& Interpolation::timeSeriesMap() const {
  return _p->_time_series_map;
}

}  // namespace ML
