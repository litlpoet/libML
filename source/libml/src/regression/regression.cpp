// Copyright (C) 2015 BK

#include <ml/regression/regression.h>

#include <ml/core/exceptions.h>

namespace ML {

class Regression::Imple {
 public:
  int _n_dim_D{0};
  int _n_dim_X{0};
  int _n_dim_Y{0};

  explicit Imple(int const& n_dim_D) : _n_dim_D(n_dim_D) {}

  Imple(int const& n_dim_D, TimeSeriesMap const& time_series_map)
      : _n_dim_D(n_dim_D) {
    setDimensions(time_series_map);
  }

  ~Imple() {}

  void setDimensions(TimeSeriesMap const& time_series_map) {
    BadInputException ex_or("Regression | Out of range");
    BadInputException ex_bix("Regression | Bad input exception");
    if (_n_dim_D < 2 || time_series_map.size() == 0) throw ex_bix;
    _n_dim_X = 1;  // time_series : X is always 1d
    _n_dim_Y = static_cast<int>(time_series_map.begin()->second.size());
    for (auto const& it : time_series_map) {
      // each sample should in between total dimension 'D'
      if (it.first < 0 || it.first > _n_dim_D - 1) throw ex_or;
      // each sample data dimension should be all the same.
      if (_n_dim_Y != static_cast<int>(it.second.size())) throw ex_bix;
    }
  }
};

Regression::Regression(int const& n_dim_D) : _p(new Imple(n_dim_D)) {}

Regression::Regression(int const& n_dim_D, TimeSeriesMap const& time_series_map)
    : _p(new Imple(n_dim_D, time_series_map)) {}

Regression::~Regression() {}

void Regression::setDimensions(TimeSeriesMap const& time_series_map) {
  _p->setDimensions(time_series_map);
}

int const& Regression::timeDimension() const { return _p->_n_dim_D; }

int const& Regression::xDimension() const { return _p->_n_dim_X; }

int const& Regression::yDimension() const { return _p->_n_dim_Y; }

}  // namespace ML
