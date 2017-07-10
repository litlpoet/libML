// Copyright (C) 2015 BK

#include <ml/regression/trainingdataset.h>

#include <iostream>
#include <vector>

namespace ML
{

class TrainingDataSet::Imple
{
 public:
  int                              _n_dim_Y{0};
  int                              _n_data{0};
  std::vector<VecN>                _X;
  std::vector<std::vector<Scalar>> _Y;

  explicit Imple(int const& n_dim_Y) : _n_dim_Y(n_dim_Y)
  {
    _Y.resize(n_dim_Y);
  }

  ~Imple()
  {
  }
};

TrainingDataSet::TrainingDataSet(int const& n_dim_Y) : _p(new Imple(n_dim_Y))
{
}

TrainingDataSet::~TrainingDataSet()
{
}

bool
TrainingDataSet::isEmpty() const
{
  return _p->_n_data == 0;
}

int const&
TrainingDataSet::size() const
{
  return _p->_n_data;
}

VecN const&
TrainingDataSet::x(int const& i) const
{
  return _p->_X.at(i);
}

Scalar const&
TrainingDataSet::y(int const& d, int const& i) const
{
  return _p->_Y.at(d).at(i);
}

std::vector<Scalar> const&
TrainingDataSet::Y(int const& d) const
{
  return _p->_Y.at(d);
}

void
TrainingDataSet::append(VecN const& x, VecN const& y)
{
  assert(y.size() == _p->_n_dim_Y);
  _p->_X.push_back(x);
  for (int d = 0; d < _p->_n_dim_Y; ++d)
    _p->_Y[d].push_back(y(d));
  _p->_n_data = _p->_X.size();
}

void
TrainingDataSet::setY(int const& d, int const& i, Scalar const& y)
{
  if (d >= _p->_n_dim_Y || i >= _p->_n_data)
    return;
  _p->_Y[d][i] = y;
}

void
TrainingDataSet::clear()
{
  _p->_X.clear();
  _p->_Y.clear();
  _p->_n_data = 0;
}

}  // namespace ML
