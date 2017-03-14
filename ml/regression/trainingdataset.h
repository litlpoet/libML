// Copyright (C) 2015 BK

#ifndef MLREGRESSION_TRAININGDATASET_H_
#define MLREGRESSION_TRAININGDATASET_H_

#include <memory>
#include <vector>

#include "core/mathtypes.h"

namespace ML {

class TrainingDataSet {
 public:
  explicit TrainingDataSet(int const& n_dim_Y);

  virtual ~TrainingDataSet();

  bool isEmpty() const;

  int const& size() const;

  VecN const& x(int const& i) const;

  Scalar const& y(int const& d, int const& i) const;

  std::vector<Scalar> const& Y(int const& d) const;

  void append(VecN const& x, VecN const& y);

  void setY(int const& d, int const& i, Scalar const& y);

  void clear();

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLREGRESSION_TRAININGDATASET_H_
