// Copyright (C) 2015 BK

#ifndef MLREGRESSION_REGRESSION_H_
#define MLREGRESSION_REGRESSION_H_

#include <memory>

namespace ML {

class Regression {
 public:
  Regression();

  virtual ~Regression();

 private:
  class Imple;
  std::unique_ptr<Imple> _p;
};

}  // namespace ML

#endif  // MLREGRESSION_REGRESSION_H_
