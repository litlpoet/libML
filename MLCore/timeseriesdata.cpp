// Copyright (c) 2015 Byungkuk Choi

#include "MLCore/timeseriesdata.h"

#include <cstdarg>

namespace ML {

T_Sample MakeTimeSample(const int& f, int n_d, ...) {
  VecN sample(n_d);

  std::va_list args;
  va_start(args, n_d);
  for (int i = 0; i < n_d; ++i) sample[i] = va_arg(args, double);
  va_end(args);

  return T_Sample(f, sample);
}

}  // namespace ML