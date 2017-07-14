// Copyright 2015 Byungkuk Choi.

#ifndef MLCORE_SPLINEBASIS_H_
#define MLCORE_SPLINEBASIS_H_

#include <ml/core/mathtypes.h>

namespace ML
{

void
CubicBSpline(Mat4x4* B)
{
  // clang-format off
  *B << -1,  3, -3, 1,
         3, -6,  3, 0,
        -3,  0,  3, 0,
         1,  4,  1, 0;
  // clang-format on
}

void
CubicBSpline1stDeriv(Mat3x4* dB)
{
  // clang-format off
  *dB << -1,  3, -3, 1,
          2, -4,  2, 0,
         -1,  0,  1, 0;
  // clang-format on
}

}  // namespace ML

#endif  // MLCORE_SPLINEBASIS_H_
