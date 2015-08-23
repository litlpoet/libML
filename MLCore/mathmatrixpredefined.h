// Copyright (c) 2015 Byungkuk Choi

#ifndef MLCORE_MATHMATRIXPREDEFINED_H_
#define MLCORE_MATHMATRIXPREDEFINED_H_

#include "MLCore/mathsparsetypes.h"

namespace ML {

bool MakeFiniteDifferenceMat(const int& dim, SpMat* L);

bool MakeFiniteDifferenceMatWithBoundary(const int& dim, SpMat* L);

bool MakeFiniteDifferenceMatWithC2Boundary(const int& dim, SpMat* L);

}  // namespace ML

#endif  // MLCORE_MATHMATRIXPREDEFINED_H_
