// Copyright (c) 2015 Byungkuk Choi

#ifndef MLCORE_MATHMATRIXPREDEFINED_H_
#define MLCORE_MATHMATRIXPREDEFINED_H_

#include "core/mathsparsetypes.h"

namespace ML {

bool MakeFiniteDifferenceMat(int const& dim, SpMat* L);

bool MakeFiniteDifferenceMatWithBoundary(int const& dim, SpMat* L);

bool MakeFiniteDifferenceMatWithC2Boundary(int const& dim, SpMat* L);

}  // namespace ML

#endif  // MLCORE_MATHMATRIXPREDEFINED_H_
