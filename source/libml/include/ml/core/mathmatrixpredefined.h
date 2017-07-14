// Copyright (c) 2015 Byungkuk Choi

#ifndef MLCORE_MATHMATRIXPREDEFINED_H_
#define MLCORE_MATHMATRIXPREDEFINED_H_

#include <memory>

#include <ml/core/mathsparsetypes.h>

using std::unique_ptr;

namespace ML
{

unique_ptr<SpMat>
MakeFiniteDifferenceMat(size_t const& dim);

unique_ptr<SpMat>
MakeFiniteDifferenceMatWithBoundary(size_t const& dim);

unique_ptr<SpMat>
MakeFiniteDifferenceMatWithC2Boundary(size_t const& dim);

}  // namespace ML

#endif  // MLCORE_MATHMATRIXPREDEFINED_H_
