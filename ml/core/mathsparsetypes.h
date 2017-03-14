// Copyright 2015 Byungkuk Choi.

#ifndef MLCORE_MATHSPARSETYPES_H_
#define MLCORE_MATHSPARSETYPES_H_

#include <Eigen/Sparse>
#include "core/mathtypes.h"

namespace ML {

typedef Eigen::Triplet<Scalar, int> Trp;
typedef Eigen::SparseMatrix<Scalar> SpMat;
typedef Eigen::SimplicialCholesky<SpMat> SimplicalCholSpMat;
typedef Eigen::SparseQR<SpMat, Eigen::NaturalOrdering<int> > SparseQR;

}  // namespace ML

#endif  // MLCORE_MATHSPARSETYPES_H_
