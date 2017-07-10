// Copyright 2015 Byungkuk Choi.

#ifndef MLCORE_MATHSPARSETYPES_H_
#define MLCORE_MATHSPARSETYPES_H_

#include <Eigen/Sparse>

#include <ml/core/mathtypes.h>

namespace ML
{

using Trp                = Eigen::Triplet<Scalar, int>;
using SpMat              = Eigen::SparseMatrix<Scalar>;
using SimplicalCholSpMat = Eigen::SimplicialCholesky<SpMat>;
using SparseQR           = Eigen::SparseQR<SpMat, Eigen::NaturalOrdering<int>>;

}  // namespace ML

#endif  // MLCORE_MATHSPARSETYPES_H_
