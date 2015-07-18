// Copyright 2015 Byungkuk Choi.

#ifndef MLMATHSPARSETYPES_HPP_
#define MLMATHSPARSETYPES_HPP_

#include <Eigen/Sparse>
#include "MLCore/mathtypes.h"

namespace ML {

typedef Eigen::Triplet<Scalar, int> Trp;
typedef Eigen::SparseMatrix<Scalar> SpMat;
typedef Eigen::SimplicialCholesky<SpMat> SimplicalCholSpMat;
typedef Eigen::SparseQR<SpMat, Eigen::NaturalOrdering<int> > SparseQR;

}  // namespace ML

#endif  // MLMATHSPARSETYPES_HPP_
