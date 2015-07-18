// Copyright 2015 Byungkuk Choi.

#ifndef MLMATHTYPES_HPP_
#define MLMATHTYPES_HPP_

#define USE_DOUBLE 0

#include <Eigen/Core>

namespace ML {

#if USE_DOUBLE
typedef double Scalar;
#else
typedef float Scalar;
#endif

#if USE_DOUBLE
typedef Eigen::VectorXd VecN;
typedef Eigen::MatrixXd MatNxN;
#else
typedef Eigen::VectorXf VecN;
typedef Eigen::MatrixXf MatNxN;
typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PmMat;
typedef Eigen::RowVectorXf RVecN;
#endif

typedef Eigen::VectorXi VecNi;

}  // namespace ML

#endif  // MLMATHTYPES_HPP_
