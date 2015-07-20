// Copyright 2015 Byungkuk Choi.

#ifndef MLCORE_MATHTYPES_H_
#define MLCORE_MATHTYPES_H_

#define USE_DOUBLE 0

#include <Eigen/Core>

namespace ML {

#if USE_DOUBLE
typedef double Scalar;
#else
typedef float Scalar;
#endif

#if USE_DOUBLE
typedef Eigen::Vector3d Vec3;
typedef Eigen::VectorXd VecN;
typedef Eigen::MatrixXd MatNxN;
typedef Eigen::RowVectorXd RVecN;
#else
typedef Eigen::Vector3f Vec3;
typedef Eigen::VectorXf VecN;
typedef Eigen::MatrixXf MatNxN;
typedef Eigen::RowVectorXf RVecN;
#endif

typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PmMat;
typedef Eigen::VectorXi VecNi;

}  // namespace ML

#endif  // MLCORE_MATHTYPES_H_
