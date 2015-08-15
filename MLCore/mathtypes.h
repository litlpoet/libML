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
typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector4d Vec4;
typedef Eigen::VectorXd VecN;
typedef Eigen::RowVector2d RVec2;
typedef Eigen::RowVector3d RVec3;
typedef Eigen::RowVector4d RVec4;
typedef Eigen::RowVectorXd RVecN;
typedef Eigen::MatrixXd MatNxN;
#else
typedef Eigen::Vector2f Vec2;
typedef Eigen::Vector3f Vec3;
typedef Eigen::Vector4f Vec4;
typedef Eigen::VectorXf VecN;
typedef Eigen::RowVector2f RVec2;
typedef Eigen::RowVector3f RVec3;
typedef Eigen::RowVector4f RVec4;
typedef Eigen::RowVectorXf RVecN;
typedef Eigen::MatrixXf MatNxN;
#endif

typedef Eigen::Matrix<Scalar, 2, 2> Mat2x2;
typedef Eigen::Matrix<Scalar, 3, 3> Mat3x3;
typedef Eigen::Matrix<Scalar, 4, 4> Mat4x4;
typedef Eigen::Matrix<Scalar, 3, 4> Mat3x4;
typedef Eigen::Matrix<Scalar, 4, 3> Mat4x3;

typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PmMat;
typedef Eigen::VectorXi VecNi;

}  // namespace ML

#endif  // MLCORE_MATHTYPES_H_
