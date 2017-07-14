// Copyright 2015 Byungkuk Choi.

#ifndef MLCORE_MATHTYPES_H_
#define MLCORE_MATHTYPES_H_

#define USE_DOUBLE 1

#include <Eigen/Core>

namespace ML
{

#if USE_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif

#if USE_DOUBLE
using Vec2   = Eigen::Vector2d;
using Vec3   = Eigen::Vector3d;
using Vec4   = Eigen::Vector4d;
using VecN   = Eigen::VectorXd;
using RVec2  = Eigen::RowVector2d;
using RVec3  = Eigen::RowVector3d;
using RVec4  = Eigen::RowVector4d;
using RVecN  = Eigen::RowVectorXd;
using MatNxN = Eigen::MatrixXd;
#else
using Vec2   = Eigen::Vector2f;
using Vec3   = Eigen::Vector3f;
using Vec4   = Eigen::Vector4f;
using VecN   = Eigen::VectorXf;
using RVec2  = Eigen::RowVector2f;
using RVec3  = Eigen::RowVector3f;
using RVec4  = Eigen::RowVector4f;
using RVecN  = Eigen::RowVectorXf;
using MatNxN = Eigen::MatrixXf;
#endif

using Mat2x2 = Eigen::Matrix<Scalar, 2, 2>;
using Mat3x3 = Eigen::Matrix<Scalar, 3, 3>;
using Mat4x4 = Eigen::Matrix<Scalar, 4, 4>;
using Mat3x4 = Eigen::Matrix<Scalar, 3, 4>;
using Mat4x3 = Eigen::Matrix<Scalar, 4, 3>;
using PmMat  = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
using VecNi  = Eigen::VectorXi;

}  // namespace ML

#endif  // MLCORE_MATHTYPES_H_
