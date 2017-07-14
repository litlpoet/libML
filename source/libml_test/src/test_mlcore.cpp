// Copyright (c) 2015 Byungkuk Choi

#include <gtest/gtest.h>

#include <ml/core/mathmatrixpredefined.h>
#include <ml/core/mathtypes.h>

TEST(MathTypeTest, SizeTest)
{
  EXPECT_EQ(sizeof(ML::Scalar) * 3, sizeof(ML::Vec3));

  auto     r = 10;
  ML::VecN test_vec(r);
  EXPECT_EQ(r, test_vec.size());

  auto       c = 20;
  ML::MatNxN test_mat(r, c);
  EXPECT_EQ(r, test_mat.rows());
  EXPECT_EQ(c, test_mat.cols());

  unique_ptr<ML::SpMat> sp  = ML::MakeFiniteDifferenceMatWithBoundary(10);
  unique_ptr<ML::SpMat> sp2 = ML::MakeFiniteDifferenceMatWithC2Boundary(10);
}
