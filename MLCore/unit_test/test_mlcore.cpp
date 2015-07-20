// Copyright (c) 2015 Byungkuk Choi

#include <gtest/gtest.h>
#include "MLCore/mathtypes.h"

TEST(MathTypeTest, SizeTest) {
  EXPECT_EQ(sizeof(ML::Scalar) * 3, sizeof(ML::Vec3));

  const int r = 10;
  ML::VecN test_vec(r);
  EXPECT_EQ(r, test_vec.size());

  const int c = 20;
  ML::MatNxN test_mat(r, c);
  EXPECT_EQ(r, test_mat.rows());
  EXPECT_EQ(c, test_mat.cols());
}
