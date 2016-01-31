// Copyright (c) 2015 Byungkuk Choi

#include "MLInterpolation/unit_test/test_gaussianinterp.h"

// Test Gaussian Interpolation (without noise assumption)

TEST_F(TestMLGaussianInterp, DimensionTest) {
  ML::MatNxN mean;
  _g_interp->solve(1.0f, 1.0f, &mean);
  EXPECT_EQ(_frames, mean.rows());
  EXPECT_EQ(3, mean.cols());
}

TEST_F(TestMLGaussianInterp, SampleValueTest) {
  ML::MatNxN mean;
  _g_interp->solve(1.0f, 1.0f, &mean);
  EXPECT_TRUE(mean.row(0).transpose() == _t_data.at(0));
  EXPECT_TRUE(mean.row(5).transpose() == _t_data.at(5));
  EXPECT_TRUE(mean.row(9).transpose() == _t_data.at(9));
}

// Test Gaussian Interpolation (with noise assumption)

TEST_F(TestMLGaussianInterpNoisy, DimensionTest) {
  ML::MatNxN mean;
  ML::MatNxN var;
  _g_interp_noisy->solve(1.f, 1.0f, &mean, &var);
  EXPECT_EQ(_frames, mean.rows());
  EXPECT_EQ(3, mean.cols());
  EXPECT_EQ(_frames, var.rows());
  EXPECT_EQ(_frames, var.cols());
}

TEST_F(TestMLGaussianInterpNoisy, PrecisionTest) {
  ML::MatNxN mean;
  ML::MatNxN var;
  _g_interp_noisy->solve(30.f, 1.0f, &mean, &var);
  std::cout << "Noisy mean with precision " << 30.0f << ":" << std::endl;
  std::cout << mean << std::endl;

  _g_interp_noisy->solve(0.01f, 1.0f, &mean, &var);
  std::cout << "Noisy mean with precision " << 0.01f << ":" << std::endl;
  std::cout << mean << std::endl;
}

// Test Multi-level B-Spline Interpolation

TEST_F(TestMLMultiLevelBSplineInterp, DimensionTest) {
  ML::MatNxN res;
  _mbsp_interp->solve(6, 2, &res);
  EXPECT_EQ(_frames, res.rows());
  EXPECT_EQ(3, res.cols());
}

TEST_F(TestMLMultiLevelBSplineInterp, PrecisionTest) {
  ML::MatNxN res;
  _mbsp_interp->solve(6, 2, &res);
  std::cout << "Multi level interp with level " << 2 << ":" << std::endl;
  std::cout << res << std::endl;
}
