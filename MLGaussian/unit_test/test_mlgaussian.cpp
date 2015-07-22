// Copyright (c) 2015 Byungkuk Choi

#include "MLGaussian/unit_test/test_gaussianinterp.h"

// Test Gaussian Interpolation (without noise assumption)

TEST_F(TestMLGaussianInterp, DimensionTest) {
  ML::MatNxN mean;
  _g_interp->solve(1.0f, &mean);
  EXPECT_EQ(_frames, mean.rows());
  EXPECT_EQ(3, mean.cols());
}

TEST_F(TestMLGaussianInterp, SampleValueTest) {
  ML::MatNxN mean;
  _g_interp->solve(1.0f, &mean);
  EXPECT_TRUE(mean.row(0).transpose() == _t_data.at(0));
  EXPECT_TRUE(mean.row(5).transpose() == _t_data.at(5));
  EXPECT_TRUE(mean.row(9).transpose() == _t_data.at(9));
}

// Test Gaussian Interpolation (with noise assumption)

TEST_F(TestMLGaussianInterpNoisy, DimensionTest) {
  ML::MatNxN mean;
  ML::MatNxN var;
  _g_interp_noisy->solve(1.f, &mean, &var);
  EXPECT_EQ(_frames, mean.rows());
  EXPECT_EQ(3, mean.cols());
  EXPECT_EQ(_frames, var.rows());
  EXPECT_EQ(_frames, var.cols());
}

TEST_F(TestMLGaussianInterpNoisy, PrecisionTest) {
  ML::MatNxN mean;
  ML::MatNxN var;
  _g_interp_noisy->solve(30.f, &mean, &var);
  std::cout << "Noisy mean with precision " << 30.0f << ":" << std::endl;
  std::cout << mean << std::endl;

  _g_interp_noisy->solve(0.01f, &mean, &var);
  std::cout << "Noisy mean with precision " << 0.01f << ":" << std::endl;
  std::cout << mean << std::endl;
}
