// Copyright (c) 2015 Byungkuk Choi

#include <gtest/gtest.h>
#include "MLGaussian/gaussianinterpolation.h"

class TestMLGaussian : public ::testing::Test {
 protected:
  void SetUp() {
    _frames = 10;
    _t_end = 1.0;

    ML::VecN vec_3d(3);

    _t_data.push_back(ML::T_Sample(0, (vec_3d << 0.0f, 0.0f, 0.0f).finished()));
    _t_data.push_back(ML::T_Sample(5, (vec_3d << 1.0f, 2.0f, 0.5f).finished()));
    _t_data.push_back(ML::T_Sample(9, (vec_3d << 0.0f, 0.0f, 0.0f).finished()));

    _g_interp = new ML::GaussianInterpolation(_frames, _t_end, _t_data);
  }

  void TearDown() { delete _g_interp; }

  int _frames;
  float _t_end;
  ML::TimeSeries _t_data;
  ML::GaussianInterpolation* _g_interp;
};

TEST_F(TestMLGaussian, DimensionTest) {
  ML::MatNxN mean;
  _g_interp->solve(1.0f, &mean);
  EXPECT_EQ(_frames, mean.rows());
  EXPECT_EQ(3, mean.cols());
}

TEST_F(TestMLGaussian, SampleValueTest) {
  ML::MatNxN mean;
  _g_interp->solve(1.0f, &mean);
  EXPECT_TRUE(mean.row(0).transpose() == _t_data.at(0).second);
  EXPECT_TRUE(mean.row(5).transpose() == _t_data.at(1).second);
  EXPECT_TRUE(mean.row(9).transpose() == _t_data.at(2).second);
}
