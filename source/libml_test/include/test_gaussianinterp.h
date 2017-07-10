// Copyright (c) 2015 Byungkuk Choi

#ifndef MLINTERPOLATION_UNIT_TEST_TEST_GAUSSIANINTERP_H_
#define MLINTERPOLATION_UNIT_TEST_TEST_GAUSSIANINTERP_H_

#include <gtest/gtest.h>

#include <ml/interpolation/gaussianinterpolation.h>
#include <ml/interpolation/gaussianinterpolationnoisy.h>
#include <ml/interpolation/multilevelbsplineinterpolation.h>

class TestMLGaussianInterp : public ::testing::Test {
 protected:
  void SetUp() {
    _frames = 10;

    _t_data.insert(ML::MakeTimeSample(0, 3, 0.0f, 0.0f, 0.0f));
    _t_data.insert(ML::MakeTimeSample(5, 3, 1.0f, 2.0f, 0.5f));
    _t_data.insert(ML::MakeTimeSample(9, 3, 0.0f, 0.0f, 0.0f));

    _g_interp = new ML::GaussianInterpolation(_frames, _t_data);
  }

  void TearDown() { delete _g_interp; }

  int _frames;
  ML::TimeSeriesMap _t_data;
  ML::GaussianInterpolation* _g_interp;
};

class TestMLGaussianInterpNoisy : public ::testing::Test {
 protected:
  void SetUp() {
    _frames = 10;

    _t_data.insert(ML::MakeTimeSample(0, 3, 0.0f, 0.0f, 0.0f));
    _t_data.insert(ML::MakeTimeSample(1, 3, 0.0f, 0.0f, 0.0f));
    _t_data.insert(ML::MakeTimeSample(5, 3, 1.0f, 2.0f, 0.5f));
    _t_data.insert(ML::MakeTimeSample(8, 3, 0.0f, 0.0f, 0.0f));
    _t_data.insert(ML::MakeTimeSample(9, 3, 0.0f, 0.0f, 0.0f));

    _g_interp_noisy = new ML::GaussianInterpolationNoisy(_frames, _t_data);
  }

  void TearDown() { delete _g_interp_noisy; }

  int _frames;
  ML::TimeSeriesMap _t_data;
  ML::GaussianInterpolationNoisy* _g_interp_noisy;
};

class TestMLMultiLevelBSplineInterp : public ::testing::Test {
 protected:
  void SetUp() {
    _frames = 10;

    _t_data.insert(ML::MakeTimeSample(0, 3, 0.0f, 0.0f, 0.0f));
    _t_data.insert(ML::MakeTimeSample(5, 3, 1.0f, 2.0f, 0.5f));
    _t_data.insert(ML::MakeTimeSample(9, 3, 0.0f, 0.0f, 0.0f));

    _mbsp_interp = new ML::MultiLevelBSplineInterpolation(_frames, _t_data);
  }

  void TearDown() { delete _mbsp_interp; }

  int _frames;
  ML::TimeSeriesMap _t_data;
  ML::MultiLevelBSplineInterpolation* _mbsp_interp;
};

#endif  // MLINTERPOLATION_UNIT_TEST_TEST_GAUSSIANINTERP_H_
