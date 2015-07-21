// Copyright (c) 2015 Byungkuk Choi

#ifndef MLGAUSSIAN_UNIT_TEST_TEST_GAUSSIANINTERP_H_
#define MLGAUSSIAN_UNIT_TEST_TEST_GAUSSIANINTERP_H_

#include <gtest/gtest.h>
#include "MLGaussian/gaussianinterpolation.h"

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
  void SetUp() {}

  void TearDown() {}
};

#endif  // MLGAUSSIAN_UNIT_TEST_TEST_GAUSSIANINTERP_H_
