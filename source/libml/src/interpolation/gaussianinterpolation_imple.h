#ifndef GAUSSIANINTERPOLATION_IMPLE_CPP
#define GAUSSIANINTERPOLATION_IMPLE_CPP

#include <ml/interpolation/gaussianinterpolation.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <ml/core/mathmatrixpredefined.h>

namespace ML
{

class GaussianInterpolation::Imple
{
 public:
  unique_ptr<MatNxN> _X2{nullptr};  // (sorted) given sample data
  unique_ptr<SpMat>  _L1{nullptr};  // prior for unknowns
  unique_ptr<SpMat>  _L2{nullptr};  // prior for knowns
  std::vector<bool>  _has_data_at_t;
  TimeSeriesMap      _time_series_map;

  Imple(int const& D, int const& D_X, TimeSeriesMap const& time_series_map)
      : _has_data_at_t(D, false)
      , _time_series_map(time_series_map)
  {
    _prepareSystem(D, D_X);
  }

  ~Imple() = default;

  bool
  solveMean(int const& D, int const& D_X, MatNxN* Mu)
  {
    SparseQR sparse_qr(*_L1);
    MatNxN   mean(_L1->cols(), D_X);
    for (int i    = 0; i < D_X; ++i)
      mean.col(i) = sparse_qr.solve(-1.0f * (*_L2) * _X2->col(i));
    Mu->resize(D, mean.cols());

    for (int i = 0, j = 0; i < D; ++i)
      if (_has_data_at_t[i])
        Mu->row(i) = _time_series_map.at(i).transpose();
      else
        Mu->row(i) = mean.row(j++);

    return true;
  }

  bool
  solveVariance(int const& D, float const& lambda, MatNxN* Sigma)
  {
    SpMat L1TL1 = _L1->transpose() * (*_L1) * std::pow(1.0f / lambda, 2);
    SpMat I(L1TL1.rows(), L1TL1.cols());
    I.setIdentity();

    SimplicalCholSpMat chol(L1TL1);
    SpMat              L1TL1_inv = chol.solve(I);
    MatNxN             var       = L1TL1_inv.diagonal().cwiseSqrt();

    Sigma->resize(D, var.cols());

    for (int i = 0, j = 0; i < D; ++i)
    {
      if (_has_data_at_t[i])
        Sigma->row(i).setZero();
      else
        Sigma->row(i) = var.row(j++);
    }

    return true;
  }

 private:
  void
  _prepareSystem(int const& D, int const& D_X)
  {
    size_t N = _time_series_map.size();
    for (auto const& it : _time_series_map)
      _has_data_at_t[it.first] = true;

    unique_ptr<SpMat> L = MakeFiniteDifferenceMat(D);
    _X2.reset(new MatNxN(N, D_X));

    // make permutation vector and arrange data accordingly
    VecNi pm_vec(D);
    int   j = 0;

    // first, find domain without data samples
    for (int i = 0; i < D; ++i)
      if (!_has_data_at_t[i])
        pm_vec[j++] = i;

    // second, find domain with data samples
    int i = 0;
    for (auto const& it : _time_series_map)
    {
      pm_vec[j++]   = it.first;
      _X2->row(i++) = it.second.transpose();
    }

    assert(j == D);

    // permute matrix L using permutation matrix (pm_mat)
    PmMat pm_mat(pm_vec);
    SpMat Lp = *L * pm_mat;

    _L1.reset(new SpMat(Lp.topLeftCorner(Lp.rows(), Lp.cols() - N)));
    _L2.reset(new SpMat(Lp.topRightCorner(Lp.rows(), N)));
    _L1->makeCompressed();
    _L2->makeCompressed();
  }
};

}  // namespace ML

#endif  // GAUSSIANINTERPOLATION_IMPLE_CPP
