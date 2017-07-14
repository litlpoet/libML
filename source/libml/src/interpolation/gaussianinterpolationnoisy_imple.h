#ifndef GAUSSIANINTERPOLATIONNOISY_IMPLE_H
#define GAUSSIANINTERPOLATIONNOISY_IMPLE_H

#include <ml/interpolation/gaussianinterpolationnoisy.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include <ml/core/mathmatrixpredefined.h>
#include <ml/interpolation/interpolationtypes.h>

using std::vector;

namespace ML
{

class GaussianInterpolationNoisy::Imple
{
 public:
  BoundaryType       _boundary{BoundaryType::None};  // boundary condition type
  bool               _prior_dirty{true};             // dirty bit for prior computation
  size_t             _n_X{0};                        // number of observed samples
  unique_ptr<MatNxN> _Y{nullptr};                    // given sample values
  unique_ptr<SpMat>  _A{nullptr};                    // linear gaussian system matrix
  unique_ptr<SpMat>  _L{nullptr};                    // temporal smoothness prior matrix
  SpMat              _L_p;                           // prior mat multiplied by lambda

  Imple(size_t const& n_dim_D, size_t const& n_dim_X, TimeSeriesMap const& time_series_map)
      : _n_X(time_series_map.size())
  {
    _prepareSystem(n_dim_D, n_dim_X, time_series_map);
  }

  Imple(size_t const&          n_dim_D,
        size_t const&          n_dim_X,
        size_t const&          sampling_rate,
        TimeSeriesDense const& time_series_dense)
      : _n_X(n_dim_D / sampling_rate)
  {
    _prepareSystem(n_dim_D, n_dim_X, sampling_rate, time_series_dense);
  }

  ~Imple() = default;

  bool
  solveSigmaAndMu(size_t const& n_dim_D,
                  size_t const& n_dim_X,
                  Scalar const& lambda,
                  Scalar const& alpha,
                  MatNxN*       Mu,
                  MatNxN*       Sig_x_y)
  {
    _preparePrior(n_dim_D);
    _multiplyLambdaToPrior(lambda);

    // MatNxN Sig_x = MatNxN::Constant(_D, _D, 1e-3);
    MatNxN Sig_x_inv = SpMat(_L_p.transpose() * _L_p);
    // MatNxN Sig_x = Sig_x_inv.llt().solve(MatNxN::Identity(n_dim_D, n_dim_D));

    SpMat  AT          = _A->transpose();
    MatNxN Sig_y       = alpha * MatNxN::Identity(_n_X, _n_X);
    MatNxN Sig_y_inv   = Sig_y.inverse();
    MatNxN Sig_x_y_inv = Sig_x_inv + AT * Sig_y_inv * (*_A);
    (*Sig_x_y)         = Sig_x_y_inv.llt().solve(MatNxN::Identity(n_dim_D, n_dim_D));
    MatNxN LGS         = (*Sig_x_y) * AT * Sig_y_inv;
    Mu->resize(n_dim_D, n_dim_X);
    for (auto i = 0ul; i < n_dim_X; ++i)
    {
      Mu->col(i) = LGS * _Y->col(i);
    }
    return true;
  }

 private:
  void
  _prepareSystem(size_t const& n_dim_D, size_t const& n_dim_X, TimeSeriesMap const& time_series_map)
  {
    _A.reset(new SpMat(static_cast<SpMatId>(_n_X), static_cast<SpMatId>(n_dim_D)));
    _Y.reset(new MatNxN(_n_X, n_dim_X));
    vector<Trp> trps;
    size_t      i = 0ul;
    for (auto const& it : time_series_map)
    {
      _Y->row(i) = it.second.transpose();
      trps.push_back(Trp(i, it.first, 1.0));
      ++i;
    }
    _A->setFromTriplets(trps.begin(), trps.end());
  }

  void
  _prepareSystem(size_t const&          n_dim_D,
                 size_t const&          n_dim_X,
                 size_t const&          sampling_rate,
                 TimeSeriesDense const& time_series_dense)
  {
    _A.reset(new SpMat(static_cast<SpMatId>(_n_X), static_cast<SpMatId>(n_dim_D)));
    _Y.reset(new MatNxN(_n_X, n_dim_X));
    vector<Trp> trps;
    for (size_t f = 0, i = 0; f < n_dim_D; f += sampling_rate, ++i)
    {
      _Y->row(i) = time_series_dense.at(f).transpose();
      trps.push_back(Trp(i, f, 1));
    }
    _A->setFromTriplets(trps.begin(), trps.end());
  }

  void
  _preparePrior(size_t const& n_dim_D)
  {
    if (!_prior_dirty)
      return;

    if (_boundary == BoundaryType::C1)
      _L = MakeFiniteDifferenceMatWithBoundary(n_dim_D);
    else if (_boundary == BoundaryType::C2)
      _L = MakeFiniteDifferenceMatWithC2Boundary(n_dim_D);
    else
      _L = MakeFiniteDifferenceMat(n_dim_D);

    _prior_dirty = false;
  }

  void
  _multiplyLambdaToPrior(Scalar const& lambda)
  {
    _L_p = (*_L);

    if (_boundary == BoundaryType::C1)
      _L_p.block(1, 0, _L->rows() - 2, _L->cols()) *= lambda;
    else if (_boundary == BoundaryType::C2)
      _L_p.block(2, 0, _L->rows() - 4, _L->cols()) *= lambda;
    else
      _L_p *= lambda;
  }
};

}  // namespace ML

#endif  // GAUSSIANINTERPOLATIONNOISY_IMPLE_H
