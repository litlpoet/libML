// Copyright 2015 Byungkuk Choi.

#include "MLInterpolation/gaussianinterpolationnoisy.h"

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "MLCore/mathmatrixpredefined.h"
#include "MLInterpolation/interpolationtypes.h"

namespace ML {

class GaussianInterpolationNoisy::Imple {
 public:
  BoundaryType _boundary{BoundaryType::None};  // boundary condition type
  bool _prior_dirty{true};  // dirty bit for prior computation
  size_t _N{0};             // number of observed samples
  MatNxN* _Y{nullptr};      // given sample values
  SpMat* _A{nullptr};       // linear gaussian system matrix
  SpMat* _L{nullptr};       // temporal smoothness prior matrix
  SpMat* _L1{nullptr};      // prior for unknowns
  SpMat* _L2{nullptr};      // prior for knowns
  SpMat _L_p;               // prior mat multiplied by lambda
  std::vector<int> _c_fs;
  std::vector<bool> _has_data_at_t;

  Imple(int const& D, int const& D_X, TimeSeriesMap const& time_series_map)
      : _N(time_series_map.size()), _has_data_at_t(D, false) {
    prepareSystem(D, D_X, time_series_map);
    prepareConstSystem(D, time_series_map);
  }

  Imple(int const& D, int const& D_X, TimeSeriesDense const& time_series_dense)
      : _N(D) {
    prepareSystem(D, D_X, time_series_dense);
  }

  ~Imple() {
    if (_Y) delete _Y;
    if (_A) delete _A;
    if (_L) delete _L;
    if (_L1) delete _L1;
    if (_L2) delete _L2;
    _Y = nullptr;
    _A = nullptr;
    _L = nullptr;
    _L1 = nullptr;
    _L2 = nullptr;
  }

  void solveConstVariance(int const& D, float const& lambda, VecN* Sigma) {
    SpMat L1TL1 = _L1->transpose() * (*_L1) * std::pow(1.0f / lambda, 2);
    SpMat I(L1TL1.rows(), L1TL1.cols());
    I.setIdentity();

    SimplicalCholSpMat chol(L1TL1);
    SpMat L1TL1_inv = chol.solve(I);
    VecN var = L1TL1_inv.diagonal().cwiseSqrt();

    Sigma->resize(D);

    for (int i = 0, j = 0; i < D; ++i) {
      if (_has_data_at_t[i])
        Sigma->row(i).setZero();
      else
        Sigma->row(i) = var.row(j++);
    }
    std::cout << "Const Sigma:" << std::endl << *Sigma << std::endl;
    std::cout << "Const Sigma is " << Sigma->rows() << " by " << Sigma->cols()
              << std::endl;
  }

  bool solveSigmaAndMu(int const& D, int const& D_X, float const& lambda,
                       float const& alpha, MatNxN* Mu, MatNxN* Sig_x_y) {
    preparePrior(D);
    multiplyLambdaToPrior(lambda);

    // MatNxN Sig_x = MatNxN::Constant(_D, _D, 1e-3);
    VecN var;
    solveConstVariance(_L->cols(), lambda, &var);
    SpMat W(_L->cols(), _L->cols());
    for (int i = 0; i < _L->cols(); ++i)
      W.coeffRef(i, i) = sqrt(var(i) + 0.001f);
    MatNxN Sig_x_inv = SpMat(_L_p.transpose() * _L_p);
    Sig_x_inv = W * Sig_x_inv;
    MatNxN Sig_x = Sig_x_inv.llt().solve(MatNxN::Identity(D, D));

    SpMat AT = _A->transpose();
    MatNxN Sig_y = alpha * MatNxN::Identity(_N, _N);
    MatNxN Sig_y_inv = Sig_y.inverse();
    MatNxN Sig_x_y_inv = Sig_x_inv + AT * Sig_y_inv * (*_A);
    (*Sig_x_y) = Sig_x_y_inv.llt().solve(MatNxN::Identity(D, D));
    MatNxN LGS = (*Sig_x_y) * AT * Sig_y_inv;
    Mu->resize(D, D_X);
    for (int i = 0; i < D_X; ++i) Mu->col(i) = LGS * _Y->col(i);
    return true;
  }

 private:
  void prepareSystem(int const& D, int const& D_X,
                     TimeSeriesMap const& time_series_map) {
    int i = 0;
    _Y = new MatNxN(_N, D_X);
    std::vector<Trp> trps;
    _c_fs.clear();
    for (auto const& it : time_series_map) {
      _Y->row(i) = it.second.transpose();
      trps.push_back(Trp(i, it.first, 1.0f));
      _c_fs.push_back(it.first);
      ++i;
    }
    _A = new SpMat(_N, D);
    _A->setFromTriplets(trps.begin(), trps.end());
    _L = new SpMat;
  }

  void prepareSystem(int const& D, int const& D_X,
                     TimeSeriesDense const& time_series_dense) {
    int i = 0;
    _Y = new MatNxN(_N, D_X);
    std::vector<Trp> trps;
    for (auto const& it : time_series_dense) {
      _Y->row(i) = it.transpose();
      trps.push_back(Trp(i, i, 1));
      ++i;
    }
    _A = new SpMat(_N, D);
    _A->setFromTriplets(trps.begin(), trps.end());
    _L = new SpMat;
  }

  void preparePrior(int const& D) {
    if (!_prior_dirty) return;
    if (_boundary == BoundaryType::C1)
      MakeFiniteDifferenceMatWithBoundary(D, _L);
    else if (_boundary == BoundaryType::C2)
      MakeFiniteDifferenceMatWithC2Boundary(D, _L);
    else
      MakeFiniteDifferenceMat(D, _L);
    _prior_dirty = false;
  }

  void prepareConstSystem(int const& D, TimeSeriesMap const& time_series_map) {
    size_t N = time_series_map.size();
    for (auto const& it : time_series_map) _has_data_at_t[it.first] = true;

    SpMat L;
    MakeFiniteDifferenceMat(D, &L);

    // make permutation vector and arrange data accordingly
    VecNi pm_vec(D);
    int j = 0;

    // first, find domain without data samples
    for (int i = 0; i < D; ++i)
      if (!_has_data_at_t[i]) pm_vec[j++] = i;

    // second, find domain with data samples
    for (auto const& it : time_series_map) pm_vec[j++] = it.first;
    assert(j == D);

    // permute matrix L using permutation matrix (pm_mat)
    PmMat pm_mat(pm_vec);
    SpMat Lp = L * pm_mat;
    _L1 = new SpMat(Lp.topLeftCorner(Lp.rows(), Lp.cols() - N));
    _L2 = new SpMat(Lp.topRightCorner(Lp.rows(), N));
    _L1->makeCompressed();
    _L2->makeCompressed();
  }

  void multiplyLambdaToPrior(float const& lambda) {
    if (_boundary == BoundaryType::C1) {
      _L_p = (*_L);
      _L_p.block(1, 0, _L->rows() - 2, _L->cols()) *= lambda;
    } else if (_boundary == BoundaryType::C2) {
      _L_p = (*_L);
      _L_p.block(2, 0, _L->rows() - 4, _L->cols()) *= lambda;
    } else {
      _L_p = (*_L);
    }
  }
};

GaussianInterpolationNoisy::GaussianInterpolationNoisy(
    int const& D, TimeSeriesMap const& time_series_map)
    : Interpolation(D, time_series_map),
      _p(new Imple(D, dataDimension(), time_series_map)) {}

GaussianInterpolationNoisy::GaussianInterpolationNoisy(
    TimeSeriesDense const& time_series_dense)
    : Interpolation(time_series_dense),
      _p(new Imple(timeDimension(), dataDimension(), time_series_dense)) {}

GaussianInterpolationNoisy::~GaussianInterpolationNoisy() {}

bool GaussianInterpolationNoisy::solve(float const& lambda, float const& alpha,
                                       MatNxN* Mu, MatNxN* Sig_x_y) {
  MatNxN S;
  bool res = _p->solveSigmaAndMu(timeDimension(), dataDimension(), lambda,
                                 alpha, Mu, &S);
  if (Sig_x_y) *Sig_x_y = S;
  return res;
}

void GaussianInterpolationNoisy::setBoundaryConstraint(int const& b_type) {
  BoundaryType b = static_cast<BoundaryType>(b_type);
  if (_p->_boundary != b) _p->_prior_dirty = true;
  _p->_boundary = b;
}

}  // namespace ML
