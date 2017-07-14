// Copyright (C) 2015 BK

#include <ml/regression/kernelfunction.h>

namespace ML
{

class KernelFunction::Imple
{
 public:
  bool _is_log_param_dirty{false};
  int  _n_dim_X{0};
  int  _n_dim_P{0};
  VecN _log_params;

  Imple() = default;

  ~Imple() = default;
};

KernelFunction::KernelFunction()
    : _p(new Imple)
{
}

KernelFunction::~KernelFunction() = default;

bool
KernelFunction::init(int const& D_X)
{
  (void)D_X;
  return false;
}

void
KernelFunction::initLogParameters(VecN const& log_params)
{
  assert(log_params.size() == _p->_n_dim_P);
  _p->_log_params         = log_params;
  _p->_is_log_param_dirty = true;
}

bool const&
KernelFunction::isParametersDirty() const
{
  return _p->_is_log_param_dirty;
}

int const&
KernelFunction::xDimension() const
{
  return _p->_n_dim_X;
}

int const&
KernelFunction::parameterDimension() const
{
  return _p->_n_dim_P;
}

VecN const&
KernelFunction::logParameters() const
{
  return _p->_log_params;
}

void
KernelFunction::setParametersDirty(bool const& dirty)
{
  _p->_is_log_param_dirty = dirty;
}

void
KernelFunction::setXDimension(int const& n_dim_X)
{
  _p->_n_dim_X = n_dim_X;
}

void
KernelFunction::setParameterDimension(int const& n_dim_P)
{
  _p->_n_dim_P    = n_dim_P;
  _p->_log_params = VecN::Zero(n_dim_P);
}

}  // namespace ML
