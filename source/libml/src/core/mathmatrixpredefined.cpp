// Copyright (c) 2015 Byungkuk Choi

#include <ml/core/mathmatrixpredefined.h>

#include <iostream>
#include <vector>

using std::vector;

namespace ML
{

unique_ptr<SpMat>
MakeFiniteDifferenceMat(size_t const& dim)
{
  auto const  r_dim = dim - 2;
  vector<Trp> triples;
  triples.reserve(3 * r_dim);

  for (auto i = 0ul; i < r_dim; ++i)
  {
    triples.push_back(Trp(i, i, -1.0));
    triples.push_back(Trp(i, i + 1, 2.0));
    triples.push_back(Trp(i, i + 2, -1.0));
  }

  unique_ptr<SpMat> L(new SpMat(static_cast<SpMatId>(r_dim), static_cast<SpMatId>(dim)));
  L->setFromTriplets(triples.begin(), triples.end());
  return L;
}

unique_ptr<SpMat>
MakeFiniteDifferenceMatWithBoundary(size_t const& dim)
{
  vector<Trp> triples;
  triples.reserve(3 * dim);

  triples.push_back(Trp(0, 0, 1.0));
  for (auto i = 1ul, n = dim - 1; i < n; ++i)
  {
    triples.push_back(Trp(i, i - 1, -1.0));
    triples.push_back(Trp(i, i, 2.0));
    triples.push_back(Trp(i, i + 1, -1.0));
  }
  triples.push_back(Trp(dim - 1, dim - 1, 1.0));

  unique_ptr<SpMat> L(new SpMat(static_cast<SpMatId>(dim), static_cast<SpMatId>(dim)));
  L->setFromTriplets(triples.begin(), triples.end());
  // MatNxN dense = *L;
  // std::cout << "Boundary Prior (C1)" << std::endl << dense << std::endl;
  return L;
}

unique_ptr<SpMat>
MakeFiniteDifferenceMatWithC2Boundary(size_t const& dim)
{
  vector<Trp> triples;
  triples.reserve(3 * dim);

  triples.push_back(Trp(0, 0, 1.0));
  triples.push_back(Trp(1, 1, 1.0));
  for (auto i = 1ul, n = dim - 1; i < n; ++i)
  {
    triples.push_back(Trp(i + 1, i - 1, -1.0));
    triples.push_back(Trp(i + 1, i, 2.0));
    triples.push_back(Trp(i + 1, i + 1, -1.0));
  }
  triples.push_back(Trp(dim, dim - 2, 1.0));
  triples.push_back(Trp(dim + 1, dim - 1, 1.0));

  unique_ptr<SpMat> L(new SpMat(static_cast<SpMatId>(dim + 2), static_cast<SpMatId>(dim)));
  L->setFromTriplets(triples.begin(), triples.end());
  // MatNxN dense = *L;
  // std::cout << "Boundary Prior (C2)" << std::endl << dense << std::endl;
  return L;
}

}  // namespace ML
