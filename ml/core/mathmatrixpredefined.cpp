// Copyright (c) 2015 Byungkuk Choi

#include "core/mathmatrixpredefined.h"

#include <iostream>
#include <vector>

namespace ML {

bool MakeFiniteDifferenceMat(int const& dim, SpMat* L) {
  int r_dim = dim - 2;
  std::vector<Trp> triples;
  triples.reserve(3 * r_dim);

  for (auto i = 0; i < r_dim; ++i) {
    triples.push_back(Trp(i, i, -1.f));
    triples.push_back(Trp(i, i + 1, 2.f));
    triples.push_back(Trp(i, i + 2, -1.f));
  }

  L->resize(r_dim, dim);
  L->setFromTriplets(triples.begin(), triples.end());

  return true;
}

bool MakeFiniteDifferenceMatWithBoundary(int const& dim, SpMat* L) {
  std::vector<Trp> triples;
  triples.reserve(3 * dim);

  triples.push_back(Trp(0, 0, 1.0f));
  for (auto i = 1; i < dim - 1; ++i) {
    triples.push_back(Trp(i, i - 1, -1.f));
    triples.push_back(Trp(i, i, 2.f));
    triples.push_back(Trp(i, i + 1, -1.f));
  }
  triples.push_back(Trp(dim - 1, dim - 1, 1.0f));

  L->resize(dim, dim);
  L->setFromTriplets(triples.begin(), triples.end());
  // MatNxN dense = *L;
  // std::cout << "Boundary Prior (C1)" << std::endl << dense << std::endl;

  return true;
}

bool MakeFiniteDifferenceMatWithC2Boundary(int const& dim, SpMat* L) {
  std::vector<Trp> triples;
  triples.reserve(3 * dim);

  triples.push_back(Trp(0, 0, 1.0f));
  triples.push_back(Trp(1, 1, 1.0f));
  for (auto i = 1; i < dim - 1; ++i) {
    triples.push_back(Trp(i + 1, i - 1, -1.f));
    triples.push_back(Trp(i + 1, i, 2.f));
    triples.push_back(Trp(i + 1, i + 1, -1.f));
  }
  triples.push_back(Trp(dim, dim - 2, 1.0f));
  triples.push_back(Trp(dim + 1, dim - 1, 1.0f));

  L->resize(dim + 2, dim);
  L->setFromTriplets(triples.begin(), triples.end());
  // MatNxN dense = *L;
  // std::cout << "Boundary Prior (C2)" << std::endl << dense << std::endl;

  return true;
}

}  // namespace ML
