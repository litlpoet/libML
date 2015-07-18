#include "MLCore/mathmatrixpredefined.h"

namespace ML {

bool MakeFiniteDiffernceMat(const int& dim, SpMat* L) {
  const int r_dim = dim - 2;
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

}  // namespace ML
