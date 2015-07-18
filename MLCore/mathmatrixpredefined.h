#ifndef _MLMATHMATRIXPREDEFINED_H_
#define _MLMATHMATRIXPREDEFINED_H_

#include <vector>
#include "MLCore/mathsparsetypes.h"

namespace ML {

bool MakeFiniteDiffernceMat(const int& dim, SpMat* L);

}  // namespace ML

#endif  // _MLMATHMATRIXPREDEFINED_H_
