#include <cmath>
#include <pybind11/pybind11.h>

namespace py = pybind11;

double poisson_quantile(const double q, const double lam) {
  double k = 0.0;
  long double pdf = std::exp(static_cast<long double>(-lam));
  long double cdf = pdf;

  while (cdf < q) {
    k += 1.0;
    pdf *= lam / k;
    if (!std::isnormal(pdf)) {
      PyErr_WarnEx(PyExc_RuntimeWarning,
                   "poisson_quantile: error calculating quantile for lambda",
                   1);
      break;
    }
    cdf += pdf;
  }
  return k;
}

void init_dists(py::module_ &mod) {
  mod.def("poisson_quantile", &poisson_quantile,
          "the quantile for a dist with the given mean");
}
