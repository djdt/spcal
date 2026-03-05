#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <tbb/tbb.h>

#include <cmath>

namespace py = pybind11;

py::array_t<double>
extract_cpln_parameters(const py::array_t<double> &signals_array,
                        const py::array_t<bool> &mask_array) {
  /*
   * Returns the Compound-Poisson-lognormal parameters.
   * See python function for more complete doc.
   *
   * @param signals array shape (samples, features)
   * @param mask of valid values, same shape as signals
   * @returns array of [..., (lam, mu, sigma)]
   */
  py::buffer_info sbuf = signals_array.request();
  py::buffer_info mbuf = mask_array.request();

  if (sbuf.ndim != 2)
    std::runtime_error("signals must have ndim of 2");
  if ((mbuf.shape[0] != sbuf.shape[0]) or (mbuf.shape[1] != sbuf.shape[1]))
    std::runtime_error("mask must be same shape as signals");

  py::array_t<double> params_array({sbuf.shape[1], py::ssize_t(3)});

  auto signals = signals_array.unchecked<2>();
  auto mask = mask_array.unchecked<2>();
  auto params = params_array.mutable_unchecked<2>();

  tbb::parallel_for(py::ssize_t(0), sbuf.shape[1], [&](py::ssize_t j) {
    long zeros = 0, valid = 0;
    double sum = 0.0;
    for (py::ssize_t i = 0; i < sbuf.shape[0]; ++i) {
      if (mask(i, j)) {
        valid++;
        if (signals(i, j) == 0.0) {
          zeros++;
        } else {
          sum += signals(i, j);
        }
      }
    }
    double lam =
        -std::log(static_cast<double>(zeros) / static_cast<double>(valid));

    auto mean = sum / static_cast<double>(valid);
    auto var = 0.0;
    for (py::ssize_t i = 0; i < sbuf.shape[0]; ++i) {
      if (mask(i, j)) {
        var += std::pow(signals(i, j) - mean, 2);
      }
    }
    var /= static_cast<double>(valid);

    auto ex = mean / lam;
    auto ex2 = var / lam;

    params(j, 0) = lam;
    params(j, 1) = std::log((ex * ex) / std::sqrt(ex2));
    params(j, 2) = std::sqrt(std::log(ex2 / (ex * ex)));
  });
  return params_array;
}

// while it would be nice to have the iterative version here, that would also
// require the lookup table to be converted

void init_extraction(py::module_ &mod) {
  mod.def("extract_cpln_parameters", &extract_cpln_parameters,
          "Extractions Compound-Poisson-lognormal parameters from data.");
}
