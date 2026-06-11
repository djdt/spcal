#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <tbb/tbb.h>

namespace py = pybind11;

py::array_t<float> spectra(py::array_t<float> &values,
                           py::array_t<long> &regions, const bool mean) {
  py::buffer_info vbuf = values.request();
  py::buffer_info rbuf = regions.request();
  bool mean_flag = false;

  if (vbuf.ndim != 2)
    throw std::runtime_error("values must have 2 dim");
  if (rbuf.ndim != 2 || rbuf.shape[1] != 2)
    throw std::runtime_error("regions must have shape (n, 2)");

  auto spectra = py::array_t<float>({rbuf.shape[0], vbuf.shape[1]});
  spectra[py::make_tuple(py::ellipsis())] = 0.f;

  auto v = values.unchecked<2>();
  auto r = regions.unchecked<2>();
  auto s = spectra.mutable_unchecked<2>();

  tbb::parallel_for(py::ssize_t(0), rbuf.shape[0], [&](py::ssize_t i) {
    for (py::ssize_t j = 0; j < vbuf.shape[1]; ++j) {
      float sum = 0.f;
      int count = 0;
      for (py::ssize_t k = r(i, 0); k < r(i, 1); ++k) {
        float val = v(k, j);
        if (!std::isnan(val)) {
          sum += val;
          count += 1;
        }
      }
      if (mean && count != 0)
        sum /= count;
      s(i, j) = sum;
    }
  });

  return spectra;
}

void init_spectra(py::module_ &mod) {
  mod.def("spectra", &spectra, "Sum between regions.", py::arg(), py::arg(),
          py::arg("mean") = true);
}
