#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <ranges>
#include <vector>

namespace py = pybind11;

void init_clustering(py::module_ &mod);
void init_extraction(py::module_ &mod);
void init_dists(py::module_ &mod);
void init_detection(py::module_ &mod);

PYBIND11_MODULE(spcalext, mod) {
  mod.doc() = "C++/pybind11 extension module for SPCal.";

  auto mod_cluster =
      mod.def_submodule("clustering", "agglomerative clustering extension");
  init_clustering(mod_cluster);

  auto mod_extraction =
      mod.def_submodule("extraction", "cpln parameter extraction");
  init_extraction(mod_extraction);

  auto mod_detection =
      mod.def_submodule("detection", "particle detection extension");
  init_detection(mod_detection);

  auto mod_dists = mod.def_submodule("dists", "poisson distribution extension");
  init_dists(mod_dists);
}
