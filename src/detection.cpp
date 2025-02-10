#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <ranges>
#include <vector>

#include <iostream>

namespace py = pybind11;
/* Based off of the scipy implementation
 * https://github.com/scipy/scipy/blob/v1.9.3/scipy/cluster/_hierarchy.pyx */

py::array_t<long> maxima(const py::array_t<double> &values,
                         const py::array_t<long> &regions) {
  py::buffer_info vbuf = values.request();
  py::buffer_info rbuf = regions.request();

  if (vbuf.ndim != 1)
    throw std::runtime_error("values must have 1 dim");
  if (rbuf.ndim != 2 || rbuf.shape[1] != 2)
    throw std::runtime_error("regions must have shape (n, 2)");

  auto argmax = py::array_t<long>(rbuf.shape[0]);

  auto v = values.unchecked<1>();
  auto r = regions.unchecked<2>();
  auto m = argmax.mutable_unchecked<1>();

  for (py::ssize_t i = 0; i < r.shape(0); ++i) {
    int max_idx = r(i, 0);
    for (int j = r(i, 0) + 1; j < r(i, 1); ++j) {
      if (v[j] > v[max_idx]) {
        max_idx = j;
      }
    }
    m[i] = max_idx;
  }
  return argmax;
}

py::tuple peak_prominence(const py::array_t<double> &values,
                          const py::array_t<long> &indicies,
                          const long max_width, const double min_value) {
  /*
   * Find the peak prominences at given indicies.
   *
   * @param values array of values
   * @param indicies location of peaks
   * @param max_width the maximum distance to search
   * @param min_value the minimum value at peak base
   * @returns prominence, left indicies, right indices
   */

  if (values.ndim() != 1)
    throw std::runtime_error("values must have 1 dim");
  if (indicies.ndim() != 1)
    throw std::runtime_error("indicies must have 1 dim");

  auto m = values.shape(0);
  auto n = indicies.shape(0);

  auto prom_array = py::array_t<double>(n);
  auto left_array = py::array_t<long>(n);
  auto right_array = py::array_t<long>(n);

  auto y = values.unchecked<1>();
  auto idx = indicies.unchecked<1>();

  auto lefts = left_array.mutable_unchecked<1>();
  auto rights = right_array.mutable_unchecked<1>();
  auto proms = prom_array.mutable_unchecked<1>();

  for (py::ssize_t i = 0; i < n; ++i) {
    double peak_height = y[idx[i]];

    int left = idx[i];
    int left_minima = left;
    while (left > 0 && y(left - 1) <= peak_height && y(left) > min_value &&
           idx[i] - left < max_width) {
      left -= 1;
      if (y(left) < y(left_minima)) {
        left_minima = left;
      }
    }
    int right = idx[i];
    int right_minima = right;
    while (right < m - 1 && y(right + 1) <= peak_height &&
           y(right) > min_value && right - idx[i] < max_width) {
      right += 1;
      if (y(right) < y(right_minima)) {
        right_minima = right;
      }
    }
    lefts[i] = left_minima;
    rights[i] = right_minima;
    proms[i] = peak_height - std::max(y(left_minima), y(right_minima));
  }
  return py::make_tuple(prom_array, left_array, right_array);
}

py::array_t<long> label_regions(const py::array_t<long> &regions_array,
                                const py::size_t size) {
  if (regions_array.ndim() != 2 || regions_array.shape(1) != 2) {
    throw std::runtime_error("regions must have shape (N, 2)");
  }

  py::array_t<long> label_array(size);
  label_array[py::make_tuple(py::ellipsis())] = 0;
  auto n = regions_array.shape(0);

  auto regions = regions_array.unchecked<2>();
  auto labels = label_array.mutable_unchecked<1>();

  int region = 1;
  for (py::ssize_t i = 0; i < n; ++i) {
    for (py::ssize_t j = regions(i, 0); j < regions(i, 1); ++j) {
      labels[j] = region;
    }
    region++;
  }
  return label_array;
}

py::array_t<long> combine_regions(const py::list &regions_list,
                                  const int allowed_overlap) {

  std::vector<py::detail::unchecked_reference<long, 2>> regions;
  std::vector<py::ssize_t> indicies(regions_list.size());

  py::ssize_t total_peaks = 0;
  for (const py::handle &obj : regions_list) {
    auto array = py::array_t < long,
         py::array::c_style | py::array::forcecast > ::ensure(obj);
    if (!array || array.ndim() != 2 || array.shape(1) != 2) {
      throw std::runtime_error("invalid shape or ndim");
    }
    regions.push_back(array.unchecked<2>());
    total_peaks += array.shape(0);
  }

  std::vector<long> *combined = new std::vector<long>;
  combined->reserve(total_peaks * 2);

  int iter = 0;
  while (iter++ < total_peaks) {
    long left = std::numeric_limits<long>::max();
    long right = 0;
    py::ssize_t leftmost_idx = 0;
    long s, e;

    // find leftmost region
    bool finished = true;
    for (size_t i = 0; i < indicies.size(); ++i) {
      if (indicies[i] < regions[i].shape(0)) {
        s = regions[i](indicies[i], 0);
        e = regions[i](indicies[i], 1);

        if (s < left) {
          left = s;
          right = e;
          leftmost_idx = i;
        }
        finished = false;
      }
    }
    if (finished) {
      break;
    }
    indicies[leftmost_idx] += 1;

    // find overlapping regions
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t i = 0; i < indicies.size(); ++i) {
        if (indicies[i] < regions[i].shape(0)) {
          s = regions[i](indicies[i], 0);
          e = regions[i](indicies[i], 1);

          if (s < right - allowed_overlap && e >= right) { // overlapping
            right = e;
            indicies[i] += 1;
            changed = true;
          } else if (e <= right) { // region is passed
            indicies[i] += 1;
          }
        }
      }
    }
    combined->push_back(left);
    combined->push_back(right);
  }

  combined->shrink_to_fit();

  auto &_comb_ref = *combined;
  // deal with any overlaps created by allowed_overlaps
  if (allowed_overlap > 0) {
    // check each pair left right overlap , and  save ,mid point
    for (size_t i = 0; i < _comb_ref.size() - 2; i += 2) {
      // (l, r), (lw, r2)
      if (_comb_ref[i + 1] > _comb_ref[i + 2]) { // r1 > l2
        long mid = (_comb_ref[i + 1] + _comb_ref[i + 2]) / 2 + 1;
        _comb_ref[i + 1] = mid; // r1 => mid
        _comb_ref[i + 2] = mid; // l2 => mid
      }
    }
  }

  // create numpy array to return
  std::array<py::ssize_t, 2> shape = {0, 2};
  shape[0] = combined->size() / 2;
  std::array<py::ssize_t, 2> stride = {2 * sizeof(long), sizeof(long)};

  py::capsule free_when_done(combined, [](void *f) {
    delete reinterpret_cast<std::vector<long> *>(f);
  });

  return py::array_t<long>(shape, stride, combined->data(), free_when_done);
}

void init_detection(py::module_ &mod) {
  mod.def("maxima", &maxima,
          "Calculates to maxima between pairs of start and end positions.");
  mod.def("peak_prominence", &peak_prominence,
          "Calculate the peak prominence at given indicies.", py::arg(),
          py::arg(), py::arg("max_width") = 50, py::arg("min_value") = 0.0);
  mod.def("label_regions", &label_regions,
          "Label regions 1 to size, points outside all regions are 0.");
  mod.def("combine_regions", &combine_regions,
          "Combine a list of regions, merging overlaps.", py::arg(),
          py::arg("allowed_overlap") = 0);
}
