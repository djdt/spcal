#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <tbb/tbb.h>

#include <cmath>
#include <vector>

namespace py = pybind11;

py::array_t<bool> local_maxima(const py::array_t<double> &values_array) {
  py::buffer_info vbuf = values_array.request();
  if (vbuf.ndim != 1)
    throw std::runtime_error("values must have 1 dim");

  py::array_t<bool> maxima_array = py::array_t<bool>(vbuf.shape);

  auto values = values_array.unchecked<1>();
  auto maxima = maxima_array.mutable_unchecked<1>();

  maxima(0) = values(0) >= values(1);
  maxima(vbuf.shape[0] - 1) =
      values(vbuf.shape[0] - 1) > values(vbuf.shape[0] - 2);

  tbb::parallel_for(py::ssize_t(1), vbuf.shape[0] - 1, [&](py::ssize_t i) {
    double val = values(i);
    maxima(i) = (val > values(i - 1)) && (val >= values(i + 1));
  });

  return maxima_array;
}

py::array_t<long> maxima_between(const py::array_t<double> &values,
                                 const py::array_t<long> &regions) {
  /*
   * The maxima of values between pairs of start and end indicies.
   *
   * @param values array of values
   * @param regions location of start, end of shape (n, 2)
   * @returns maximum values
   */

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

  tbb::parallel_for(py::ssize_t(0), rbuf.shape[0], [&](py::ssize_t i) {
    int max_idx = r(i, 0);
    for (int j = r(i, 0) + 1; j < r(i, 1); ++j) {
      if (v[j] > v[max_idx]) {
        max_idx = j;
      }
    }
    m[i] = max_idx;
  });
  return argmax;
}

py::tuple peak_prominence(const py::array_t<double> &values,
                          const py::array_t<long> &indicies,
                          const long max_width, const py::object &min_base) {
  /*
   * Find the peak prominences at given indicies.
   *
   * @param values array of values
   * @param indicies location of peaks
   * @param max_width the maximum distance to search
   * @param min_base the minimum value at peak base, array or float
   * @returns prominence, left indicies, right indices
   */

  py::buffer_info vbuf = values.request();
  py::buffer_info ibuf = indicies.request();

  if (vbuf.ndim != 1)
    throw std::runtime_error("values must have 1 dim");
  if (ibuf.ndim != 1)
    throw std::runtime_error("indicies must have 1 dim");

  // default for min_value if double
  bool min_is_array = false;
  double min_value = 0.0;
  py::array_t<double> min_array;

  if (py::isinstance<py::array>(min_base)) {
    min_is_array = true;
    min_array =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            min_base);
    if (min_array.size() != vbuf.size) { // check shape
      throw std::runtime_error("min_base must be same size as values");
    }
  } else {
    try {
      min_value = min_base.cast<double>();
    } catch (const std::exception &) {
      throw std::runtime_error("min_base must be array or float");
    }
  }
  auto prom_array = py::array_t<double>(ibuf.shape[0]);
  auto left_array = py::array_t<long>(ibuf.shape[0]);
  auto right_array = py::array_t<long>(ibuf.shape[0]);

  auto y = values.unchecked<1>();
  auto idx = indicies.unchecked<1>();
  auto min = min_array.unchecked<1>();

  auto lefts = left_array.mutable_unchecked<1>();
  auto rights = right_array.mutable_unchecked<1>();
  auto proms = prom_array.mutable_unchecked<1>();

  tbb::parallel_for(py::ssize_t(0), ibuf.shape[0], [&](py::ssize_t i) {
    double peak_height = y[idx[i]];

    int left = idx[i];
    int left_minima = left;
    while (left > 0 && y(left - 1) <= peak_height &&
           idx[i] - left < max_width &&
           y(left) > (min_is_array ? min(left) : min_value)) {
      left -= 1;
      if (y(left) < y(left_minima)) {
        left_minima = left;
      }
    }
    int right = idx[i];
    int right_minima = right;
    while (right < vbuf.shape[0] - 1 && y(right + 1) <= peak_height &&
           right - idx[i] < max_width &&
           y(right) > (min_is_array ? min(right) : min_value)) {
      right += 1;
      if (y(right) < y(right_minima)) {
        right_minima = right;
      }
    }
    lefts[i] = left_minima;
    rights[i] = right_minima;
    proms[i] = peak_height - std::max(y(left_minima), y(right_minima));
  });
  return py::make_tuple(prom_array, left_array, right_array);
}

py::tuple split_peaks(const py::array_t<double> &prominence_array,
                      const py::array_t<long> &left_array,
                      const py::array_t<long> &right_array,
                      const double prominence_required) {
  /* Split overlaping peaks that are a fraction the max prominence.
   * @param prominence
   * @param left
   * @param right
   * @param prominence_required fraction of max overlap prominence to split
   *
   * @returns new left and right arrays
   */
  py::buffer_info pbuf = prominence_array.request();
  py::buffer_info lbuf = left_array.request();
  py::buffer_info rbuf = right_array.request();

  std::vector<long> *split_left = new std::vector<long>;
  std::vector<long> *split_right = new std::vector<long>;
  split_left->reserve(pbuf.size);
  split_right->reserve(pbuf.size);

  auto prom = prominence_array.unchecked<1>();
  auto lefts = left_array.unchecked<1>();
  auto rights = right_array.unchecked<1>();

  py::ssize_t i = 0;
  while (i < pbuf.shape[0]) {

    double max_prom = prom(i);
    // find overlaps and max prominence
    py::ssize_t j = i + 1;
    py::ssize_t k = i;
    while ((j < lbuf.shape[0]) && ((lefts(j) < rights(k++)))) {
      max_prom = std::max(max_prom, prom(j++));
    }
    // early exit for single peak
    if (j == i + 1) {
      split_left->push_back(lefts(i));
      split_right->push_back(rights(i));
    } else {
      // limit to unique peaks with required prominence
      std::vector<py::ssize_t> valid;
      for (py::ssize_t k = i; k < j; ++k) {
        if ((k > 0) && (lefts(k) == lefts(k - 1)) &&
            ((rights(k) == rights(k - 1)))) {
          continue;
        }
        if (prom(k) >= max_prom * prominence_required) {
          valid.push_back(k);
        }
      }

      long current_left = lefts(0);
      long current_right = rights(valid[0]);

      split_left->push_back(current_left);
      for (py::ssize_t k = 1; k < valid.size(); ++k) {
        py::ssize_t idx = valid[k];
        if (lefts(idx) > current_left) {
          current_left = lefts(idx);
        } else {
          current_left = std::min(rights(idx), current_right);
          current_right = std::max(rights(idx), current_right);
        }
        split_left->push_back(current_left);
        split_right->push_back(current_left);
      }
      split_right->push_back(current_right);
    }
    i = j;
  }
  split_left->shrink_to_fit();
  split_right->shrink_to_fit();
  // create numpy array to return
  py::capsule free_when_done_left(split_left, [](void *f) {
    delete reinterpret_cast<std::vector<long> *>(f);
  });
  py::capsule free_when_done_right(split_right, [](void *f) {
    delete reinterpret_cast<std::vector<long> *>(f);
  });

  return py::make_tuple(
      py::array_t<long>({split_left->size()}, {sizeof(long)},
                        split_left->data(), free_when_done_left),
      py::array_t<long>({split_right->size()}, {sizeof(long)},
                        split_right->data(), free_when_done_right));
}

py::array_t<long> label_regions(const py::array_t<long> &regions_array,
                                const py::size_t size) {
  py::buffer_info rbuf = regions_array.request();
  if (rbuf.ndim != 2 || rbuf.shape[1] != 2) {
    throw std::runtime_error("regions must have shape (N, 2)");
  }

  py::array_t<long> label_array(size);
  label_array[py::make_tuple(py::ellipsis())] = 0;
  auto n = rbuf.shape[0];

  auto regions = regions_array.unchecked<2>();
  auto labels = label_array.mutable_unchecked<1>();

  tbb::parallel_for(py::ssize_t(0), n, [&](py::ssize_t i) {
    for (py::ssize_t j = regions(i, 0); j < regions(i, 1); ++j) {
      labels[j] = i + 1;
    }
  });
  return label_array;
}

py::array_t<long> combine_regions(const py::list &regions_list,
                                  const int allowed_overlap) {
  std::vector<py::detail::unchecked_reference<long, 2>> regions;
  std::vector<py::ssize_t> indicies(regions_list.size());

  py::ssize_t total_peaks = 0;
  for (const py::handle &obj : regions_list) {
    auto array =
        py::array_t<long, py::array::c_style | py::array::forcecast>::ensure(
            obj);
    py::buffer_info abuf = array.request();
    if (!array || abuf.ndim != 2 || abuf.shape[1] != 2) {
      throw std::runtime_error("invalid shape or ndim");
    }
    regions.push_back(array.unchecked<2>());
    total_peaks += abuf.shape[0];
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

          if (e <= right) { // passed
            indicies[i] += 1;
            changed = true;
          } else if (s < right - allowed_overlap) { // overlap
            right = e;
            indicies[i] += 1;
            changed = true;
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
  if (allowed_overlap > 0 && combined->size() > 2) {
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
  mod.def("local_maxima", &local_maxima, "Boolean array of local maxima.");
  mod.def("maxima_between", &maxima_between,
          "Calculates the maxima between pairs of start and end positions.");
  mod.def("peak_prominence", &peak_prominence,
          "Calculate the peak prominence at given indicies.", py::arg(),
          py::arg(), py::arg("max_width") = 100, py::arg("min_base") = 0.0);
  mod.def("label_regions", &label_regions,
          "Label regions 1 to size, points outside all regions are 0.");
  mod.def("combine_regions", &combine_regions,
          "Combine a list of regions, merging overlaps.", py::arg(),
          py::arg("allowed_overlap") = 0);
  mod.def("split_peaks", &split_peaks,
          "Split overlapping peaks with some fraction of the maximum overlap "
          "prominence.");
}
