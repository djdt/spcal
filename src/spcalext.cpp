#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>
#include <ranges>
#include <thread>
#include <vector>

namespace py = pybind11;
/* Based off of the scipy implementation
 * https://github.com/scipy/scipy/blob/v1.9.3/scipy/cluster/_hierarchy.pyx */

inline py::ssize_t condensed_index(const py::ssize_t i, py::ssize_t const j,
                                   const py::ssize_t n) {
  if (i < j)
    return n * i - (i * (i + 1) / 2) + (j - i - 1);
  else
    return n * j - (j * (j + 1) / 2) + (i - j - 1);
}

py::array_t<double> pairwise_euclidean(const py::array_t<double> &X) {
  if (X.ndim() != 2)
    throw std::runtime_error("X must have 2 dims");

  py::ssize_t n = X.shape(0), m = X.shape(1);
  auto dists = py::array_t<double>(n * (n - 1) / 2);

  auto x = X.unchecked<2>();
  auto d = dists.mutable_unchecked<1>();

  auto idx = std::ranges::views::iota(static_cast<py::ssize_t>(0), n);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [&](py::ssize_t i) {
                  for (py::ssize_t j = i + 1; j < n; ++j) {
                    double sum = 0.0;
                    for (py::ssize_t k = 0; k < m; ++k) {
                      double dist = x(i, k) - x(j, k);
                      sum += dist * dist;
                    }
                    d(condensed_index(i, j, n)) = std::sqrt(sum);
                  }
                });
  return dists;
}

inline int find_root(std::vector<int> &parents, int x) {
  int p = x;
  while (parents[x] != x) {
    x = parents[x];
  }
  while (parents[p] != x) {
    p = parents[p];
    parents[p] = x;
  }
  return x;
}

void label(py::array_t<int> &Z, const int n) {
  if (Z.ndim() != 2 || Z.shape(1) != 2)
    throw std::runtime_error("Z must have shape (n, 2)");

  auto z = Z.mutable_unchecked<2>();

  auto parents = std::vector<int>(2 * n - 1);
  std::iota(parents.begin(), parents.end(), 0);
  auto sizes = std::vector<int>(2 * n - 1, 1);

  for (int i = 0; i < n - 1; ++i) {
    int x_root = find_root(parents, z(i, 0));
    int y_root = find_root(parents, z(i, 1));
    if (x_root < y_root) {
      z(i, 0) = x_root;
      z(i, 1) = y_root;
    } else {
      z(i, 0) = y_root;
      z(i, 1) = x_root;
    }
    // merge roots
    sizes[n + i] = sizes[x_root] + sizes[y_root];
    parents[x_root] = n + i;
    parents[y_root] = n + i;
  }
}

py::tuple mst_linkage(const py::array_t<double> &dists_array, const int n) {

  auto dists = dists_array.unchecked<1>();

  auto z1 = std::vector<int>(n - 1);
  auto z2 = std::vector<int>(n - 1);
  auto zd_idx = std::vector<std::pair<double, int>>(n - 1);

  auto merged = std::vector<bool>(n, false);
  auto min_dists =
      std::vector<double>(n, std::numeric_limits<double>::infinity());

  int x = 0, y = 0;
  for (int i = 0; i < n - 1; ++i) {
    double min = std::numeric_limits<double>::infinity();
    merged[x] = true;

    auto jdx = std::ranges::views::iota(0, n);
    std::for_each(std::execution::seq, jdx.begin(), jdx.end(), [&](int j) {
      if (merged[j])
        return;
      double dist = dists(condensed_index(x, j, n));
      if (min_dists[j] > dist) {
        min_dists[j] = dist;
      }
      if (min_dists[j] < min) {
        min = min_dists[j];
        y = j;
      }
    });

    z1[i] = x;
    z2[i] = y;
    zd_idx[i].first = min;
    zd_idx[i].second = i;
    x = y;
  }

  // par_unseq causes crash in Pyinstaller created exe
#ifdef SEQSORT
#pragma message("Building ext using sequential sort.")
  auto sortexc = std::execution::seq;
#else
  auto sortexc = std::execution::par_unseq;
#endif
  std::sort(sortexc, zd_idx.begin(), zd_idx.end(),
            [](const std::pair<double, int> &a,
               const std::pair<double, int> &b) { return a.first < b.first; });
  auto Z = py::array_t<int>({n - 1, 2});
  auto z = Z.mutable_unchecked<2>();
  auto ZD = py::array_t<double>(n - 1);
  auto zd = ZD.mutable_unchecked<1>();

  for (int i = 0; i < n - 1; ++i) {
    z(i, 0) = z1[zd_idx[i].second];
    z(i, 1) = z2[zd_idx[i].second];
    zd(i) = zd_idx[i].first;
  }

  label(Z, n);

  return py::make_tuple(Z, ZD);
}

py::array_t<int> cluster_by_distance(const py::array_t<int> &Z,
                                     const py::array_t<double> &ZD,
                                     const double cluster_dist) {
  if (Z.ndim() != 2 || Z.shape(1) != 2)
    throw std::runtime_error("Z must have shape (n, 2)");
  if (Z.shape(0) != ZD.shape(0))
    throw std::runtime_error("ZD must have same length as first dim of Z");

  int n = static_cast<int>(Z.shape(0)) + 1;

  auto max_dist = std::vector<double>(n - 1);
  auto nodes = std::vector<int>(n);
  auto visited = std::vector<bool>(n * 2, false);

  auto z = Z.unchecked<2>();
  auto zd = ZD.unchecked<1>();

  // Get the maximum distance for each cluster
  int k = 0;
  nodes[0] = 2 * n - 2;
  while (k >= 0) {
    int root = nodes[k] - n;
    int i = z(root, 0);
    int j = z(root, 1);

    if (i >= n && !visited[i]) {
      visited[i] = true;
      nodes[++k] = i;
      continue;
    }
    if (j >= n && !visited[j]) {
      visited[j] = true;
      nodes[++k] = j;
      continue;
    }

    double max = zd(root);

    if (i >= n && max_dist[i - n] > max)
      max = max_dist[i - n];
    if (j >= n && max_dist[j - n] > max)
      max = max_dist[j - n];
    max_dist[root] = max;

    k -= 1;
  }

  auto T = py::array_t<int>(n);
  auto t = T.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    t(i) = 0;
  }

  // std::execution::par fails to fill
  std::fill(std::execution::seq, visited.begin(), visited.end(), false);

  // Cluster nodes by distance
  int cluster_leader = -1, cluster_number = 0;
  nodes[0] = 2 * n - 2;
  k = 0;
  while (k >= 0) {
    int root = nodes[k] - n;
    int i = z(root, 0);
    int j = z(root, 1);

    if (cluster_leader == -1 && max_dist[root] <= cluster_dist) {
      cluster_leader = root;
      cluster_number += 1;
    }

    if (i >= n && !visited[i]) {
      visited[i] = true;
      nodes[++k] = i;
      continue;
    }
    if (j >= n && !visited[j]) {
      visited[j] = true;
      nodes[++k] = j;
      continue;
    }
    if (i < n) {
      if (cluster_leader == -1)
        cluster_number += 1;
      t(i) = cluster_number;
    }
    if (j < n) {
      if (cluster_leader == -1)
        cluster_number += 1;
      t(j) = cluster_number;
    }
    if (cluster_leader == root)
      cluster_leader = -1;
    k -= 1;
  }
  return T;
}

py::array_t<int> maxima(const py::array_t<double> &values,
                        const py::array_t<int> &regions) {
  py::buffer_info vbuf = values.request(), rbuf = regions.request();

  if (vbuf.ndim != 1)
    throw std::runtime_error("values must have 1 dim");
  if (rbuf.ndim != 2 || rbuf.shape[1] != 2)
    throw std::runtime_error("regions must have shape (n, 2)");

  auto argmax = py::array_t<int>(rbuf.shape[0]);

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

double poisson_quantile(double q, double lam) {
  double k = 0.0;
  long double pdf = std::exp(static_cast<long double>(-lam));
  long double cdf = pdf;

  while (cdf < q) {
    k += 1.0;
    pdf *= lam / k;
    if (!std::isnormal(pdf)) {
      std::cerr << "poisson_qunatile: error calculating quantile for lambda="
                << lam << std::endl;
      break;
    }
    cdf += pdf;
  }
  return k;
}

py::tuple peak_prominence(const py::array_t<double> &values,
                          const py::array_t<int> &indicies,
                          const double minimum) {

  if (values.ndim() != 1)
    throw std::runtime_error("values must have 1 dim");
  if (indicies.ndim() != 1)
    throw std::runtime_error("indicies must have 1 dim");

  auto m = values.shape(0);
  auto n = indicies.shape(0);

  auto prom_array = py::array_t<double>(n);
  auto left_array = py::array_t<int>(n);
  auto right_array = py::array_t<int>(n);

  auto y = values.unchecked<1>();
  auto idx = indicies.unchecked<1>();

  auto lefts = left_array.mutable_unchecked<1>();
  auto rights = right_array.mutable_unchecked<1>();
  auto proms = prom_array.mutable_unchecked<1>();

  for (py::ssize_t i = 0; i < n; ++i) {
    double peak_height = y[idx[i]];

    int left = idx[i];
    int left_minima = left;
    int iter = 0;
    while (left > 0 && y[left - 1] <= peak_height && y[left] > minimum) {
      left -= 1;
      if (y[left] <= y[left_minima]) {
        left_minima = left;
      }
    }
    int right = idx[i];
    int right_minima = right;
    iter = 0;
    while (right < m - 1 && y[right + 1] <= peak_height && y[right] > minimum) {
      right += 1;
      if (y[right] <= y[right_minima]) {
        right_minima = right;
      }
    }
    lefts[i] = left_minima;
    rights[i] = right_minima;
    proms[i] = peak_height - std::max(y[left_minima], y[right_minima]);
  }
  return py::make_tuple(prom_array, left_array, right_array);
}

py::array_t<int> label_regions(const py::array_t<int> &regions_array,
                               const py::ssize_t size) {
  if (regions_array.ndim() != 2 || regions_array.shape(1) != 2) {
    throw std::runtime_error("regions must have shape (N, 2)");
  }

  py::array_t<int> label_array(size);
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

py::array_t<int> combined_regions(const py::list &regions_list) {

  std::vector<int> lefts;
  std::vector<int> rights;

  std::vector<py::detail::unchecked_reference<int, 2>> regions;
  regions.reserve(regions_list.size());
  std::vector<int> current_pos(regions_list.size());
  for (int i = 0; i < current_pos.size(); ++i) {
    regions.push_back(
        static_cast<py::array_t<int>>(regions_list[i]).unchecked<2>());
    current_pos[i] = 0;
  }

  int left = std::numeric_limits<int>::max();
  int right = 0;
  while (true) {
    // leftmost region
    int leftmost = 0;
    for (int i = 0; i < current_pos.size(); ++i) {
      if (regions[i](current_pos[i], 0) < left) {
        left = regions[i](current_pos[i], 0);
        right = regions[i](current_pos[i], 0);
        leftmost = i;
      }
    }
    bool changed = false;
    while (!changed) {
      for (int i = 0; i < current_pos.size(); ++i) {
        if (regions[i](current_pos[i], 1) > right) {
          right = regions[i](current_pos[i]++, 0);
          changed = true;
        }
      }
    }
  }
}

PYBIND11_MODULE(spcalext, mod) {
  mod.doc() = "extension module for SPCal.";

  mod.def("pairwise_euclidean", &pairwise_euclidean,
          "Calculates the euclidean distance for an array");
  mod.def("mst_linkage", &mst_linkage,
          "Return the minimum-spanning-tree linkage.");
  mod.def("cluster_by_distance", &cluster_by_distance,
          "Cluster using MST linkage.");
  mod.def("maxima", &maxima,
          "Calculates to maxima between pairs of start and end positions.");
  mod.def("poisson_quantile", &poisson_quantile,
          "Quantile (k) for a given q and lambda.");
  mod.def("peak_prominence", &peak_prominence,
          "Calculate the peak prominence at given indicies.");
  mod.def("label_regions", &label_regions,
          "Label regions 1 to size, points outside all regions are 0.");
}
