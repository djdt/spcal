#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <execution>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <ranges>

namespace py = pybind11;
/* Based off of the scipy implementation
 * https://github.com/scipy/scipy/blob/v1.9.3/scipy/cluster/_hierarchy.pyx */

inline int condensed_index(int i, int j, int n) {
  if (i < j)
    return n * i - (i * (i + 1) / 2) + (j - i - 1);
  else
    return n * j - (j * (j + 1) / 2) + (i - j - 1);
}

py::array_t<double> pairwise_euclidean(py::array_t<double> X) {
  if (X.ndim() != 2)
    throw std::runtime_error("X must have 2 dims");

  py::ssize_t n = X.shape(0), m = X.shape(1);
  auto dists = py::array_t<double>(n * (n - 1) / 2);

  auto x = X.unchecked<2>();
  auto d = dists.mutable_unchecked<1>();

  auto idx = std::vector<int>(n);
  std::iota(idx.begin(), idx.end(), 0);
  // auto idx = std::ranges::views::iota(0, static_cast<int>(n));
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](int i) {
    for (ssize_t j = i + 1; j < n; ++j) {
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

void label(py::array_t<int> &Z, int n) {
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

py::tuple mst_linkage(py::array_t<double> Dists, int n) {

  auto dists = Dists.unchecked<1>();

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

    for (int j = 0; j < n; ++j) {
      if (merged[j])
        continue;
      double dist = dists(condensed_index(x, j, n));
      if (min_dists[j] > dist) {
        min_dists[j] = dist;
      }
      if (min_dists[j] < min) {
        min = min_dists[j];
        y = j;
      }
    }

    z1[i] = x;
    z2[i] = y;
    zd_idx[i].first = min;
    zd_idx[i].second = i;
    x = y;
  }

  std::sort(std::execution::par_unseq, zd_idx.begin(), zd_idx.end(),
            [](std::pair<double, int> a, std::pair<double, int> b) {
              return a.first < b.first;
            });
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

py::array_t<int> cluster_by_distance(py::array_t<int> Z, py::array_t<double> ZD,
                                     double cluster_dist) {
  if (Z.ndim() != 2 || Z.shape(1) != 2)
    throw std::runtime_error("Z must have shape (n, 2)");
  if (Z.shape(0) != ZD.shape(0))
    throw std::runtime_error("ZD must have same length as first dim of Z");

  py::ssize_t n = Z.shape(0) + 1;

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
  auto Tbuf = T.request();
  std::memset(Tbuf.ptr, 0, Tbuf.size * Tbuf.itemsize);
  auto t = T.mutable_unchecked<1>();

  std::fill(std::execution::par_unseq, visited.begin(), visited.end(), false);

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

py::array_t<int> maxima(py::array_t<double> values, py::array_t<int> regions) {
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
    for (py::ssize_t j = r(i, 0) + 1; j < r(i, 1); ++j) {
      if (v[j] > v[max_idx]) {
        max_idx = j;
      }
    }
    m[i] = max_idx;
  }
  return argmax;
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
}
