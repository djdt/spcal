#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <execution>
#include <iostream>
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

  auto idx = std::ranges::views::iota(static_cast<py::ssize_t>(0), n);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](int i)
  {
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
// inline int find_root(int *parents, int x) {
//   int p = x;
//   while (parents[x] != x)
//     x = parents[x];
//
//   while (parents[p] != x) {
//     p = parents[p];
//     parents[p] = x;
//   }
//   return x;
// }
//
// inline int merge_roots(int *parents, int *sizes, int n, int x, npy_int y)
// {
//   int size = sizes[x] + sizes[y];
//   sizes[n] = size;
//   parents[x] = n;
//   parents[y] = n;
//   return size;
// }
//
// void label(int *Z, int n) {
//   int *parents = malloc((2 * n - 1) * sizeof(int));
//   int *sizes = malloc((2 * n - 1) * sizeof(int));
//   int next = n;
//   int x, y, x_root, y_root;
//   for (int i = 0; i < 2 * n - 1; ++i) {
//     parents[i] = i;
//     sizes[i] = 1;
//   }
//
//   for (npy_intp i = 0; i < n - 1; ++i) {
//     x = Z[i * 3];
//     y = Z[i * 3 + 1];
//     x_root = find_root(parents, x);
//     y_root = find_root(parents, y);
//     if (x_root < y_root) {
//       Z[i * 3] = x_root;
//       Z[i * 3 + 1] = y_root;
//     } else {
//       Z[i * 3] = y_root;
//       Z[i * 3 + 1] = x_root;
//     }
//     Z[i * 3 + 2] = merge_roots(parents, sizes, next, x_root, y_root);
//     next += 1;
//   }
//
//   free(parents);
//   free(sizes);
// }
//
py::array_t<double> mst_linkage(py::array_t<double> dists, int n) {
  auto Z = py::array_t<int>({n - 1, 3});
  auto ZD = py::array_t<double>(n, INFINITY);

  // auto Z1 = std::vector<int>(n-1);
  // auto Z2 = std::vector<int>(n-1);
  //
  auto Z3 = Z[py::make_tuple(py::slice(0, n - 1, 1), 0)];
  auto z = Z.mutable_unchecked<3>();
  for (py::ssize_t i = 0; i < Z.shape(0); ++i) {
    z(i, 2) = static_cast<int>(i);
  }
  std::fill(std::execution::par_unseq, Z3.begin(), Z3.end(), 0);
}
// static PyObject *mst_linkage(PyObject *self, PyObject *args) {
//   PyObject *in;
//   PyArrayObject *PDarray;
//   npy_intp n;
//
//   if (!PyArg_ParseTuple(args, "On:mst_linkage", &in, &n))
//     return nullptr;
//
//   PDarray =
//       (PyArrayObject *)PyArray_FROM_OTF(in, NPY_DOUBLE,
//       NPY_ARRAY_IN_ARRAY);
//   if (!PDarray) {
//     return nullptr;
//   }
//   // m = n*(n+1)/2
//   // npy_intp n = 1 + (-1 + (int)sqrt(1 + 8 * PyArray_DIM(PDarray, 0))) /
//   2;
//
//   const double *PD = (const double *)PyArray_DATA(PDarray);
//   int *Z1 = malloc((n - 1) * sizeof(int));
//   int *Z2 = malloc((n - 1) * sizeof(int));
//   struct argsort *Z3 = malloc((n - 1) * sizeof(struct argsort));
//
//   uint8_t *M = calloc(n, sizeof(uint8_t));
//   double *D = malloc(n * sizeof(double));
//
//   // We use Z[:, 2] as M, tracking merged
//   // Init arrays (ZD = 0), D = inf
//   npy_intp i;
// #pragma omp parallel for
//   for (i = 0; i < n - 1; ++i) {
//     D[i] = INFINITY;
//     Z3[i].index = i;
//   }
//   D[n - 1] = INFINITY;
//
//   int x = 0, y = 0;
//   for (int i = 0; i < n - 1; ++i) {
//     double min = INFINITY;
//     M[x] = 1;
//
// #pragma omp parallel shared(D, PD, M)
//     {
//       double tmin = min;
//       int ty = y;
//
//       int j;
// #pragma omp for
//       for (j = 0; j < n; ++j) {
//         if (M[j] == 1)
//           continue;
//
//         double dist = PD[condensed_index(x, j, n)];
//
//         if (D[j] > dist)
//           D[j] = dist;
//         if (D[j] < tmin) {
//           ty = j;
//           tmin = D[j];
//         }
//       }
// #pragma omp critical
//       {
//         if (tmin < min) {
//           min = tmin;
//           y = ty;
//         }
//       }
//     }
//
//     Z1[i] = x;
//     Z2[i] = y;
//     Z3[i].value = min;
//     x = y;
//   }
//
//   free(M);
//   free(D);
//   Py_DECREF(PDarray);
//
//   // Sort
//   quicksort_argsort(Z3, n - 1);
//
//   PyArrayObject *Zarray, *ZDarray;
//   npy_intp Zdims[] = {n - 1, 3};
//   npy_intp ZDdims[] = {n - 1};
//   Zarray = (PyArrayObject *)PyArray_SimpleNew(2, Zdims, NPY_INT);
//   ZDarray = (PyArrayObject *)PyArray_SimpleNew(1, ZDdims, NPY_DOUBLE);
//
//   int *Z = (int *)PyArray_DATA(Zarray);
//   double *ZD = (double *)PyArray_DATA(ZDarray);
//
//   for (int i = 0; i < n - 1; ++i) {
//     Z[i * 3] = Z1[Z3[i].index];
//     Z[i * 3 + 1] = Z2[Z3[i].index];
//     ZD[i] = Z3[i].value;
//   }
//
//   free(Z1);
//   free(Z2);
//   free(Z3);
//
//   label(Z, n);
//
//   return PyTuple_Pack(2, Zarray, ZDarray);
// }
//
// py::array_t<double> cluster_by_distance(py::array_t<int> Z,
// py::array_t<double> dists) {
//     // maximum distance for each cluster
//   double *MD = malloc((n - 1) * sizeof(double));
//     std::vector<double> max_dist(n - 1);
//   int *N = malloc(n * sizeof(int));            // current nodes
//   uint8_t *V = calloc(n * 2, sizeof(uint8_t)); // visted nodes
//
//   double max;
//   int root, i, j, k = 0;
//   N[0] = 2 * n - 2;
//   while (k >= 0) {
//     root = N[k] - n;
//     i = Z[root * 3];
//     j = Z[root * 3 + 1];
//
//     if (i >= n && V[i] != 1) {
//       V[i] = 1;
//       N[++k] = i;
//       continue;
//     }
//     if (j >= n && V[j] != 1) {
//       V[j] = 1;
//       N[++k] = j;
//       continue;
//     }
//
//     max = ZD[root];
//
//     if (i >= n && MD[i - n] > max)
//       max = MD[i - n];
//     if (j >= n && MD[j - n] > max)
//       max = MD[j - n];
//     MD[root] = max;
//
//     k -= 1;
//   }
// }
// static PyObject *cluster_by_distance(PyObject *self, PyObject *args) {
//   PyObject *in[2];
//   PyArrayObject *Zarray, *ZDarray, *Tarray;
//   double cluster_dist;
//
//   if (!PyArg_ParseTuple(args, "OOd:cluster", &in[0], &in[1],
//   &cluster_dist))
//     return nullptr;
//   Zarray =
//       (PyArrayObject *)PyArray_FROM_OTF(in[0], NPY_INT,
//       NPY_ARRAY_IN_ARRAY);
//   if (!Zarray)
//     return nullptr;
//   if (PyArray_NDIM(Zarray) != 2 || PyArray_DIM(Zarray, 1) != 3) {
//     PyErr_SetString(PyExc_ValueError, "Z must be be of shape (n, 3).");
//     Py_DECREF(Zarray);
//     return nullptr;
//   }
//
//   ZDarray =
//       (PyArrayObject *)PyArray_FROM_OTF(in[1], NPY_DOUBLE,
//       NPY_ARRAY_IN_ARRAY);
//   if (!ZDarray) {
//     Py_DECREF(Zarray);
//     return nullptr;
//   }
//
//   int n = PyArray_DIM(Zarray, 0) + 1;
//
//   int *Z = (int *)PyArray_DATA(Zarray);
//   const double *ZD = (const double *)PyArray_DATA(ZDarray);
//
//   // Get the maximum distance for each cluster
//   double *MD = malloc((n - 1) * sizeof(double));
//   int *N = malloc(n * sizeof(int));            // current nodes
//   uint8_t *V = calloc(n * 2, sizeof(uint8_t)); // visted nodes
//
//   double max;
//   int root, i, j, k = 0;
//   N[0] = 2 * n - 2;
//   while (k >= 0) {
//     root = N[k] - n;
//     i = Z[root * 3];
//     j = Z[root * 3 + 1];
//
//     if (i >= n && V[i] != 1) {
//       V[i] = 1;
//       N[++k] = i;
//       continue;
//     }
//     if (j >= n && V[j] != 1) {
//       V[j] = 1;
//       N[++k] = j;
//       continue;
//     }
//
//     max = ZD[root];
//
//     if (i >= n && MD[i - n] > max)
//       max = MD[i - n];
//     if (j >= n && MD[j - n] > max)
//       max = MD[j - n];
//     MD[root] = max;
//
//     k -= 1;
//   }
//
//   // cluster nodes by distance
//   npy_intp dims[] = {n};
//   Tarray = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT, 0);
//   int *T = (int *)PyArray_DATA(Tarray);
//   memset(V, 0, n * 2 * sizeof(uint8_t));
//
//   int cluster_leader = -1, cluster_number = 0;
//
//   k = 0;
//   N[0] = 2 * n - 2;
//   while (k >= 0) {
//     root = N[k] - n;
//     i = Z[root * 3];
//     j = Z[root * 3 + 1];
//
//     if (cluster_leader == -1 && MD[root] <= cluster_dist) {
//       cluster_leader = root;
//       cluster_number += 1;
//     }
//
//     if (i >= n && V[i] != 1) {
//       V[i] = 1;
//       N[++k] = i;
//       continue;
//     }
//     if (j >= n && V[j] != 1) {
//       V[j] = 1;
//       N[++k] = j;
//       continue;
//     }
//     if (i < n) {
//       if (cluster_leader == -1)
//         cluster_number += 1;
//       T[i] = cluster_number;
//     }
//     if (j < n) {
//       if (cluster_leader == -1)
//         cluster_number += 1;
//       T[j] = cluster_number;
//     }
//     if (cluster_leader == root)
//       cluster_leader = -1;
//     k -= 1;
//   }
//
//   free(MD);
//   free(N);
//   free(V);
//
//   Py_DECREF(Zarray);
//   Py_DECREF(ZDarray);
//
//   return (PyObject *)Tarray;
// }
//
//

py::array_t<int> maxima(py::array_t<double> values, py::array_t<int> regions) {
  py::buffer_info vbuf = values.request(), rbuf = regions.request();

  if (vbuf.ndim != 1)
    throw std::runtime_error("values must have 1 dim");
  if (rbuf.ndim != 2 || rbuf.shape[1] != 2)
    throw std::runtime_error("regions must have shape (n, 2)");

  auto argmax = py::array_t<int>(rbuf.shape[0]);
  py::buffer_info mbuf = argmax.request();

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

void tester(py::array_t<double> x) {
  std::cout << x.size() << std::endl;
  for (auto &y : x) {
    std::cout << "[ ";
    for (auto &z : y) {
      std::cout << z.cast<double>() << ", " << std::endl;
    }
    std::cout << "]" << std::endl;
  }
}

PYBIND11_MODULE(spcalext, mod) {
  mod.doc() = "extension module for SPCal.";

  mod.def("tester", &tester, "test");
  mod.def("pairwise_euclidean", &pairwise_euclidean,
          "Calculates the euclidean distance for an array");
  // mod.def("mst_linkage", &mst_linkage,
  //         "Return the minimum-spanning-tree linkage.");
  // mod.def("cluster_by_distance", &cluster_by_distance,
  //         "Cluster using MST linkage.");
  mod.def("maxima", &maxima,
          "Calculates to maxima between pairs of start and end positions.");
}
