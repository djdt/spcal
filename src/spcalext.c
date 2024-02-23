#define PY_SSIZE_T_CLEAN
#define PI 3.14159265358979323846
#define SQ2PI sqrt(2.0 * PI)

#include <Python.h>
#include <numpy/arrayobject.h>

/* Based off of the scipy implementation
 * https://github.com/scipy/scipy/blob/v1.9.3/scipy/cluster/_hierarchy.pyx */

inline double euclidean(const double *X, npy_intp i, npy_intp j, npy_intp m) {
  double sum = 0.0;
  for (npy_intp k = 0; k < m; ++k) {
    double dist = X[i * m + k] - X[j * m + k];
    sum += dist * dist;
  }
  return sqrt(sum);
}

inline int condensed_index(int i, int j, int n) {
  if (i < j)
    return n * i - (i * (i + 1) / 2) + (j - i - 1);
  else
    return n * j - (j * (j + 1) / 2) + (i - j - 1);
}

static PyObject *pairwise_euclidean(PyObject *self, PyObject *args) {
  PyObject *in;
  PyArrayObject *Xarray, *Darray;

  if (!PyArg_ParseTuple(args, "O:pdist", &in))
    return NULL;
  Xarray =
      (PyArrayObject *)PyArray_FROM_OTF(in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!Xarray) {
    return NULL;
  }
  if (PyArray_NDIM(Xarray) != 2) {
    PyErr_SetString(PyExc_ValueError, "array must be 2 dimensional.");
    Py_DECREF(Xarray);
    return NULL;
  }

  npy_intp n = PyArray_DIM(Xarray, 0);
  npy_intp m = PyArray_DIM(Xarray, 1);

  npy_intp dims[] = {n * (n - 1) / 2};
  Darray = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (!Darray) {
    PyErr_Format(PyExc_ValueError,
                 "unable to allocate %ld bytes for dist array.",
                 dims[0] * sizeof(double));
    Py_DECREF(Xarray);
    return NULL;
  }

  const double *X = (const double *)PyArray_DATA(Xarray);
  double *D = (double *)PyArray_DATA(Darray);

#pragma omp parallel for
  for (npy_intp i = 0; i < n; ++i) {
    for (npy_intp j = i + 1; j < n; ++j) {
      D[condensed_index(i, j, n)] = euclidean(X, i, j, m);
    }
  }
  Py_DECREF(Xarray);
  return (PyObject *)Darray;
}

struct argsort {
  double value;
  int index;
};

int argsort_cmp(const void *a, const void *b) {
  struct argsort *as = (struct argsort *)a;
  struct argsort *bs = (struct argsort *)b;
  if (as->value > bs->value)
    return 1;
  else if (as->value < bs->value)
    return -1;
  else
    return 0;
}

inline int find_root(int *parents, int x) {
  int p = x;
  while (parents[x] != x)
    x = parents[x];

  while (parents[p] != x) {
    p = parents[p];
    parents[p] = x;
  }
  return x;
}

inline int merge_roots(int *parents, int *sizes, int n, int x, npy_int y) {
  int size = sizes[x] + sizes[y];
  sizes[n] = size;
  parents[x] = n;
  parents[y] = n;
  return size;
}

void label(int *Z, int n) {
  int *parents = malloc((2 * n - 1) * sizeof(int));
  int *sizes = malloc((2 * n - 1) * sizeof(int));
  int next = n;
  int x, y, x_root, y_root;
  for (int i = 0; i < 2 * n - 1; ++i) {
    parents[i] = i;
    sizes[i] = 1;
  }

  for (npy_intp i = 0; i < n - 1; ++i) {
    x = Z[i * 3];
    y = Z[i * 3 + 1];
    x_root = find_root(parents, x);
    y_root = find_root(parents, y);
    if (x_root < y_root) {
      Z[i * 3] = x_root;
      Z[i * 3 + 1] = y_root;
    } else {
      Z[i * 3] = y_root;
      Z[i * 3 + 1] = x_root;
    }
    Z[i * 3 + 2] = merge_roots(parents, sizes, next, x_root, y_root);
    next += 1;
  }

  free(parents);
  free(sizes);
}

void _merge_argsort(struct argsort *x, int n, struct argsort *t) {
  int i = 0, j = n / 2, ti = 0;

  while (i < n / 2 && j < n) {
    if (x[i].value < x[j].value) {
      t[ti++] = x[i++];
    } else {
      t[ti++] = x[j++];
    }
  }
  while (i < n / 2) {
    t[ti++] = x[i++];
  }
  while (j < n) {
    t[ti++] = x[j++];
  }
  memcpy(x, t, n * sizeof(struct argsort));
}

void _mergesort_argsort_rec(struct argsort *x, int n, struct argsort *t) {
  if (n < 2)
    return;
#pragma omp task shared(x) if (n > 1000)
  _mergesort_argsort_rec(x, n / 2, t);
#pragma omp task shared(x) if (n > 1000)
  _mergesort_argsort_rec(x + n / 2, n - n / 2, t + n / 2);
#pragma omp taskwait
  _merge_argsort(x, n, t);
}
void mergesort_argsort(struct argsort *x, int n) {
  struct argsort *t = malloc(n * sizeof(struct argsort));
#pragma omp parallel
  {
#pragma omp single
    _mergesort_argsort_rec(x, n, t);
  }
  free(t);
}

static PyObject *mst_linkage(PyObject *self, PyObject *args) {
  PyObject *in;
  PyArrayObject *PDarray;
  int n;

  if (!PyArg_ParseTuple(args, "Oi:mst_linkage", &in, &n))
    return NULL;

  PDarray =
      (PyArrayObject *)PyArray_FROM_OTF(in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!PDarray) {
    return NULL;
  }

  const double *PD = (const double *)PyArray_DATA(PDarray);
  int *Z1 = malloc((n - 1) * sizeof(int));
  int *Z2 = malloc((n - 1) * sizeof(int));
  struct argsort *Z3 = malloc((n - 1) * sizeof(struct argsort));

  uint8_t *M = calloc(n, sizeof(uint8_t));
  double *D = malloc(n * sizeof(double));

  // We use Z[:, 2] as M, tracking merged
  // Init arrays (ZD = 0), D = inf
  for (npy_intp i = 0; i < n - 1; ++i) {
    D[i] = INFINITY;
    Z3[i].index = i;
  }
  D[n - 1] = INFINITY;

  int x = 0, y = 0;
  for (int i = 0; i < n - 1; ++i) {
    double min = INFINITY;
    double dist;
    M[x] = 1;

    for (int j = 0; j < n; ++j) {
      if (M[j] == 1)
        continue;

      dist = PD[condensed_index(x, j, n)];

      if (D[j] > dist)
        D[j] = dist;
      if (D[j] < min) {
        y = j;
        min = D[j];
      }
    }

    Z1[i] = x;
    Z2[i] = y;
    Z3[i].value = min;
    x = y;
  }

  free(M);
  free(D);
  Py_DECREF(PDarray);

  // Sort
  mergesort_argsort(Z3, n - 1);

  PyArrayObject *Zarray, *ZDarray;
  npy_intp Zdims[] = {n - 1, 3};
  npy_intp ZDdims[] = {n - 1};
  Zarray = (PyArrayObject *)PyArray_SimpleNew(2, Zdims, NPY_INT);
  ZDarray = (PyArrayObject *)PyArray_SimpleNew(1, ZDdims, NPY_DOUBLE);

  int *Z = (int *)PyArray_DATA(Zarray);
  double *ZD = (double *)PyArray_DATA(ZDarray);

  for (int i = 0; i < n - 1; ++i) {
    Z[i * 3] = Z1[Z3[i].index];
    Z[i * 3 + 1] = Z2[Z3[i].index];
    ZD[i] = Z3[i].value;
  }

  free(Z1);
  free(Z2);
  free(Z3);

  label(Z, n);

  return PyTuple_Pack(2, Zarray, ZDarray);
}

static PyObject *cluster_by_distance(PyObject *self, PyObject *args) {
  PyObject *in[2];
  PyArrayObject *Zarray, *ZDarray, *Tarray;
  double cluster_dist;

  if (!PyArg_ParseTuple(args, "OOd:cluster", &in[0], &in[1], &cluster_dist))
    return NULL;
  Zarray =
      (PyArrayObject *)PyArray_FROM_OTF(in[0], NPY_INT, NPY_ARRAY_IN_ARRAY);
  if (!Zarray)
    return NULL;
  if (PyArray_NDIM(Zarray) != 2 || PyArray_DIM(Zarray, 1) != 3) {
    PyErr_SetString(PyExc_ValueError, "Z must be be of shape (n, 3).");
    Py_DECREF(Zarray);
    return NULL;
  }

  ZDarray =
      (PyArrayObject *)PyArray_FROM_OTF(in[1], NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!ZDarray) {
    Py_DECREF(Zarray);
    return NULL;
  }

  int n = PyArray_DIM(Zarray, 0) + 1;

  int *Z = (int *)PyArray_DATA(Zarray);
  const double *ZD = (const double *)PyArray_DATA(ZDarray);

  // Get the maximum distance for each cluster
  double *MD = malloc((n - 1) * sizeof(double));
  int *N = malloc(n * sizeof(int));            // current nodes
  uint8_t *V = calloc(n * 2, sizeof(uint8_t)); // visted nodes

  double max;
  int root, i, j, k = 0;
  N[0] = 2 * n - 2;
  while (k >= 0) {
    root = N[k] - n;
    i = Z[root * 3];
    j = Z[root * 3 + 1];

    if (i >= n && V[i] != 1) {
      V[i] = 1;
      N[++k] = i;
      continue;
    }
    if (j >= n && V[j] != 1) {
      V[j] = 1;
      N[++k] = j;
      continue;
    }

    max = ZD[root];

    if (i >= n && MD[i - n] > max)
      max = MD[i - n];
    if (j >= n && MD[j - n] > max)
      max = MD[j - n];
    MD[root] = max;

    k -= 1;
  }

  // cluster nodes by distance
  npy_intp dims[] = {n};
  Tarray = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT, 0);
  int *T = (int *)PyArray_DATA(Tarray);
  memset(V, 0, n * 2 * sizeof(uint8_t));

  int cluster_leader = -1, cluster_number = 0;

  k = 0;
  N[0] = 2 * n - 2;
  while (k >= 0) {
    root = N[k] - n;
    i = Z[root * 3];
    j = Z[root * 3 + 1];

    if (cluster_leader == -1 && MD[root] <= cluster_dist) {
      cluster_leader = root;
      cluster_number += 1;
    }

    if (i >= n && V[i] != 1) {
      V[i] = 1;
      N[++k] = i;
      continue;
    }
    if (j >= n && V[j] != 1) {
      V[j] = 1;
      N[++k] = j;
      continue;
    }
    if (i < n) {
      if (cluster_leader == -1)
        cluster_number += 1;
      T[i] = cluster_number;
    }
    if (j < n) {
      if (cluster_leader == -1)
        cluster_number += 1;
      T[j] = cluster_number;
    }
    if (cluster_leader == root)
      cluster_leader = -1;
    k -= 1;
  }

  free(MD);
  free(N);
  free(V);

  Py_DECREF(Zarray);
  Py_DECREF(ZDarray);

  return (PyObject *)Tarray;
}

static PyObject *maxima(PyObject *self, PyObject *args) {
  PyObject *in[2];
  PyArrayObject *Xarray, *Iarray;

  if (!PyArg_ParseTuple(args, "OO", &in[0], &in[1]))
    return NULL;

  Xarray =
      (PyArrayObject *)PyArray_FROM_OTF(in[0], NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!Xarray)
    return NULL;
  Iarray =
      (PyArrayObject *)PyArray_FROM_OTF(in[1], NPY_INTP, NPY_ARRAY_IN_ARRAY);

  if (!Iarray) {
    Py_DECREF(Xarray);
    return NULL;
  }

  npy_intp n = PyArray_SIZE(Iarray) / 2;
  npy_intp dims[] = {n};
  PyArrayObject *Rarray = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT);

  const double *X = (const double *)PyArray_DATA(Xarray);
  const npy_intp *I = (const npy_intp *)PyArray_DATA(Iarray);

  int *R = (int *)PyArray_DATA(Rarray);

  int argmax;
  for (int i = 0; i < n; ++i) {
    argmax = I[i * 2];

    for (int j = I[i * 2]; j < I[i * 2 + 1]; ++j) {
      if (X[j] > X[argmax]) {
        argmax = j;
      }
    }
    R[i] = argmax;
  }
  Py_DECREF(Xarray);
  Py_DECREF(Iarray);

  return (PyObject *)Rarray;
}

double *normal_pdf(const double *x, double mu, double sigma, int size) {
  double *pdf = malloc(sizeof(double) * size);
  for (npy_intp i = 0; i < size; ++i) {
    pdf[i] = 1.0 / (sigma * SQ2PI) * exp(-0.5 * pow((x[i] - mu) / sigma, 2));
  }
  return pdf;
}

double *lognormal_pdf(const double *x, double mu, double sigma, int size) {
  double *pdf = malloc(sizeof(double) * size);
  for (npy_intp i = 0; i < size; ++i) {
    pdf[i] = 1.0 / (x[i] * sigma * SQ2PI) *
             exp(-0.5 * pow((log(x[i]) - mu) / sigma, 2));
  }
  return pdf;
}

static PyMethodDef spcal_methods[] = {
    // Clustering
    {"pairwise_euclidean", pairwise_euclidean, METH_VARARGS,
     "Calculate euclidean pairwise distance for array."},
    {"mst_linkage", mst_linkage, METH_VARARGS,
     "Return the minimum spanning tree linkage."},
    {"cluster_by_distance", cluster_by_distance, METH_VARARGS,
     "Cluster using the MST linkage."},
    // Other
    {"maxima", maxima, METH_VARARGS,
     "Calculates maxima between pairs of start and end positions."},
    // Fitting
    /* {"fit_normal", fit_normal, METH_VARARGS, "Fit a normal pdf to the
       input."}, */
    /* {"fit_lognormal", fit_lognormal, METH_VARARGS, */
    /*  "Fit a lognormal pdf to the input."}, */
    /* {"nelder_mead", nelder_mead, METH_VARARGS, */
    /*  "Find the minima of a function."}, */
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef spcal_module = {PyModuleDef_HEAD_INIT, "spcal_module",
                                          "Extension module for SPCal.", -1,
                                          spcal_methods};

PyMODINIT_FUNC PyInit_spcalext(void) {
  PyObject *m;
  m = PyModule_Create(&spcal_module);
  import_array();
  if (PyErr_Occurred())
    return NULL;
  return m;
}
