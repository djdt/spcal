#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

inline double sqeclidean(const double *X, npy_intp i, npy_intp j, npy_intp m) {
  double sum = 0.0;
  for (npy_intp k = 0; k < m; ++k) {
    double dist = X[i * m + k] - X[j * m + k];
    sum += dist * dist;
  }
  return sum;
}

static PyObject *pdist_square(PyObject *self, PyObject *args) {
  PyArrayObject *Xarray, *Darray;

  if (!PyArg_ParseTuple(args, "O!:pdist", &PyArray_Type, &Xarray))
    return NULL;
  if (!PyArray_Check(Xarray))
    return NULL;

  npy_intp n = PyArray_DIM(Xarray, 0);
  npy_intp m = PyArray_DIM(Xarray, 1);

  npy_intp dims[] = {n * (n - 1) / 2};
  Darray = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  const double *X = (const double *)PyArray_DATA(Xarray);
  double *D = (double *)PyArray_DATA(Darray);

  npy_intp i, j, k = 0;
  for (i = 0; i < n; ++i) {
    for (j = i + 1; j < n; ++j, ++k) {
      D[k] = sqeclidean(X, i, j, m);
    }
  }
  return (PyObject *)Darray;
}

inline npy_intp condensed_index(npy_intp i, npy_intp j, npy_intp n) {
  if (i < j)
    return n * i - (i * (i + 1) / 2) + (j - i - 1);
  else
    return n * j - (j * (j + 1) / 2) + (i - j - 1);
}

struct argsort {
  double value;
  npy_intp index;
};

int argsort_cmp(const void *a, const void *b) {
  struct argsort *as = (struct argsort *)a;
  struct argsort *bs = (struct argsort *)b;
  if ((*as).value > (*bs).value)
    return 1;
  else if ((*as).value < (*bs).value)
    return -1;
  else
    return 0;
}

inline npy_intp find_root(npy_intp *parents, npy_intp x) {
  npy_intp p = x;
  while (parents[x] != x)
    x = parents[x];

  while (parents[p] != x) {
    p = parents[p];
    parents[p] = x;
  }
  return x;
}

inline npy_intp merge_roots(npy_intp *parents, npy_intp *sizes, npy_intp n,
                            npy_intp x, npy_intp y) {
  npy_intp size = sizes[x] + sizes[y];
  sizes[n] = size;
  parents[x] = n;
  parents[y] = n;
  return size;
}

void label(PyArrayObject *Zarray, npy_intp n) {
  npy_intp *Z = (npy_intp *)PyArray_DATA(Zarray);

  npy_intp *parents = malloc((2 * n - 1) * sizeof(npy_intp));
  npy_intp *sizes = malloc((2 * n - 1) * sizeof(npy_intp));
  npy_intp next = n;
  npy_intp x, y, x_root, y_root;
  for (npy_intp i = 0; i < 2 * n - 1; ++i) {
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

static PyObject *mst_linkage(PyObject *self, PyObject *args) {
  PyArrayObject *PDarray;
  npy_intp n;

  if (!PyArg_ParseTuple(args, "O!n:mst_linkage", &PyArray_Type, &PDarray, &n))
    return NULL;
  if (!PyArray_Check(PDarray))
    return NULL;

  const double *PD = (const double *)PyArray_DATA(PDarray);
  npy_intp *Z1 = malloc((n - 1) * sizeof(npy_intp));
  npy_intp *Z2 = malloc((n - 1) * sizeof(npy_intp));
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

  npy_intp x = 0, y = 0;
  double dist, min;
  for (npy_intp i = 0; i < n - 1; ++i) {
    min = INFINITY;
    M[x] = 1;

    for (npy_intp j = 0; j < n; ++j) {
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

  // Sort
  qsort(Z3, n - 1, sizeof(Z3[0]), argsort_cmp);

  PyArrayObject *Zarray, *ZDarray;
  npy_intp Zdims[] = {n - 1, 3};
  npy_intp ZDdims[] = {n - 1};
  Zarray = (PyArrayObject *)PyArray_SimpleNew(2, Zdims, NPY_LONG);
  ZDarray = (PyArrayObject *)PyArray_SimpleNew(1, ZDdims, NPY_DOUBLE);

  npy_intp *Z = (npy_intp *)PyArray_DATA(Zarray);
  double *ZD = (double *)PyArray_DATA(ZDarray);

  for (npy_intp i = 0; i < n - 1; ++i) {
    Z[i * 3] = Z1[Z3[i].index];
    Z[i * 3 + 1] = Z2[Z3[i].index];
    ZD[i] = Z3[i].value;
  }

  free(Z1);
  free(Z2);
  free(Z3);

  label(Zarray, n);

  return PyTuple_Pack(2, Zarray, ZDarray);
}

static PyMethodDef spcal_methods[] = {
    /* {"hierarchical_spcal", spcal_linkage, METH_VARARGS, */
    /*  "Perform aglomerative hierarchical spcaling."}, */
    {"pdist_square", pdist_square, METH_VARARGS,
     "Calculate squared euclidean pairwise distance for array."},
    {"mst_linkage", mst_linkage, METH_VARARGS,
     "Return the minimum spanning tree linkage."},
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
