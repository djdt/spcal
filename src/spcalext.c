#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

inline double sqeclidean(PyArrayObject *X, npy_intp u, npy_intp v, npy_intp n) {
  double sum = 0.0;
  for (npy_intp i = 0; i < n; ++i) {
    double dist = *(double *)PyArray_GETPTR2(X, u, i) -
                  *(double *)PyArray_GETPTR2(X, v, i);
    sum += dist * dist;
  }
  return sum;
}

static PyObject *pdist(PyObject *self, PyObject *args) {
  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!:pdist", &PyArray_Type, &X))
    return NULL;

  npy_intp n = PyArray_DIM(X, 0);
  npy_intp m = PyArray_DIM(X, 1);

  npy_intp dims[] = {n * (n - 1) / 2};
  PyArrayObject *dists =
      (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double *D = (double *)PyArray_DATA(dists);

  npy_intp i, j, k = 0;
  NPY_BEGIN_ALLOW_THREADS;
  for (i = 0; i < n; ++i) {
    for (j = i + 1; j < n; ++j, k++) {
      D[k] = sqeclidean(X, i, j, m);
    }
  }
  NPY_END_ALLOW_THREADS;
  return (PyObject *)dists;
}

inline npy_intp cindex(npy_intp n, npy_intp x, npy_intp y) {
  if (x < y)
    return n * x - (x * (x + 1) / 2) + (y - x - 1);
  else
    return n * y - (y * (y + 1) / 2) + (x - y - 1);
}

static PyObject *single_linkage(PyObject *self, PyObject *args) {
  PyArrayObject *_D;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &_D))
    return NULL;

  npy_intp n = PyArray_DIM(_D, 0);

  uint32_t *Z1 = malloc((n - 1) * sizeof(uint32_t));
  uint32_t *Z2 = malloc((n - 1) * sizeof(uint32_t));
  double *Z3 = malloc((n - 1) * sizeof(double));

  double *D = malloc(n * sizeof(double));
  for (int i = 0; i < n; ++i) {
    D[i] = INFINITY;
  }
  uint8_t *M = calloc(n, sizeof(uint8_t));

  npy_intp x, y = 0;
  double min, dist;
  for (npy_intp i = 0; i < n - 1; ++i) {
    min = INFINITY;
    M[x] = 1;
    for (npy_intp j = 0; j < n; ++j) {
      if (M[i] == 1)
        continue;
      dist = *(double *)PyArray_GETPTR1(_D, cindex(n, x, j));

      if (D[i] > dist)
        D[i] = dist;
      if (D[i] < min) {
        y = j;
        min = D[i];
      }
    }
    Z1[i] = x;
    Z2[i] = y;
    Z3[i] = min;
    x = y;
  }

  uint32_t roots = m

      for (uint32_t i = 0; i < n - 1; ++i) {}

  free(D);
  free(M);

  Py_DecRef(&in);
}

static PyMethodDef spcal_methods[] = {
    /* {"hierarchical_spcal", spcal_linkage, METH_VARARGS, */
    /*  "Perform aglomerative hierarchical spcaling."}, */
    {"pdist", pdist, METH_VARARGS, "Calculate pairwise distance for array."},
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
