#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

double sqeclidean(PyArrayObject *X, npy_intp u, npy_intp v, npy_intp n) {
  double sum = 0.0;
  for (npy_intp i = 0; i < n; ++i) {
    double dist = *(double *)PyArray_GETPTR2(X, u, i) -
                  *(double *)PyArray_GETPTR2(X, v, i);
    sum += dist * dist;
  }
  return sum;
}

static PyObject *pdist(PyObject *self, PyObject *args) {
  PyObject in;
  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O", &in))
    return NULL;

  X = (PyArrayObject *)PyArray_FROM_OTF(&in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (X == NULL)
    return NULL;

  npy_intp n = PyArray_DIM(X, 0);
  npy_intp m = PyArray_DIM(X, 1);

    npy_intp dims[] = {n, m}
  PyArrayObject *dists = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  npy_intp i, j;
  for (i = 0; i < n; ++i) {
    for (j = i + 1; j < m; ++j) {
      dist = sqeclidean(X, i, v);
    }
  }
}

uint8_t polygonf_contains_point(PyArrayObject *poly, double x, double y) {
  npy_intp n = PyArray_DIM(poly, 0);

  if (n < 3)
    return 0;

  double px, py, lx, ly;
  uint8_t flag, lflag;
  uint8_t inside = 0;

  lx = *(double *)PyArray_GETPTR2(poly, n - 1, 0);
  ly = *(double *)PyArray_GETPTR2(poly, n - 1, 1);
  lflag = ly >= y;

  for (npy_intp i = 0; i < n; ++i) {
    px = *(double *)PyArray_GETPTR2(poly, i, 0);
    py = *(double *)PyArray_GETPTR2(poly, i, 1);
    flag = py >= y;

    if (lflag != flag) {
      if (((py - y) * (lx - px) >= (px - x) * (ly - py)) == flag) {
        inside ^= 1;
      }
    }

    lx = px;
    ly = py;
    lflag = flag;
  }

  return inside;
}

inline npy_intp cindex(npy_intp n, npy_intp x, npy_intp y) {
  if (x < y)
    return n * x - (x * (x + 1) / 2) + (y - x - 1);
  else
    return n * y - (y * (y + 1) / 2) + (x - y - 1);
}

static PyObject *single_linkage(PyObject *self, PyObject *args) {
  PyObject in;
  PyArrayObject *dists;

  if (!PyArg_ParseTuple(args, "O", &in))
    return NULL;

  dists =
      (PyArrayObject *)PyArray_FROM_OTF(&in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (dists == NULL)
    return NULL;

  npy_intp n = PyArray_DIM(dists, 0);

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
      dist = *(double *)PyArray_GETPTR1(dists, cindex(n, x, j));

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

static PyObject *polyext_polygonf_contains_points(PyObject *self,
                                                  PyObject *args) {
  PyObject *in[2];
  PyArrayObject *poly, *points;

  if (!PyArg_ParseTuple(args, "OO", &in[0], &in[1]))
    return NULL;

  poly =
      (PyArrayObject *)PyArray_FROM_OTF(in[0], NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (poly == NULL)
    return NULL;
  points =
      (PyArrayObject *)PyArray_FROM_OTF(in[1], NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (points == NULL) {
    Py_XDECREF(in[0]);
    return NULL;
  }

  PyArrayObject *res;
  npy_intp dims[] = {PyArray_DIM(points, 0)};
  res = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);

  npy_intp n = PyArray_DIM(points, 0);

  for (npy_intp i = 0; i < n; ++i) {
    uint8_t *out = (uint8_t *)PyArray_GETPTR1(res, i);
    *out =
        polygonf_contains_point(poly, *(double *)PyArray_GETPTR2(points, i, 0),
                                *(double *)PyArray_GETPTR2(points, i, 1));
  }

  Py_DECREF(in[0]);
  Py_DECREF(in[1]);

  return (PyObject *)res;
}

static PyMethodDef polyext_methods[] = {
    /* { "polygon_contains_points", */
    /*     polyext_polygon_contains_points, */
    /*     METH_VARARGS, */
    /*     "Check if multiple points are within a int type polygon." }, */
    {"hierarchical_cluster", cluster_linkage, METH_VARARGS,
     "Perform aglomerative hierarchical clustering."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef custermodeule = {
    PyModuleDef_HEAD_INIT, "cluster_module", "Clustering extension modulde.",
    -1, polyext_methods};

PyMODINIT_FUNC PyInit_polyext(void) {
  PyObject *m;
  m = PyModule_Create(&clustermodule);
  import_array();
  if (PyErr_Occurred())
    return NULL;
  return m;
}
