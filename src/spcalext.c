#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

inline double sqeclidean(const double *X, int i, int j, int m) {
  double sum = 0.0;
  for (int k = 0; k < m; ++k) {
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

  int n = PyArray_DIM(Xarray, 0);
  int m = PyArray_DIM(Xarray, 1);

  npy_intp dims[] = {n * (n - 1) / 2};
  Darray = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  const double *X = (const double *)PyArray_DATA(Xarray);
  double *D = (double *)PyArray_DATA(Darray);

  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j, ++k) {
      D[k] = sqeclidean(X, i, j, m);
    }
  }
  return (PyObject *)Darray;
}

inline int condensed_index(int i, int j, int n) {
  if (i < j)
    return n * i - (i * (i + 1) / 2) + (j - i - 1);
  else
    return n * j - (j * (j + 1) / 2) + (i - j - 1);
}

struct argsort {
  double value;
  int index;
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

void label(PyArrayObject *Zarray, int n) {
  int *Z = (int *)PyArray_DATA(Zarray);

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

static PyObject *mst_linkage(PyObject *self, PyObject *args) {
  PyArrayObject *PDarray;
  int n;

  if (!PyArg_ParseTuple(args, "O!i:mst_linkage", &PyArray_Type, &PDarray, &n))
    return NULL;
  if (!PyArray_Check(PDarray))
    return NULL;

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
  double dist, min;
  for (int i = 0; i < n - 1; ++i) {
    min = INFINITY;
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

  // Sort
  qsort(Z3, n - 1, sizeof(Z3[0]), argsort_cmp);

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

  label(Zarray, n);

  return PyTuple_Pack(2, Zarray, ZDarray);
}

static PyObject *cluster(PyObject *self, PyObject *args) {
  PyArrayObject *Zarray, *ZDarray;
  double dist;
  int n;

  if (!PyArg_ParseTuple(args, "O!O!di:mst_linkage", &PyArray_Type, &Zarray,
                        &PyArray_Type, &ZDarray, &dist, &n))
    return NULL;
  if (!PyArray_Check(Zarray))
    return NULL;
  if (!PyArray_Check(ZDarray))
    return NULL;

  // Get the maximum distance for each cluster
}

static PyMethodDef spcal_methods[] = {
    /* {"hierarchical_spcal", spcal_linkage, METH_VARARGS, */
    /*  "Perform aglomerative hierarchical spcaling."}, */
    {"pdist_square", pdist_square, METH_VARARGS,
     "Calculate squared euclidean pairwise distance for array."},
    {"mst_linkage", mst_linkage, METH_VARARGS,
     "Return the minimum spanning tree linkage."},
    {"cluster", cluster, METH_VARARGS, "Cluster using the MST linkage."},
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
