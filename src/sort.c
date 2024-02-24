#include "sort.h"
#include <stdlib.h>
#include <string.h>

void merge_argsort(struct argsort *x, int n, struct argsort *t) {
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

void mergesort_argsort_rec(struct argsort *x, int n, struct argsort *t) {
  if (n < 2)
    return;
#pragma omp task shared(x) if (n > 1000)
  mergesort_argsort_rec(x, n / 2, t);
#pragma omp task shared(x) if (n > 1000)
  mergesort_argsort_rec(x + n / 2, n - n / 2, t + n / 2);
#pragma omp taskwait
  merge_argsort(x, n, t);
}

void mergesort_argsort(struct argsort *x, int n) {
  struct argsort *t = malloc(n * sizeof(struct argsort));
#pragma omp parallel
  {
#pragma omp single
    mergesort_argsort_rec(x, n, t);
  }
  free(t);
}

int partition_argsort(struct argsort *x, int low, int high) {
  double pivot = x[low].value;
  struct argsort t;
  int i = low - 1;

  for (size_t j = low; j <= high; ++j) {
    if (x[j].value < pivot) {
      i++;
      t = x[i];
      x[i] = x[j];
      x[j] = t;
    }
  }
  i++;
  t = x[i];
  x[i] = x[high];
  x[high] = t;
  return i + 1;
}

void quicksort_argsort_rec(struct argsort *x, int low, int high) {
  if (low < high) {
    int pivot = partition_argsort(x, low, high);
#pragma omp task shared(x) if ((high - low) > 1000)
    quicksort_argsort_rec(x, low, pivot - 1);
#pragma omp task shared(x) if ((high - low) > 1000)
    quicksort_argsort_rec(x, pivot + 1, high);
  }
}

void quicksort_argsort(struct argsort *x, int n) {
#pragma omp parallel
  {
#pragma omp single
    quicksort_argsort_rec(x, 0, n - 1);
  }
}
