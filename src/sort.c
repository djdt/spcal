#include "sort.h"
#include <string.h>

void merge_argsort(struct argsort *x, size_t n, struct argsort *t) {
  size_t i = 0, j = n / 2, ti = 0;

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

void mergesort_argsort_rec(struct argsort *x, size_t n, struct argsort *t) {
  if (n < 2)
    return;
#pragma omp task shared(x) if (n > 1000)
  mergesort_argsort_rec(x, n / 2, t);
#pragma omp task shared(x) if (n > 1000)
  mergesort_argsort_rec(x + n / 2, n - n / 2, t + n / 2);
#pragma omp taskwait
  merge_argsort(x, n, t);
}

void mergesort_argsort(struct argsort *x, size_t n) {
  struct argsort *t = malloc(n * sizeof(struct argsort));
#pragma omp parallel
  {
#pragma omp single
    mergesort_argsort_rec(x, n, t);
  }
  free(t);
}

long partition_argsort(struct argsort *x, size_t low, size_t high) {
  double pivot = x[low].value;
  struct argsort t;

  long i = low - 1;
  long j = high + 1;

  while (1) {
    while (x[++i].value < pivot)
      ;
    while (x[--j].value > pivot)
      ;
    if (i >= j) {
      return j;
    }
    t = x[i];
    x[i] = x[j];
    x[j] = t;
  }
}

void quicksort_argsort_rec(struct argsort *x, size_t low, size_t high) {
  if (low < high) {
    size_t pivot = partition_argsort(x, low, high);
#pragma omp task shared(x) if ((high - low) > 1000)
    quicksort_argsort_rec(x, low, pivot);
#pragma omp task shared(x) if ((high - low) > 1000)
    quicksort_argsort_rec(x, pivot + 1, high);
  }
}

void quicksort_argsort(struct argsort *x, size_t n) {
#pragma omp parallel
  {
#pragma omp single
    quicksort_argsort_rec(x, 0, n - 1);
  }
}
